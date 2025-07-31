"""
Hybrid indexing implementation for Enhanced RAG system.
Combines dense, sparse, and ColBERT embeddings for optimal retrieval.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import contextlib
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    from .mock_components import MockSentenceTransformer as SentenceTransformer
    
from .embeddings import EmbeddingManager, EmbeddingProvider, TaskType
    
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    
try:
    import faiss
except ImportError:
    from .mock_components import MockFAISS as faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import logging
from datetime import datetime
from dataclasses import dataclass

from ..config import settings

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Document representation for indexing"""
    id: str
    text: str
    metadata: Dict[str, Any]
    embeddings: Optional[np.ndarray] = None


class HybridIndexer:
    """Implements hybrid indexing with dense, sparse, and ColBERT embeddings"""
    
    def __init__(self, settings_obj: Optional[Any] = None):
        self.settings = settings_obj or settings
        self.embedding_manager = EmbeddingManager()
        self.setup_models()
        self.indices = {}
        
    def setup_models(self):
        """Initialize embedding models"""
        logger.info("Setting up indexing models...")
        
        # Dense embedding model
        self.dense_model = SentenceTransformer(
            f"sentence-transformers/all-MiniLM-L6-v2"  # Using a reliable model
        )
        
        # Sparse embedding components
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95
        )
        
        # ColBERT model if enabled
        if self.settings.enable_colbert and TRANSFORMERS_AVAILABLE:
            try:
                self.colbert_tokenizer = AutoTokenizer.from_pretrained(
                    "bert-base-uncased"  # Using BERT as fallback
                )
                self.colbert_model = AutoModel.from_pretrained(
                    "bert-base-uncased"
                )
                logger.info("ColBERT models loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load ColBERT models: {e}")
                self.settings.enable_colbert = False
        elif self.settings.enable_colbert:
            logger.warning("Transformers not available, disabling ColBERT")
            self.settings.enable_colbert = False
                
    def create_hybrid_index(
        self,
        corpus_id: str,
        dimension: int = 384  # MiniLM dimension
    ) -> Dict[str, Any]:
        """Create hybrid index structure"""
        
        indices = {
            "corpus_id": corpus_id,
            "dense_index": None,
            "sparse_index": None,
            "colbert_index": None,
            "metadata": {
                "dimension": dimension,
                "created_at": datetime.utcnow().isoformat(),
                "num_documents": 0
            }
        }
        
        # Dense index (FAISS)
        if self.settings.indexing_strategy in ["dense", "hybrid"]:
            indices["dense_index"] = faiss.IndexFlatIP(dimension)
            if self.settings.enable_clustering:
                # Add IVF for large-scale indexing
                nlist = 100
                quantizer = faiss.IndexFlatIP(dimension)
                indices["dense_index"] = faiss.IndexIVFFlat(
                    quantizer, dimension, nlist
                )
                
        # Sparse index structure
        if self.settings.indexing_strategy in ["sparse", "hybrid"]:
            indices["sparse_index"] = {
                "vectors": None,  # Will store scipy sparse matrix
                "vocabulary": None,
                "idf_weights": None
            }
            
        # ColBERT index
        if self.settings.enable_colbert:
            indices["colbert_index"] = {
                "token_embeddings": [],
                "doc_boundaries": [0],
                "token_to_doc_mapping": {}
            }
            
        self.indices[corpus_id] = indices
        logger.info(f"Created hybrid index for corpus: {corpus_id}")
        return indices
        
    def generate_dense_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """Generate dense embeddings for texts"""
        
        # Use embedding manager for flexibility
        embeddings = self.embedding_manager.encode(
            texts,
            provider=self.embedding_manager.default_provider,
            normalize=True
        )
            
        return embeddings
    
    def generate_sparse_embeddings(
        self,
        texts: List[str]
    ) -> Tuple[sp.csr_matrix, Dict[str, int], np.ndarray]:
        """Generate sparse embeddings using TF-IDF"""
        
        # Fit or transform based on whether vocabulary exists
        if not hasattr(self.tfidf_vectorizer, 'vocabulary_'):
            sparse_matrix = self.tfidf_vectorizer.fit_transform(texts)
        else:
            sparse_matrix = self.tfidf_vectorizer.transform(texts)
            
        vocabulary = self.tfidf_vectorizer.vocabulary_
        idf_weights = self.tfidf_vectorizer.idf_
        
        return sparse_matrix, vocabulary, idf_weights
    
    def generate_colbert_embeddings(
        self,
        texts: List[str],
        max_length: int = 512
    ) -> Dict[str, Any]:
        """Generate ColBERT-style token embeddings"""
        
        if not self.settings.enable_colbert:
            return None
            
        all_token_embeddings = []
        doc_boundaries = [0]
        
        with torch.no_grad() if TRANSFORMERS_AVAILABLE else contextlib.nullcontext():
            for text in texts:
                # Tokenize
                inputs = self.colbert_tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding=True
                )
                
                # Get token embeddings
                outputs = self.colbert_model(**inputs)
                token_embeddings = outputs.last_hidden_state.squeeze(0)
                
                # Normalize
                token_embeddings = torch.nn.functional.normalize(
                    token_embeddings, p=2, dim=-1
                )
                
                all_token_embeddings.append(token_embeddings.cpu().numpy())
                doc_boundaries.append(
                    doc_boundaries[-1] + token_embeddings.shape[0]
                )
        
        return {
            "token_embeddings": np.vstack(all_token_embeddings) if all_token_embeddings else np.array([]),
            "doc_boundaries": doc_boundaries,
            "num_docs": len(texts)
        }
    
    def add_documents_to_index(
        self,
        corpus_id: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Add documents to hybrid index"""
        
        if corpus_id not in self.indices:
            raise ValueError(f"Corpus {corpus_id} not found")
            
        indices = self.indices[corpus_id]
        texts = [doc["text"] for doc in documents]
        doc_ids = [doc["id"] for doc in documents]
        
        results = {
            "dense_indexed": 0,
            "sparse_indexed": 0,
            "colbert_indexed": 0
        }
        
        # Dense indexing
        if indices["dense_index"] is not None:
            dense_embeddings = self.generate_dense_embeddings(texts)
            
            if hasattr(indices["dense_index"], "train"):
                # Train IVF index if needed
                indices["dense_index"].train(dense_embeddings)
                
            indices["dense_index"].add(dense_embeddings)
            results["dense_indexed"] = len(dense_embeddings)
            
        # Sparse indexing
        if indices["sparse_index"] is not None:
            sparse_matrix, vocab, idf = self.generate_sparse_embeddings(texts)
            
            if indices["sparse_index"]["vectors"] is None:
                indices["sparse_index"]["vectors"] = sparse_matrix
            else:
                indices["sparse_index"]["vectors"] = sp.vstack([
                    indices["sparse_index"]["vectors"],
                    sparse_matrix
                ])
                
            indices["sparse_index"]["vocabulary"] = vocab
            indices["sparse_index"]["idf_weights"] = idf
            results["sparse_indexed"] = sparse_matrix.shape[0]
            
        # ColBERT indexing
        if self.settings.enable_colbert and indices["colbert_index"]:
            colbert_data = self.generate_colbert_embeddings(texts)
            
            if colbert_data and colbert_data["num_docs"] > 0:
                indices["colbert_index"]["token_embeddings"].append(
                    colbert_data["token_embeddings"]
                )
                indices["colbert_index"]["doc_boundaries"].extend(
                    colbert_data["doc_boundaries"][1:]  # Skip first 0
                )
                results["colbert_indexed"] = colbert_data["num_docs"]
        
        # Update metadata
        indices["metadata"]["num_documents"] += len(documents)
            
        return results


class CorpusManager:
    """Manages corpus creation and configuration"""
    
    def __init__(self, settings_obj: Optional[Any] = None):
        self.settings = settings_obj or settings
        self.indexer = HybridIndexer(settings_obj)
        
    def create_corpus(
        self,
        name: str,
        description: str,
        indexing_strategy: str = "hybrid",
        chunk_size: int = 512,
        enable_colbert: bool = True,
        embedding_provider: str = "sentence_transformers",
        embedding_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new corpus with specified configuration"""
        
        # Configure embedding provider
        if embedding_config:
            self._configure_embedding_provider(embedding_provider, embedding_config)
        
        # Create index
        corpus_id = f"corpus_{name}_{datetime.utcnow().timestamp()}"
        index = self.indexer.create_hybrid_index(corpus_id)
        
        corpus = {
            "id": corpus_id,
            "name": name,
            "description": description,
            "indexing_strategy": indexing_strategy,
            "chunk_size": chunk_size,
            "enable_colbert": enable_colbert,
            "embedding_provider": embedding_provider,
            "embedding_config": embedding_config or {},
            "created_at": datetime.utcnow().isoformat(),
            "index": index
        }
        
        return corpus
    
    def _configure_embedding_provider(self, provider: str, config: Dict[str, Any]):
        """Configure the embedding provider for the indexer"""
        from .embeddings import EmbeddingProvider, GeminiEmbedding, VertexAIEmbedding, SentenceTransformerEmbedding, TaskType
        
        if provider == "gemini":
            from .rate_limiter import RateLimitTier
            # Get rate limit tier from settings
            tier_map = {
                "free": RateLimitTier.FREE,
                "tier_1": RateLimitTier.TIER_1,
                "tier_2": RateLimitTier.TIER_2,
                "tier_3": RateLimitTier.TIER_3
            }
            rate_tier = tier_map.get(self.settings.rate_limit_tier, RateLimitTier.FREE)
            
            model = GeminiEmbedding(
                model_name=config.get("gemini_model", "gemini-embedding-001"),
                task_type=TaskType(config.get("gemini_task_type", "RETRIEVAL_DOCUMENT")),
                output_dimensionality=config.get("gemini_dimensionality", 768),
                rate_limit_tier=rate_tier,
                batch_size=self.settings.embedding_batch_size
            )
            self.indexer.embedding_manager.add_model(EmbeddingProvider.GEMINI, model)
            self.indexer.embedding_manager.default_provider = EmbeddingProvider.GEMINI
            
        elif provider == "vertex_ai":
            model = VertexAIEmbedding(
                model_name=config.get("vertex_model", "text-embedding-005")
            )
            self.indexer.embedding_manager.add_model(EmbeddingProvider.VERTEX_AI, model)
            self.indexer.embedding_manager.default_provider = EmbeddingProvider.VERTEX_AI
            
        elif provider == "sentence_transformers":
            model = SentenceTransformerEmbedding(
                model_name=f"sentence-transformers/{config.get('st_model', 'all-MiniLM-L6-v2')}"
            )
            self.indexer.embedding_manager.add_model(EmbeddingProvider.SENTENCE_TRANSFORMERS, model)
            self.indexer.embedding_manager.default_provider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    
    def get_config(self, corpus_id: str) -> Dict[str, Any]:
        """Get corpus configuration"""
        if corpus_id not in self.indexer.indices:
            raise ValueError(f"Corpus {corpus_id} not found")
            
        index = self.indexer.indices[corpus_id]
        return {
            "corpus_id": corpus_id,
            "metadata": index["metadata"],
            "has_dense_index": index["dense_index"] is not None,
            "has_sparse_index": index["sparse_index"] is not None,
            "has_colbert_index": index["colbert_index"] is not None
        }