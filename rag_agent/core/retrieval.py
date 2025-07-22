"""
Advanced retrieval implementation with ColBERT and hybrid search.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import logging
import scipy.sparse as sp

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    import faiss
except ImportError:
    from .mock_components import MockFAISS as faiss
    
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    from .mock_components import MockSentenceTransformer as SentenceTransformer

from ..config import settings
from ..optimization.caching import RetrievalCache

logger = logging.getLogger(__name__)


class ColBERTRetriever:
    """Implements ColBERT-style late interaction retrieval"""
    
    def __init__(self, settings_obj: Optional[Any] = None):
        self.settings = settings_obj or settings
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None
        
    def colbert_search(
        self,
        query_embeddings: np.ndarray,
        doc_token_embeddings: List[np.ndarray],
        doc_boundaries: List[int],
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Perform ColBERT late interaction search"""
        
        if not TORCH_AVAILABLE:
            # Return mock results if torch not available
            return [(i, float(np.random.rand())) for i in range(min(top_k, 5))]
            
        # Convert to tensors
        query_tensor = torch.tensor(query_embeddings).to(self.device)
        
        doc_scores = []
        
        for doc_idx in range(len(doc_boundaries) - 1):
            start_idx = doc_boundaries[doc_idx]
            end_idx = doc_boundaries[doc_idx + 1]
            
            # Get document token embeddings
            doc_tokens = torch.tensor(
                doc_token_embeddings[start_idx:end_idx]
            ).to(self.device)
            
            # Compute late interaction score
            score = self._compute_late_interaction(
                query_tensor,
                doc_tokens
            )
            
            doc_scores.append((doc_idx, score))
            
        # Sort by score
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_scores[:top_k]
    
    def _compute_late_interaction(
        self,
        query_embeddings: Any,  # torch.Tensor when available
        doc_embeddings: Any  # torch.Tensor when available
    ) -> float:
        """Compute ColBERT late interaction score"""
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(
            query_embeddings,
            doc_embeddings.T
        )
        
        # Max-pooling over document dimension
        max_similarities = torch.max(similarity_matrix, dim=1)[0]
        
        # Sum over query dimension
        score = torch.sum(max_similarities).item()
        
        return score


class DenseRetriever:
    """Dense retrieval using FAISS"""
    
    def __init__(self, settings_obj: Optional[Any] = None):
        self.settings = settings_obj or settings
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        
    def search(
        self,
        query: str,
        index: Any,  # faiss.Index when available
        top_k: int = 10,
        doc_metadata: Optional[Dict[int, Dict]] = None
    ) -> List[Tuple[int, float, Dict]]:
        """Perform dense retrieval"""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        )[0]
        
        # Search in FAISS index
        scores, indices = index.search(
            query_embedding.reshape(1, -1),
            top_k
        )
        
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx >= 0:  # Valid result
                metadata = doc_metadata.get(idx, {}) if doc_metadata else {}
                results.append((idx, float(score), metadata))
                
        return results


class SparseRetriever:
    """Sparse retrieval using TF-IDF"""
    
    def __init__(self, settings_obj: Optional[Any] = None):
        self.settings = settings_obj or settings
        
    def search(
        self,
        query: str,
        sparse_index: Dict[str, Any],
        top_k: int = 10,
        doc_metadata: Optional[Dict[int, Dict]] = None
    ) -> List[Tuple[int, float, Dict]]:
        """Perform sparse retrieval"""
        
        if sparse_index["vectors"] is None:
            return []
            
        # Transform query using the same vectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create vectorizer with the same vocabulary
        vectorizer = TfidfVectorizer(vocabulary=sparse_index["vocabulary"])
        query_vector = vectorizer.transform([query])
        
        # Calculate similarities
        similarities = sparse_index["vectors"].dot(query_vector.T).toarray().flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                metadata = doc_metadata.get(idx, {}) if doc_metadata else {}
                results.append((idx, float(similarities[idx]), metadata))
                
        return results


class HybridRetriever:
    """Implements hybrid retrieval combining multiple strategies"""
    
    def __init__(self, settings_obj: Optional[Any] = None):
        self.settings = settings_obj or settings
        self.dense_retriever = DenseRetriever(settings_obj)
        self.sparse_retriever = SparseRetriever(settings_obj)
        self.colbert_retriever = ColBERTRetriever(settings_obj)
        
        # Initialize cache if enabled
        if self.settings.enable_caching:
            self.cache = RetrievalCache(settings_obj)
        else:
            self.cache = None
            
        # Store indices
        self.indices = {}
        
    def set_index(self, corpus_id: str, index: Dict[str, Any]):
        """Set index for a corpus"""
        self.indices[corpus_id] = index
        
    def get_index(self, corpus_id: str) -> Dict[str, Any]:
        """Get index for a corpus"""
        return self.indices.get(corpus_id)
        
    def search(
        self,
        query: str,
        corpus_id: str,
        top_k: int = 10,
        alpha: float = 0.5,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search"""
        
        # Check cache
        if use_cache and self.cache:
            cached_results = self.cache.get(query, corpus_id)
            if cached_results:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_results
                
        # Get index
        index = self.get_index(corpus_id)
        if not index:
            logger.error(f"No index found for corpus: {corpus_id}")
            return []
        
        results = {
            "dense": [],
            "sparse": [],
            "colbert": []
        }
        
        # Dense retrieval
        if index.get("dense_index"):
            results["dense"] = self.dense_retriever.search(
                query, index["dense_index"], top_k * 2
            )
            
        # Sparse retrieval
        if index.get("sparse_index"):
            results["sparse"] = self.sparse_retriever.search(
                query, index["sparse_index"], top_k * 2
            )
            
        # ColBERT retrieval
        if self.settings.enable_colbert and index.get("colbert_index"):
            # Generate query embeddings for ColBERT
            query_embeddings = self._generate_query_embeddings(query)
            if query_embeddings is not None and index["colbert_index"]["token_embeddings"]:
                colbert_results = self.colbert_retriever.colbert_search(
                    query_embeddings,
                    index["colbert_index"]["token_embeddings"],
                    index["colbert_index"]["doc_boundaries"],
                    top_k * 2
                )
                # Convert to standard format
                results["colbert"] = [
                    (idx, score, {}) for idx, score in colbert_results
                ]
            
        # Combine results
        combined_results = self._combine_results(
            results,
            alpha,
            top_k
        )
        
        # Rerank if enabled
        if self.settings.enable_reranking:
            combined_results = self._rerank_results(
                query,
                combined_results
            )
            
        # Cache results
        if use_cache and self.cache:
            self.cache.set(query, corpus_id, combined_results)
            
        return combined_results
    
    def _generate_query_embeddings(self, query: str) -> Optional[np.ndarray]:
        """Generate embeddings for ColBERT query"""
        try:
            # Use the same embedding model as in indexing
            embeddings = self.dense_retriever.embedding_model.encode(
                [query],
                normalize_embeddings=True
            )
            return embeddings[0]
        except Exception as e:
            logger.error(f"Failed to generate query embeddings: {e}")
            return None
    
    def _combine_results(
        self,
        results: Dict[str, List],
        alpha: float,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Combine results from different retrievers"""
        
        # Normalize scores
        normalized_results = self._normalize_scores(results)
        
        # Aggregate scores
        doc_scores = defaultdict(float)
        doc_metadata = {}
        
        # Weight dense results
        for doc_id, score, metadata in normalized_results["dense"]:
            doc_scores[doc_id] += alpha * score
            doc_metadata[doc_id] = metadata
            
        # Weight sparse results
        for doc_id, score, metadata in normalized_results["sparse"]:
            doc_scores[doc_id] += (1 - alpha) * score
            if doc_id not in doc_metadata:
                doc_metadata[doc_id] = metadata
                
        # Add ColBERT scores if available
        if normalized_results.get("colbert"):
            colbert_weight = 0.3  # Additional weight for ColBERT
            for doc_id, score, metadata in normalized_results["colbert"]:
                doc_scores[doc_id] += colbert_weight * score
                if doc_id not in doc_metadata:
                    doc_metadata[doc_id] = metadata
                    
        # Sort by combined score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Format results
        combined_results = []
        for doc_id, score in sorted_docs:
            combined_results.append({
                "doc_id": doc_id,
                "score": score,
                "metadata": doc_metadata[doc_id]
            })
            
        return combined_results
    
    def _normalize_scores(self, results: Dict[str, List]) -> Dict[str, List]:
        """Normalize scores across different retrievers"""
        
        normalized = {}
        
        for retriever_type, retriever_results in results.items():
            if not retriever_results:
                normalized[retriever_type] = []
                continue
                
            # Extract scores
            scores = [score for _, score, _ in retriever_results]
            
            if scores:
                min_score = min(scores)
                max_score = max(scores)
                
                if max_score > min_score:
                    # Min-max normalization
                    normalized_results = []
                    for doc_id, score, metadata in retriever_results:
                        norm_score = (score - min_score) / (max_score - min_score)
                        normalized_results.append((doc_id, norm_score, metadata))
                    normalized[retriever_type] = normalized_results
                else:
                    # All scores are the same
                    normalized[retriever_type] = [
                        (doc_id, 1.0, metadata) 
                        for doc_id, score, metadata in retriever_results
                    ]
            else:
                normalized[retriever_type] = []
                
        return normalized
    
    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank results using cross-encoder"""
        
        # For now, return results as-is
        # In production, implement actual reranking model
        logger.info("Reranking not implemented yet, returning original order")
        return results