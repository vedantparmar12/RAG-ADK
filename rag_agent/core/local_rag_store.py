"""
Local RAG store implementation with Ollama integration for Windows.
Supports both local Ollama models and Vertex AI RAG.
"""

import os
import json
import hashlib
import pickle
import shutil
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging
import numpy as np
from datetime import datetime
import warnings
import asyncio
import httpx
import requests
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Local embeddings will not be available.")

try:
    from .qdrant_client import QdrantRAGClient, get_qdrant_client
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("Qdrant client not available. Vector store functionality will be limited.")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("langchain-text-splitters not installed. Text splitting will use basic implementation.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. Will use simple numpy search.")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank-bm25 not installed. BM25 search will not be available.")

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("Google Generative AI not installed.")

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False
    logger.warning("PyArrow not installed. Parquet persistence will not be available.")

from ..config import settings


class RAGMode(Enum):
    """RAG system modes"""
    LOCAL = "local"
    VERTEX_AI = "vertex_ai"
    HYBRID = "hybrid"

@dataclass
class OllamaModel:
    """Ollama model configuration"""
    name: str
    tag: str = "latest"
    context_length: int = 4096
    embedding_dim: Optional[int] = None
    
    @property
    def full_name(self) -> str:
        return f"{self.name}:{self.tag}"

@dataclass
class LocalRAGConfig:
    """Configuration for Local RAG system"""
    storage_path: Path = field(default_factory=lambda: Path("./local_rag_store"))
    ollama_host: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"  # Default Ollama embedding model
    generation_model: str = "llama3.2"  # Default Ollama generation model
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    use_bm25: bool = True
    use_reranker: bool = False
    batch_size: int = 32
    max_workers: int = 4
    use_parquet: bool = True  # Enable Parquet persistence
    
    # Performance optimizations
    use_caching: bool = True
    use_redis: bool = False
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 3600  # 1 hour
    embedding_cache_ttl: int = 86400  # 24 hours
    
    # Advanced retrieval
    use_async_search: bool = True
    use_cross_encoder: bool = True
    use_compression: bool = True
    use_query_enhancement: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    compression_threshold: float = 0.3
    max_sentences_per_chunk: int = 3
    
    # Hybrid search weights
    vector_weight: float = 0.5
    fts_weight: float = 0.3
    bm25_weight: float = 0.2

@dataclass
class LocalDocument:
    """Represents a document in the local store"""
    id: str
    source_uri: str
    display_name: str
    content: str
    chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str

@dataclass
class LocalCorpus:
    """Represents a corpus in the local store"""
    name: str
    display_name: str
    description: str
    created_at: str
    updated_at: str
    document_count: int
    metadata: Dict[str, Any]

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host.rstrip('/')
        self.api_url = f"{self.host}/api"
        
    def list_models(self) -> List[Dict[str, Any]]:
        """List available Ollama models"""
        try:
            response = requests.get(f"{self.api_url}/tags")
            response.raise_for_status()
            return response.json().get('models', [])
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry"""
        try:
            response = requests.post(
                f"{self.api_url}/pull",
                json={"name": model_name},
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'status' in data:
                        logger.info(f"Pulling {model_name}: {data['status']}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def generate_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Generate embedding using Ollama model"""
        try:
            response = requests.post(
                f"{self.api_url}/embeddings",
                json={"model": model, "prompt": text}
            )
            response.raise_for_status()
            return response.json().get('embedding')
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def generate_completion(self, prompt: str, model: str, **kwargs) -> Optional[str]:
        """Generate text completion using Ollama model"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                **kwargs
            }
            
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload
            )
            response.raise_for_status()
            return response.json().get('response')
        except Exception as e:
            logger.error(f"Failed to generate completion: {e}")
            return None
    
    async def generate_completion_async(self, prompt: str, model: str, **kwargs) -> Optional[str]:
        """Async version of generate_completion"""
        async with httpx.AsyncClient() as client:
            try:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    **kwargs
                }
                
                response = await client.post(
                    f"{self.api_url}/generate",
                    json=payload,
                    timeout=60.0
                )
                response.raise_for_status()
                return response.json().get('response')
            except Exception as e:
                logger.error(f"Failed to generate async completion: {e}")
                return None


class LocalRAGStore:
    """Local RAG storage system with Ollama integration and advanced search capabilities"""
    
    def __init__(self, config: Optional[LocalRAGConfig] = None, storage_path: str = "./rag_storage"):
        self.config = config or LocalRAGConfig()
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize paths
        self.corpora_path = self.storage_path / "corpora"
        self.corpora_path.mkdir(exist_ok=True)
        
        self.index_path = self.storage_path / "indices"
        self.index_path.mkdir(exist_ok=True)
        
        self.cache_path = self.storage_path / "cache"
        self.cache_path.mkdir(exist_ok=True)
        
        self.lancedb_path = self.storage_path / "lancedb"
        self.lancedb_path.mkdir(exist_ok=True)
        
        # Initialize Ollama client
        self.ollama_client = OllamaClient(self.config.ollama_host)
        
        # Initialize embedding model
        self.embedding_model = None
        self.use_ollama_embeddings = self._check_ollama_embedding_model()
        
        if not self.use_ollama_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Initialized local embedding model: all-MiniLM-L6-v2")
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
        
        # Initialize vector/FTS stores
        self.qdrant = get_qdrant_client() if QDRANT_AVAILABLE else None
        self.fts = None
        try:
            from .fts_client import MeiliClient
            self.fts = MeiliClient(url=os.getenv("MEILI_URL", "http://localhost:7700"))
        except Exception:
            logger.warning("FTS client not available; FTS disabled.")
        
        # Initialize caching
        self.cache_manager = None
        if self.config.use_caching:
            try:
                from .cache_manager import create_cache_manager
                self.cache_manager = create_cache_manager(
                    use_redis=self.config.use_redis,
                    redis_url=self.config.redis_url
                )
                logger.info("Initialized cache manager")
            except Exception as e:
                logger.warning(f"Failed to initialize cache: {e}")
        
        # Initialize retrieval optimizer
        self.retrieval_optimizer = None
        if self.config.use_cross_encoder or self.config.use_compression:
            try:
                from .reranking import create_retrieval_optimizer
                self.retrieval_optimizer = create_retrieval_optimizer(
                    enable_reranking=self.config.use_cross_encoder,
                    enable_compression=self.config.use_compression,
                    enable_query_enhancement=self.config.use_query_enhancement
                )
                logger.info("Initialized retrieval optimizer")
            except Exception as e:
                logger.warning(f"Failed to initialize retrieval optimizer: {e}")
        
        # Initialize BM25 index
        self.bm25_index = {}
        
        # Initialize text splitter
        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        else:
            self.text_splitter = None
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Metadata storage
        self.metadata_file = self.storage_path / "metadata.json"
        self.metadata = self._load_metadata()
        
        # Initialize Gemini if available
        self.genai_model = None
        if GENAI_AVAILABLE and settings.google_api_key:
            try:
                genai.configure(api_key=settings.google_api_key)
                self.genai_model = genai.GenerativeModel(settings.generation_model)
                logger.info("Initialized Google Generative AI model")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
    
    def _check_ollama_embedding_model(self) -> bool:
        """Check if Ollama embedding model is available"""
        try:
            models = self.ollama_client.list_models()
            for model in models:
                if self.config.embedding_model in model.get('name', ''):
                    logger.info(f"Found Ollama embedding model: {self.config.embedding_model}")
                    return True
            
            # Try to pull the model
            logger.info(f"Pulling Ollama embedding model: {self.config.embedding_model}")
            if self.ollama_client.pull_model(self.config.embedding_model):
                return True
        except Exception as e:
            logger.error(f"Failed to check/pull Ollama embedding model: {e}")
        
        return False
    
    def init_vector_store(self):
        """Deprecated: LanceDB init (replaced by Qdrant + Meili)."""
        pass
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        else:
            return {
                "corpora": {},
                "created_at": datetime.now().isoformat(),
                "version": "2.0.0"
            }
    
    def _save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)
    
    def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text using Ollama or local model with caching"""
        # Check cache first
        if self.cache_manager:
            cached_embedding = self.cache_manager.get_embedding(text, self.config.embedding_model)
            if cached_embedding is not None:
                return cached_embedding
        
        try:
            embedding = None
            if self.use_ollama_embeddings:
                # Use Ollama for embeddings
                embedding_list = self.ollama_client.generate_embedding(
                    text, 
                    self.config.embedding_model
                )
                if embedding_list:
                    embedding = np.array(embedding_list)
            elif self.embedding_model:
                # Use sentence-transformers
                embedding = self.embedding_model.encode(text)
                embedding = np.array(embedding)
            
            # Cache the embedding
            if embedding is not None and self.cache_manager:
                self.cache_manager.set_embedding(
                    text, 
                    self.config.embedding_model, 
                    embedding, 
                    self.config.embedding_cache_ttl
                )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
        
        return None
    
    def _generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts in batch"""
        embeddings = []
        
        if self.use_ollama_embeddings:
            # Process in batches for Ollama
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                
                # Use thread pool for parallel processing
                with ThreadPoolExecutor(max_workers=min(4, len(batch))) as executor:
                    futures = [
                        executor.submit(self._generate_embedding, text) 
                        for text in batch
                    ]
                    
                    for future in as_completed(futures):
                        emb = future.result()
                        if emb is not None:
                            embeddings.append(emb)
        elif self.embedding_model:
            # Batch encode with sentence-transformers
            try:
                batch_embeddings = self.embedding_model.encode(texts, batch_size=self.config.batch_size)
                embeddings = [np.array(emb) for emb in batch_embeddings]
            except Exception as e:
                logger.error(f"Failed to generate batch embeddings: {e}")
        
        return embeddings
    
    def _split_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata"""
        chunks = []
        
        if self.text_splitter:
            text_chunks = self.text_splitter.split_text(text)
        else:
            # Basic splitting implementation
            text_chunks = []
            for i in range(0, len(text), self.config.chunk_size - self.config.chunk_overlap):
                chunk = text[i:i + self.config.chunk_size]
                if chunk:
                    text_chunks.append(chunk)
        
        # Add metadata to each chunk
        for i, chunk_text in enumerate(text_chunks):
            chunk_data = {
                'text': chunk_text,
                'chunk_index': i,
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat()
            }
            chunks.append(chunk_data)
        
        return chunks
    
    def _get_corpus_path(self, corpus_name: str) -> Path:
        """Get the storage path for a corpus"""
        # Sanitize corpus name for filesystem
        safe_name = corpus_name.replace("/", "_").replace("\\", "_")
        return self.corpora_path / safe_name
    
    def _get_corpus_info_path(self, corpus_name: str) -> Path:
        """Get the path for corpus metadata"""
        return self._get_corpus_path(corpus_name) / "corpus_info.json"
    
    def _get_documents_path(self, corpus_name: str) -> Path:
        """Get the path for documents storage"""
        return self._get_corpus_path(corpus_name) / "documents"
    
    def _get_index_path(self, corpus_name: str) -> Path:
        """Get the path for index storage"""
        return self._get_corpus_path(corpus_name) / "index"
    
    def create_corpus(self, corpus_name: str, display_name: str = None, description: str = None) -> LocalCorpus:
        """Create a new corpus and backends (Qdrant collection + FTS index)."""
        corpus_id = hashlib.md5(corpus_name.encode()).hexdigest()[:12]
        corpus_path = self._get_corpus_path(corpus_name)
        
        if corpus_path.exists():
            # Load existing corpus
            return self.get_corpus(corpus_name)
        
        # Create corpus structure
        corpus_path.mkdir(exist_ok=True)
        self._get_documents_path(corpus_name).mkdir(exist_ok=True)
        self._get_index_path(corpus_name).mkdir(exist_ok=True)
        
        # Create corpus metadata
        corpus = LocalCorpus(
            name=corpus_name,
            display_name=display_name or corpus_name,
            description=description or "",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            document_count=0,
            metadata={'corpus_id': corpus_id, 'embedding_model': self.config.embedding_model}
        )
        
        # Save corpus info
        with open(self._get_corpus_info_path(corpus_name), 'w') as f:
            json.dump(asdict(corpus), f, indent=2)
        
        # Update global metadata
        self.metadata["corpora"][corpus_name] = asdict(corpus)
        self._save_metadata()
        
        # Create Qdrant collection
        if self.qdrant:
            try:
                # Default to 384; will still accept any consistent vector size
                self.qdrant.create_corpus_collection(corpus_name=corpus_name, vector_dim=384, distance="Cosine")
                logger.info(f"Created Qdrant collection for corpus {corpus_name}")
            except Exception as e:
                logger.error(f"Failed to create Qdrant collection: {e}")
        
        # Create FTS index
        if self.fts:
            try:
                self.fts.ensure_index(index=f"corpus_{corpus_id}")
            except Exception as e:
                logger.warning(f"Failed to init FTS index: {e}")
        
        # Initialize BM25 index for this corpus
        self.bm25_index[corpus_name] = {'documents': [], 'index': None}
        
        return corpus
    
    def get_corpus(self, corpus_name: str) -> Optional[LocalCorpus]:
        """Get corpus information"""
        info_path = self._get_corpus_info_path(corpus_name)
        
        if not info_path.exists():
            return None
        
        with open(info_path, 'r') as f:
            data = json.load(f)
        
        return LocalCorpus(**data)
    
    def list_corpora(self) -> List[LocalCorpus]:
        """List all available corpora"""
        corpora = []
        
        # First check metadata
        for corpus_name, corpus_data in self.metadata.get("corpora", {}).items():
            try:
                corpus = LocalCorpus(**corpus_data)
                corpora.append(corpus)
            except:
                pass
        
        # Also check directory structure
        for corpus_dir in self.corpora_path.iterdir():
            if corpus_dir.is_dir():
                corpus = self.get_corpus(corpus_dir.name)
                if corpus and corpus.name not in [c.name for c in corpora]:
                    corpora.append(corpus)
        
        return corpora
    
    def delete_corpus(self, corpus_name: str) -> bool:
        """Delete a corpus and all its data"""
        corpus_path = self._get_corpus_path(corpus_name)
        
        if not corpus_path.exists() and corpus_name not in self.metadata.get("corpora", {}):
            return False
        
        # Get corpus ID
        corpus_data = self.metadata.get("corpora", {}).get(corpus_name, {})
        corpus_id = corpus_data.get('metadata', {}).get('corpus_id')
        
        # Delete corpus directory
        if corpus_path.exists():
            shutil.rmtree(corpus_path)
        
        # Delete LanceDB table
        if self.db and LANCEDB_AVAILABLE and corpus_id:
            try:
                table_name = f"corpus_{corpus_id}"
                self.db.drop_table(table_name)
            except Exception as e:
                logger.warning(f"Failed to drop LanceDB table: {e}")
        
        # Remove from BM25 index
        if corpus_name in self.bm25_index:
            del self.bm25_index[corpus_name]
        
        # Update metadata
        if corpus_name in self.metadata.get("corpora", {}):
            del self.metadata["corpora"][corpus_name]
            self._save_metadata()
        
        logger.info(f"Deleted corpus {corpus_name}")
        return True
    
    def add_document(
        self, 
        corpus_name: str, 
        source_uri: str, 
        content: str = None,
        display_name: str = None,
        metadata: Dict[str, Any] = None,
        file_path: str = None
    ) -> LocalDocument:
        """Add a document to the corpus with enhanced processing"""
        corpus = self.get_corpus(corpus_name)
        if not corpus:
            raise ValueError(f"Corpus '{corpus_name}' does not exist")
        
        # Extract content from file if provided
        if file_path and not content:
            content = self._extract_content_from_file(file_path)
            if not content:
                raise ValueError(f"Failed to extract content from {file_path}")
        elif not content:
            raise ValueError("Either content or file_path must be provided")
        
        # Generate document ID
        doc_id = hashlib.md5(f"{source_uri}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        # Split text into chunks with metadata
        chunks = self._split_text(content, metadata)
        
        # Generate embeddings for all chunks in batch
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self._generate_embeddings_batch(chunk_texts)
        
        # Create document object
        document = LocalDocument(
            id=doc_id,
            source_uri=source_uri,
            display_name=display_name or os.path.basename(source_uri),
            content=content[:1000] + "..." if len(content) > 1000 else content,
            chunks=[{
                "id": f"{doc_id}_chunk_{i}",
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {}),
                "chunk_index": i
            } for i, chunk in enumerate(chunks)],
            metadata=metadata or {},
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        # Save document
        doc_path = self._get_documents_path(corpus_name) / f"{doc_id}.json"
        with open(doc_path, 'w') as f:
            json.dump(asdict(document), f, indent=2)
        
        # Get corpus ID
        corpus_id = corpus.metadata.get('corpus_id')
        
        # Upsert to Qdrant
        if self.qdrant and embeddings and corpus_id:
            try:
                docs_to_upsert = []
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    docs_to_upsert.append({
                        "id": f"{doc_id}_{i}",
                        "text": chunk['text'],
                        "vector": embedding.tolist(),
                        "document_id": doc_id,
                        "chunk_index": i,
                        "metadata": {**(chunk.get('metadata', {})), **(metadata or {})},
                        "timestamp": chunk['timestamp']
                    })
                self.qdrant.upsert_documents(corpus_name=corpus_name, documents=docs_to_upsert)
                logger.info(f"Upserted {len(docs_to_upsert)} chunks to Qdrant")
            except Exception as e:
                logger.error(f"Failed to upsert to Qdrant: {e}")
        
        # Upsert to FTS
        if self.fts and corpus_id:
            try:
                index = f"corpus_{corpus_id}"
                docs = []
                for i, chunk in enumerate(chunks):
                    docs.append({
                        "id": f"{doc_id}_{i}",
                        "text": chunk['text'],
                        "document_id": doc_id,
                        "chunk_index": i,
                        "metadata": chunk.get('metadata', {})
                    })
                self.fts.upsert_documents(index=index, docs=docs)
            except Exception as e:
                logger.warning(f"FTS upsert failed: {e}")
        
        # Save to Parquet if enabled
        if self.config.use_parquet and ARROW_AVAILABLE:
            self._save_chunks_to_parquet(corpus_name, doc_id, chunks, embeddings)
        
        # Update BM25 index
        if self.config.use_bm25 and BM25_AVAILABLE:
            self._update_bm25_index(corpus_name, chunk_texts, doc_id)
        
        # Update corpus metadata
        corpus.document_count += 1
        corpus.updated_at = datetime.now().isoformat()
        corpus_data = asdict(corpus)
        with open(self._get_corpus_info_path(corpus_name), 'w') as f:
            json.dump(corpus_data, f, indent=2)
        
        self.metadata["corpora"][corpus_name] = corpus_data
        self._save_metadata()
        
        return document
    
    def _extract_content_from_file(self, file_path: str) -> Optional[str]:
        """Extract text content from various file types"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        try:
            # Handle different file types
            if file_path.suffix.lower() == '.pdf':
                return self._extract_pdf_content(file_path)
            elif file_path.suffix.lower() in ['.txt', '.md']:
                return file_path.read_text(encoding='utf-8')
            elif file_path.suffix.lower() in ['.doc', '.docx']:
                return self._extract_docx_content(file_path)
            else:
                # Try to read as text
                return file_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to extract content from {file_path}: {e}")
            return None
    
    def _extract_pdf_content(self, file_path: Path) -> Optional[str]:
        """Extract text from PDF file"""
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            logger.error("PyPDF2 not installed. Cannot extract PDF content.")
        except Exception as e:
            logger.error(f"Failed to extract PDF content: {e}")
        return None
    
    def _extract_docx_content(self, file_path: Path) -> Optional[str]:
        """Extract text from DOCX file"""
        try:
            import docx2txt
            return docx2txt.process(str(file_path))
        except ImportError:
            logger.error("docx2txt not installed. Cannot extract DOCX content.")
        except Exception as e:
            logger.error(f"Failed to extract DOCX content: {e}")
        return None
    
    def _update_bm25_index(self, corpus_name: str, texts: List[str], doc_id: str):
        """Update BM25 index for a corpus"""
        if not BM25_AVAILABLE:
            return
        
        try:
            # Tokenize texts
            tokenized_texts = [text.lower().split() for text in texts]
            
            # Add to corpus documents
            for i, (text, tokens) in enumerate(zip(texts, tokenized_texts)):
                self.bm25_index[corpus_name]['documents'].append({
                    'id': f"{doc_id}_{i}",
                    'text': text,
                    'tokens': tokens
                })
            
            # Rebuild BM25 index
            all_tokens = [doc['tokens'] for doc in self.bm25_index[corpus_name]['documents']]
            self.bm25_index[corpus_name]['index'] = BM25Okapi(all_tokens)
            
            logger.info(f"Updated BM25 index for corpus {corpus_name}")
        except Exception as e:
            logger.error(f"Failed to update BM25 index: {e}")
    
    def get_document(self, corpus_name: str, document_id: str) -> Optional[LocalDocument]:
        """Get a specific document"""
        doc_path = self._get_documents_path(corpus_name) / f"{document_id}.json"
        
        if not doc_path.exists():
            return None
        
        with open(doc_path, 'r') as f:
            data = json.load(f)
        
        return LocalDocument(**data)
    
    def list_documents(self, corpus_name: str) -> List[LocalDocument]:
        """List all documents in a corpus"""
        documents = []
        docs_path = self._get_documents_path(corpus_name)
        
        if not docs_path.exists():
            return documents
        
        for doc_file in docs_path.glob("*.json"):
            with open(doc_file, 'r') as f:
                data = json.load(f)
            documents.append(LocalDocument(**data))
        
        return documents
    
    def delete_document(self, corpus_name: str, document_id: str) -> bool:
        """Delete a document from the corpus"""
        doc_path = self._get_documents_path(corpus_name) / f"{document_id}.json"
        
        if not doc_path.exists():
            return False
        
        # Load document to get chunk IDs
        document = self.get_document(corpus_name, document_id)
        if document:
            # Remove from vector/FTS backends
            try:
                if self.qdrant:
                    self.qdrant.delete_document(corpus_name, document_id)
            except Exception as e:
                logger.warning(f"Qdrant delete failed: {e}")
            try:
                if self.fts:
                    corpus = self.get_corpus(corpus_name)
                    if corpus:
                        corpus_id = corpus.metadata.get('corpus_id')
                        if corpus_id:
                            self.fts.delete_documents_by_document_id(f"corpus_{corpus_id}", document_id)
            except Exception as e:
                logger.warning(f"FTS delete failed: {e}")
                
                # Delete document file
                doc_path.unlink()
                
                # Update corpus count
                corpus = self.get_corpus(corpus_name)
                if corpus:
                    corpus.document_count = max(0, corpus.document_count - 1)
                    corpus.updated_at = datetime.now().isoformat()
                    with open(self._get_corpus_info_path(corpus_name), 'w') as f:
                        json.dump(asdict(corpus), f, indent=2)
                
                return True
        
        return False
    
    def search(
        self,
        corpus_name: str,
        query: str,
        top_k: int = None,
        use_hybrid: bool = True,
        rerank: bool = None,
        where: str = None,
        metadata_filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Advanced search with hybrid retrieval and optional reranking"""
        corpus = self.get_corpus(corpus_name)
        if not corpus:
            raise ValueError(f"Corpus '{corpus_name}' does not exist")
        
        top_k = top_k or self.config.top_k
        rerank = rerank if rerank is not None else self.config.use_reranker
        
        # Get corpus ID
        corpus_id = corpus.metadata.get('corpus_id')
        if not corpus_id:
            logger.warning(f"Corpus {corpus_name} missing corpus_id")
            return []
        
        results = []
        
        # Build Qdrant filter from WHERE clause or metadata filters
        qdrant_filter = None
        if where:
            from .qdrant_filters import QdrantFilterBuilder
            qdrant_filter = QdrantFilterBuilder.parse_where_clause(where)
        elif metadata_filters:
            from .qdrant_filters import create_metadata_filter
            qdrant_filter = create_metadata_filter(metadata_filters)
        
        # Vector search
        vector_results = self._vector_search(corpus_name, query, top_k * 2 if use_hybrid else top_k, qdrant_filter)
        
        # FTS search
        fts_results = []
        if self.fts:
            fts_results = self._fts_search(corpus_id, query, top_k * 2)
        
        # BM25 search
        bm25_results = []
        if use_hybrid and self.config.use_bm25:
            bm25_results = self._bm25_search(corpus_name, query, top_k * 2)
        
        # Use async parallel search if enabled
        if self.config.use_async_search:
            return self._search_async(corpus_name, query, top_k, use_hybrid, rerank, qdrant_filter, corpus_id)
        
        # Combine results (vector + fts + bm25)
        candidates = vector_results
        if fts_results:
            candidates = self._hybrid_fusion(candidates, fts_results, max(top_k, 10))
        if bm25_results:
            candidates = self._hybrid_fusion(candidates, bm25_results, top_k)
        results = candidates[:top_k]
        
        # Apply retrieval optimizations
        if self.retrieval_optimizer:
            results = self.retrieval_optimizer.optimize_retrieval(
                query, results, top_k, self.ollama_client
            )
        elif rerank:
            results = self._rerank_results(query, results, top_k)
        
        return results
    
    def _vector_search(self, corpus_name: str, query: str, top_k: int, qdrant_filter=None) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        results = []
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        if self.qdrant and query_embedding is not None:
            try:
                search_results = self.qdrant.search_vectors(
                    corpus_name=corpus_name,
                    query_vector=query_embedding.tolist(),
                    top_k=top_k,
                    query_filter=qdrant_filter
                )
                for r in search_results:
                    results.append({
                        "id": r.get("id"),
                        "text": r.get("text"),
                        "score": r.get("score", 0.0),
                        "metadata": r.get("metadata", {}),
                        "document_id": r.get("document_id"),
                        "chunk_index": r.get("chunk_index"),
                    })
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
        
        return results
    
    def _fts_search(self, corpus_id: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        try:
            if not self.fts:
                return results
            index = f"corpus_{corpus_id}"
            hits = self.fts.search(index=index, query=query, limit=top_k)
            for h in hits:
                results.append(h)
        except Exception as e:
            logger.warning(f"FTS search failed: {e}")
        return results

    def _bm25_search(self, corpus_name: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search"""
        if not BM25_AVAILABLE or corpus_name not in self.bm25_index:
            return []
        
        results = []
        
        try:
            index_data = self.bm25_index[corpus_name]
            if not index_data['index']:
                return []
            
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            scores = index_data['index'].get_scores(query_tokens)
            
            # Get top-k results
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            for idx in top_indices:
                if scores[idx] > 0:
                    doc = index_data['documents'][idx]
                    results.append({
                        "id": doc['id'],
                        "text": doc['text'],
                        "score": float(scores[idx]),
                        "metadata": {},
                        "type": "bm25"
                    })
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
        
        return results
    
    def _hybrid_fusion(self, vector_results: List[Dict], bm25_results: List[Dict], 
                      top_k: int) -> List[Dict[str, Any]]:
        """Combine vector and BM25 results using reciprocal rank fusion"""
        # Create score maps
        vector_scores = {r['id']: i for i, r in enumerate(vector_results)}
        bm25_scores = {r['id']: i for i, r in enumerate(bm25_results)}
        
        # Calculate fusion scores
        fusion_scores = {}
        all_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        
        for doc_id in all_ids:
            vector_rank = vector_scores.get(doc_id, len(vector_results))
            bm25_rank = bm25_scores.get(doc_id, len(bm25_results))
            
            # Reciprocal rank fusion
            fusion_score = 1.0 / (vector_rank + 1) + 1.0 / (bm25_rank + 1)
            fusion_scores[doc_id] = fusion_score
        
        # Sort by fusion score
        sorted_ids = sorted(fusion_scores.keys(), key=lambda x: fusion_scores[x], reverse=True)
        
        # Build result list
        results = []
        id_to_result = {r['id']: r for r in vector_results + bm25_results}
        
        for doc_id in sorted_ids[:top_k]:
            result = id_to_result[doc_id].copy()
            result['fusion_score'] = fusion_scores[doc_id]
            results.append(result)
        
        return results
    
    def _rerank_results(self, query: str, results: List[Dict], top_k: int) -> List[Dict[str, Any]]:
        """Rerank results using Ollama model"""
        if not self.ollama_client:
            return results[:top_k]
        
        try:
            # Score each result
            for result in results:
                prompt = f"""Rate the relevance of this text to the query on a scale of 0-10.
                Query: {query}
                Text: {result['text'][:500]}
                Respond with just a number between 0 and 10."""
                
                response = self.ollama_client.generate_completion(
                    prompt, 
                    self.config.generation_model,
                    temperature=0.0
                )
                
                if response:
                    try:
                        score = float(response.strip())
                        result['rerank_score'] = score
                    except:
                        result['rerank_score'] = 0.0
                else:
                    result['rerank_score'] = 0.0
            
            # Sort by rerank score
            results.sort(key=lambda x: x.get('rerank_score', 0.0), reverse=True)
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
        
        return results[:top_k]
    
    def _search_async(self, corpus_name: str, query: str, top_k: int, use_hybrid: bool, rerank: bool, qdrant_filter, corpus_id: str) -> List[Dict[str, Any]]:
        """Perform async parallel search across vector, FTS, and BM25."""
        import asyncio
        
        async def async_search_wrapper():
            # Create async tasks for parallel execution
            tasks = []
            
            # Vector search task
            async def vector_task():
                return self._vector_search(corpus_name, query, top_k * 2 if use_hybrid else top_k, qdrant_filter)
            
            # FTS search task
            async def fts_task():
                if self.fts:
                    return self._fts_search(corpus_id, query, top_k * 2)
                return []
            
            # BM25 search task
            async def bm25_task():
                if use_hybrid and self.config.use_bm25:
                    return self._bm25_search(corpus_name, query, top_k * 2)
                return []
            
            # Execute all searches in parallel
            vector_results, fts_results, bm25_results = await asyncio.gather(
                vector_task(),
                fts_task(),
                bm25_task(),
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(vector_results, Exception):
                logger.error(f"Vector search failed: {vector_results}")
                vector_results = []
            if isinstance(fts_results, Exception):
                logger.error(f"FTS search failed: {fts_results}")
                fts_results = []
            if isinstance(bm25_results, Exception):
                logger.error(f"BM25 search failed: {bm25_results}")
                bm25_results = []
            
            # Combine results
            candidates = vector_results
            if fts_results:
                candidates = self._hybrid_fusion(candidates, fts_results, max(top_k, 10))
            if bm25_results:
                candidates = self._hybrid_fusion(candidates, bm25_results, top_k)
            
            results = candidates[:top_k]
            
            # Apply optimizations
            if self.retrieval_optimizer:
                results = self.retrieval_optimizer.optimize_retrieval(
                    query, results, top_k, self.ollama_client
                )
            elif rerank:
                results = self._rerank_results(query, results, top_k)
            
            return results
        
        # Run async search
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, async_search_wrapper())
                    return future.result()
            else:
                return asyncio.run(async_search_wrapper())
        except Exception as e:
            logger.error(f"Async search failed: {e}")
            # Fallback to sequential search
            vector_results = self._vector_search(corpus_name, query, top_k * 2 if use_hybrid else top_k, qdrant_filter)
            return vector_results[:top_k]
    
    def _save_chunks_to_parquet(self, corpus_name: str, doc_id: str, chunks: List[Dict], embeddings: List[np.ndarray]):
        """Save chunks to Parquet file for Arrow-friendly workflows."""
        if not ARROW_AVAILABLE:
            return
        
        try:
            parquet_path = self._get_corpus_path(corpus_name) / "chunks.parquet"
            
            # Prepare data for Parquet
            records = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                record = {
                    'id': f"{doc_id}_{i}",
                    'document_id': doc_id,
                    'chunk_index': i,
                    'text': chunk['text'],
                    'vector': embedding.tolist(),
                    'metadata': json.dumps(chunk.get('metadata', {})),
                    'timestamp': chunk['timestamp']
                }
                records.append(record)
            
            # Define Arrow schema
            schema = pa.schema([
                pa.field('id', pa.string()),
                pa.field('document_id', pa.string()),
                pa.field('chunk_index', pa.int32()),
                pa.field('text', pa.string()),
                pa.field('vector', pa.list_(pa.float32())),
                pa.field('metadata', pa.string()),
                pa.field('timestamp', pa.string())
            ])
            
            # Create table
            table = pa.Table.from_pylist(records, schema=schema)
            
            # Append to existing file or create new
            if parquet_path.exists():
                # Read existing and concatenate
                existing_table = pq.read_table(parquet_path)
                combined_table = pa.concat_tables([existing_table, table])
                pq.write_table(combined_table, parquet_path)
            else:
                pq.write_table(table, parquet_path)
            
            logger.info(f"Saved {len(chunks)} chunks to Parquet: {parquet_path}")
            
        except Exception as e:
            logger.error(f"Failed to save chunks to Parquet: {e}")
    
    def query(
        self, 
        corpus_name: str, 
        query: str, 
        top_k: int = None,
        model: str = None,
        use_hybrid: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Query the corpus and generate an answer using Ollama or Gemini"""
        # Search for relevant documents
        results = self.search(
            corpus_name=corpus_name,
            query=query,
            top_k=top_k or self.config.top_k,
            use_hybrid=use_hybrid
        )
        
        # Format results
        formatted_results = []
        for result in results:
            doc_id = result.get('document_id')
            if doc_id:
                document = self.get_document(corpus_name, doc_id)
                if document:
                    formatted_results.append({
                        "text": result["text"],
                        "score": result.get("score", 0.0),
                        "source_uri": document.source_uri,
                        "source_name": document.display_name,
                        "chunk_index": result.get("chunk_index", 0),
                        "metadata": result.get("metadata", {})
                    })
        
        # Generate answer
        answer = ""
        if formatted_results:
            context = "\n\n".join([
                f"[Source: {r['source_name']}]\n{r['text']}"
                for r in formatted_results[:top_k or self.config.top_k]
            ])
            
            prompt = f"""You are a helpful assistant. Use the following context to answer the question.
            If the answer cannot be found in the context, say so.
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:"""
            
            # Try Ollama first, then Gemini if available
            model = model or self.config.generation_model
            
            try:
                if self.ollama_client:
                    answer = self.ollama_client.generate_completion(
                        prompt,
                        model,
                        temperature=0.7,
                        max_tokens=500
                    )
                elif self.genai_model:
                    response = self.genai_model.generate_content(prompt)
                    answer = response.text
                else:
                    answer = "No generation model available."
            except Exception as e:
                answer = f"Error generating answer: {str(e)}"
        else:
            answer = "No relevant information found in the corpus."
        
        return {
            "query": query,
            "answer": answer,
            "sources": formatted_results,
            "corpus_name": corpus_name,
            "results_count": len(formatted_results),
            "model_used": model
        }
    
    def import_from_url(self, corpus_name: str, url: str) -> LocalDocument:
        """Import a document from a URL (Google Drive, etc.)"""
        import requests
        from urllib.parse import urlparse, parse_qs
        
        # Handle Google Drive URLs
        if "drive.google.com" in url:
            # Extract file ID from various Google Drive URL formats
            parsed = urlparse(url)
            file_id = None
            
            if "/file/d/" in url:
                # Format: https://drive.google.com/file/d/FILE_ID/...
                parts = url.split("/file/d/")[1].split("/")
                file_id = parts[0]
            elif "id=" in url:
                # Format: https://drive.google.com/open?id=FILE_ID
                params = parse_qs(parsed.query)
                file_id = params.get("id", [None])[0]
            
            if file_id:
                # Use direct download URL
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                
                try:
                    response = requests.get(download_url)
                    response.raise_for_status()
                    content = response.text
                    
                    return self.add_document(
                        corpus_name=corpus_name,
                        source_uri=url,
                        content=content,
                        display_name=f"Google Drive Document ({file_id})",
                        metadata={"source_type": "google_drive", "file_id": file_id}
                    )
                except Exception as e:
                    raise ValueError(f"Failed to download from Google Drive: {str(e)}")
            else:
                raise ValueError("Could not extract file ID from Google Drive URL")
        
        # Handle regular URLs
        else:
            try:
                response = requests.get(url)
                response.raise_for_status()
                content = response.text
                
                return self.add_document(
                    corpus_name=corpus_name,
                    source_uri=url,
                    content=content,
                    display_name=os.path.basename(urlparse(url).path) or url,
                    metadata={"source_type": "web_url"}
                )
            except Exception as e:
                raise ValueError(f"Failed to download from URL: {str(e)}")


    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available Ollama models"""
        models = self.ollama_client.list_models()
        
        embedding_models = []
        generation_models = []
        
        for model in models:
            model_name = model.get('name', '')
            
            # Categorize models
            if 'embed' in model_name.lower() or 'nomic' in model_name.lower():
                embedding_models.append(model_name)
            else:
                generation_models.append(model_name)
        
        return {
            'embedding_models': embedding_models,
            'generation_models': generation_models
        }
    
    def pull_model(self, model_name: str) -> bool:
        """Pull an Ollama model"""
        return self.ollama_client.pull_model(model_name)
    
    def set_generation_model(self, model_name: str):
        """Set the generation model for answers"""
        self.config.generation_model = model_name
        logger.info(f"Set generation model to: {model_name}")
    
    def set_embedding_model(self, model_name: str):
        """Set the embedding model and reinitialize if needed"""
        self.config.embedding_model = model_name
        self.use_ollama_embeddings = self._check_ollama_embedding_model()
        logger.info(f"Set embedding model to: {model_name}")
    
    def search_with_filter(self, corpus_name: str, query: str, where: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search with SQL-like WHERE clause filtering.
        
        Examples:
        - search_with_filter("my_corpus", "python tutorial", "metadata.category = 'programming'")
        - search_with_filter("docs", "machine learning", "chunk_index >= 2 AND document_id != 'old_doc'")
        """
        return self.search(
            corpus_name=corpus_name,
            query=query,
            top_k=top_k,
            where=where
        )
    
    def search_by_metadata(self, corpus_name: str, query: str, metadata_filters: Dict[str, Any], top_k: int = None) -> List[Dict[str, Any]]:
        """Search with metadata key-value filters.
        
        Examples:
        - search_by_metadata("corpus", "query", {"category": "tech", "priority": [1, 2, 3]})
        """
        return self.search(
            corpus_name=corpus_name,
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters
        )
    
    def get_parquet_data(self, corpus_name: str) -> Optional[pa.Table]:
        """Get the Parquet data for a corpus if available."""
        if not ARROW_AVAILABLE:
            logger.warning("PyArrow not available for Parquet access")
            return None
        
        try:
            parquet_path = self._get_corpus_path(corpus_name) / "chunks.parquet"
            if parquet_path.exists():
                return pq.read_table(parquet_path)
        except Exception as e:
            logger.error(f"Failed to read Parquet data: {e}")
        
        return None
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        
        if self.db:
            # LanceDB doesn't need explicit cleanup
            pass

# Global instance
local_rag_store = None

def get_local_rag_store(config: Optional[LocalRAGConfig] = None) -> LocalRAGStore:
    """Get or create the global LocalRAGStore instance"""
    global local_rag_store
    if local_rag_store is None:
        local_rag_store = LocalRAGStore(config=config)
    return local_rag_store