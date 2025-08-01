"""
Advanced document processing with intelligent chunking and layout awareness.
"""

try:
    from google.cloud import documentai_v1 as documentai
    DOCUMENTAI_AVAILABLE = True
except ImportError:
    DOCUMENTAI_AVAILABLE = False
    
try:
    from google.cloud import storage
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass
import logging
import numpy as np
import scipy.sparse as sp
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    from .mock_components import MockSentenceTransformer as SentenceTransformer

from ..config import settings
from .context_cache import context_cache_manager, initialize_context_cache

logger = logging.getLogger(__name__)

# Initialize context cache on module load
initialize_context_cache()

@dataclass
class Chunk:
    """Represents a document chunk with metadata"""
    text: str
    chunk_id: str
    document_id: str
    start_idx: int
    end_idx: int
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    sparse_embedding: Optional[sp.csr_matrix] = None

    
class IntelligentChunker:
    """Implements advanced chunking strategies"""
    
    def __init__(self, settings_obj: Optional[Any] = None):
        self.settings = settings_obj or settings
        self.setup_processors()
        
    def setup_processors(self):
        """Initialize document processors"""
        # Embedding model for semantic similarity
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        
        # Document AI client for layout parsing
        if self.settings.enable_layout_parser and DOCUMENTAI_AVAILABLE:
            try:
                self.docai_client = documentai.DocumentProcessorServiceClient()
                if self.settings.layout_processor_id:
                    self.layout_processor_name = (
                        f"projects/{self.settings.project_id}/"
                        f"locations/{self.settings.location}/"
                        f"processors/{self.settings.layout_processor_id}"
                    )
                else:
                    self.layout_processor_name = None
                    logger.warning("Layout processor ID not configured")
            except Exception as e:
                logger.warning(f"Failed to initialize Document AI client: {e}")
                self.settings.enable_layout_parser = False
        elif self.settings.enable_layout_parser:
            logger.warning("Document AI not available, disabling layout parser")
            self.settings.enable_layout_parser = False
    
    def chunk_document(
        self,
        text: str,
        document_id: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        use_semantic: bool = True
    ) -> List[Chunk]:
        """Chunk document with various strategies"""
        
        if chunk_size is None:
            chunk_size = self.settings.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.settings.chunk_overlap
            
        if use_semantic and self.settings.enable_semantic_segmentation:
            return self._semantic_chunking(
                text, document_id, chunk_size, chunk_overlap
            )
        else:
            return self._sliding_window_chunking(
                text, document_id, chunk_size, chunk_overlap
            )
    
    def _sliding_window_chunking(
        self,
        text: str,
        document_id: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Chunk]:
        """Basic sliding window chunking"""
        
        chunks = []
        tokens = text.split()
        
        for i in range(0, len(tokens), chunk_size - chunk_overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = " ".join(chunk_tokens)
            
            chunk = Chunk(
                text=chunk_text,
                chunk_id=f"{document_id}_chunk_{len(chunks)}",
                document_id=document_id,
                start_idx=i,
                end_idx=min(i + chunk_size, len(tokens)),
                metadata={
                    "chunking_method": "sliding_window",
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }
            )
            chunks.append(chunk)
            
        return chunks
    
    def _semantic_chunking(
        self,
        text: str,
        document_id: str,
        target_chunk_size: int,
        chunk_overlap: int
    ) -> List[Chunk]:
        """Semantic chunking based on sentence similarity"""
        
        # Split into sentences
        sentences = self._split_sentences(text)
        if not sentences:
            return self._sliding_window_chunking(
                text, document_id, target_chunk_size, chunk_overlap
            )
            
        # Calculate sentence embeddings
        sentence_embeddings = self.embedding_model.encode(sentences)
        
        # Group sentences into chunks based on similarity
        chunks = []
        current_chunk = [sentences[0]]
        current_chunk_embedding = sentence_embeddings[0]
        current_tokens = len(sentences[0].split())
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_tokens = len(sentence.split())
            
            # Check if adding this sentence would exceed target size
            if current_tokens + sentence_tokens > target_chunk_size:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=f"{document_id}_chunk_{len(chunks)}",
                    document_id=document_id,
                    start_idx=len(chunks) * target_chunk_size,
                    end_idx=(len(chunks) + 1) * target_chunk_size,
                    metadata={
                        "chunking_method": "semantic",
                        "num_sentences": len(current_chunk)
                    }
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = max(1, chunk_overlap // 20)  # Rough estimate
                current_chunk = current_chunk[-overlap_sentences:] + [sentence]
                current_tokens = sum(len(s.split()) for s in current_chunk)
            else:
                # Calculate similarity
                similarity = np.dot(current_chunk_embedding, sentence_embeddings[i]) / (
                    np.linalg.norm(current_chunk_embedding) * np.linalg.norm(sentence_embeddings[i])
                )
                
                # Add to current chunk if similar enough
                if similarity > 0.7:  # Threshold for semantic similarity
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
                    # Update chunk embedding
                    current_chunk_embedding = np.mean(
                        sentence_embeddings[i-len(current_chunk)+1:i+1],
                        axis=0
                    )
                else:
                    # Start new chunk due to semantic boundary
                    chunk_text = " ".join(current_chunk)
                    chunk = Chunk(
                        text=chunk_text,
                        chunk_id=f"{document_id}_chunk_{len(chunks)}",
                        document_id=document_id,
                        start_idx=len(chunks) * target_chunk_size,
                        end_idx=(len(chunks) + 1) * target_chunk_size,
                        metadata={
                            "chunking_method": "semantic",
                            "num_sentences": len(current_chunk),
                            "semantic_boundary": True
                        }
                    )
                    chunks.append(chunk)
                    
                    current_chunk = [sentence]
                    current_chunk_embedding = sentence_embeddings[i]
                    current_tokens = sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk = Chunk(
                text=chunk_text,
                chunk_id=f"{document_id}_chunk_{len(chunks)}",
                document_id=document_id,
                start_idx=len(chunks) * target_chunk_size,
                end_idx=len(text.split()),
                metadata={
                    "chunking_method": "semantic",
                    "num_sentences": len(current_chunk)
                }
            )
            chunks.append(chunk)
            
        return chunks
    
    def apply_late_chunking(
        self,
        document_text: str,
        embeddings: np.ndarray,
        chunk_size: int = 512
    ) -> List[Chunk]:
        """Apply late chunking after embedding generation"""
        
        # Tokenize document
        tokens = document_text.split()
        
        # Calculate token-level similarities
        token_similarities = self._calculate_token_similarities(embeddings)
        
        # Find optimal chunk boundaries
        boundaries = self._find_chunk_boundaries(
            token_similarities,
            chunk_size
        )
        
        # Create chunks
        chunks = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            chunk_text = " ".join(tokens[start_idx:end_idx])
            chunk_embedding = np.mean(
                embeddings[start_idx:end_idx],
                axis=0
            )
            
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"late_chunk_{i}",
                document_id="unknown",
                start_idx=start_idx,
                end_idx=end_idx,
                metadata={
                    "late_chunked": True,
                    "original_length": len(tokens)
                },
                embedding=chunk_embedding
            ))
            
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - can be improved with NLTK or spaCy
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_token_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate similarities between adjacent tokens"""
        similarities = []
        
        for i in range(len(embeddings) - 1):
            similarity = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
            similarities.append(similarity)
            
        return np.array(similarities)
    
    def _find_chunk_boundaries(
        self,
        similarities: np.ndarray,
        target_size: int
    ) -> List[int]:
        """Find optimal chunk boundaries based on similarities"""
        
        boundaries = [0]
        current_size = 0
        
        for i in range(1, len(similarities)):
            current_size += 1
            
            # Check if we should create boundary
            if current_size >= target_size:
                # Find local minimum in similarity
                window_start = max(0, i - 10)
                window_end = min(len(similarities), i + 10)
                window = similarities[window_start:window_end]
                
                if len(window) > 0:
                    local_min_idx = np.argmin(window) + window_start
                    boundaries.append(local_min_idx)
                    current_size = 0
                    
        boundaries.append(len(similarities))
        return boundaries


class DocumentProcessor:
    """Main document processing orchestrator"""
    
    def __init__(self, settings_obj: Optional[Any] = None):
        self.settings = settings_obj or settings
        self.chunker = IntelligentChunker(settings_obj)
        
    def process_documents(
        self,
        corpus_id: str,
        uris: List[str],
        use_layout_parser: bool = True,
        enable_late_chunking: bool = True,
        create_context_cache: bool = True
    ) -> Dict[str, Any]:
        """Process documents with advanced techniques and optional caching"""
        
        results = {
            "processed_count": 0,
            "total_chunks": 0,
            "processing_time": 0,
            "errors": [],
            "cache_created": False,
            "cache_id": None
        }
        
        import time
        start_time = time.time()
        
        all_chunks = []
        all_texts = []
        
        for uri in uris:
            try:
                # Load document content
                content = self._load_document(uri)
                
                # Generate document ID
                doc_id = f"doc_{corpus_id}_{results['processed_count']}"
                
                # Chunk document
                chunks = self.chunker.chunk_document(
                    content,
                    doc_id,
                    use_semantic=True
                )
                
                all_chunks.extend(chunks)
                all_texts.append(content)
                
                results["processed_count"] += 1
                results["total_chunks"] += len(chunks)
                
            except Exception as e:
                logger.error(f"Error processing {uri}: {e}")
                results["errors"].append({
                    "uri": uri,
                    "error": str(e)
                })
        
        # Create context cache if enabled and documents were processed
        if (create_context_cache and 
            context_cache_manager and 
            settings.cache_corpus_documents and
            all_texts):
            try:
                cache_entry = context_cache_manager.create_corpus_cache(
                    corpus_id=corpus_id,
                    documents=all_texts,
                    system_instruction=(
                        f"You are analyzing documents from corpus '{corpus_id}'. "
                        f"This corpus contains {len(all_texts)} documents with {len(all_chunks)} chunks. "
                        "Use this information to answer questions accurately based on the content."
                    ),
                    ttl_seconds=settings.context_cache_ttl
                )
                
                if cache_entry:
                    results["cache_created"] = True
                    results["cache_id"] = cache_entry.cache_id
                    results["cached_tokens"] = cache_entry.token_count
                    logger.info(f"Created context cache for corpus {corpus_id}: {cache_entry.cache_id}")
                    
            except Exception as e:
                logger.error(f"Failed to create context cache: {e}")
                
        results["processing_time"] = time.time() - start_time
        results["chunks"] = all_chunks  # Store chunks for indexing
        return results
    
    def _load_document(self, uri: str) -> str:
        """Load document content from URI"""
        # Simple implementation - in production, handle different sources
        # (Google Drive, GCS, local files, etc.)
        
        if uri.startswith("gs://"):
            # Google Cloud Storage
            return self._load_from_gcs(uri)
        elif uri.startswith("https://drive.google.com"):
            # Google Drive
            return self._load_from_drive(uri)
        else:
            # Local file
            with open(uri, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _load_from_gcs(self, uri: str) -> str:
        """Load document from Google Cloud Storage"""
        if not STORAGE_AVAILABLE:
            logger.warning(f"Google Cloud Storage not available, cannot load: {uri}")
            return "Sample document content"
            
        # Parse bucket and blob name
        parts = uri.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        return blob.download_as_text()
    
    def _load_from_drive(self, uri: str) -> str:
        """Load document from Google Drive"""
        # This is a placeholder - implement actual Google Drive API integration
        logger.warning(f"Google Drive loading not implemented for: {uri}")
        return "Document content from Google Drive"