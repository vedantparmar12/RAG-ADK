"""
Multi-provider embedding support including Gemini API embeddings.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Literal
from enum import Enum
import logging
from abc import ABC, abstractmethod

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    
try:
    import vertexai
    from vertexai.language_models import TextEmbeddingModel
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False

from sklearn.metrics.pairwise import cosine_similarity
from ..config import settings
from .rate_limiter import RateLimiter, RateLimitTier, BatchProcessor

logger = logging.getLogger(__name__)


class EmbeddingProvider(str, Enum):
    """Available embedding providers"""
    GEMINI = "gemini"
    VERTEX_AI = "vertex_ai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    MULTIMODAL = "multimodal"


class TaskType(str, Enum):
    """Gemini embedding task types"""
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    CODE_RETRIEVAL_QUERY = "CODE_RETRIEVAL_QUERY"
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    FACT_VERIFICATION = "FACT_VERIFICATION"


class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    
    @abstractmethod
    def encode(
        self, 
        texts: Union[str, List[str]], 
        **kwargs
    ) -> np.ndarray:
        """Encode texts into embeddings"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass


class GeminiEmbedding(EmbeddingModel):
    """Gemini API embedding model with rate limiting"""
    
    def __init__(
        self,
        model_name: str = "gemini-embedding-001",
        task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT,
        output_dimensionality: Optional[int] = None,
        rate_limit_tier: RateLimitTier = RateLimitTier.FREE,
        batch_size: int = 10
    ):
        if not GENAI_AVAILABLE:
            raise ImportError("Google GenAI library not available. Install with: pip install google-genai")
            
        self.model_name = model_name
        self.task_type = task_type
        self.output_dimensionality = output_dimensionality or 768  # Default to 768
        self.client = genai.Client()
        self.batch_size = batch_size
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            model="gemini-embedding-001",
            tier=rate_limit_tier
        )
        
        # Validate output dimensionality
        if self.output_dimensionality not in [768, 1536, 3072]:
            logger.warning(f"Non-standard output dimensionality: {self.output_dimensionality}. Recommended: 768, 1536, or 3072")
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        task_type: Optional[TaskType] = None,
        normalize: bool = True,
        show_progress: bool = False,
        **kwargs
    ) -> np.ndarray:
        """Encode texts using Gemini embedding API with rate limiting"""
        
        if isinstance(texts, str):
            texts = [texts]
            
        task_type = task_type or self.task_type
        
        # Process in batches for rate limiting
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Estimate tokens for rate limiting
            estimated_tokens = sum(self.rate_limiter.estimate_tokens(text) for text in batch)
            
            if show_progress and i > 0:
                logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
            
            try:
                # Use rate limiter for API call
                import asyncio
                
                async def _embed_batch():
                    return await self.rate_limiter.execute_with_retry(
                        self._embed_content,
                        batch,
                        task_type,
                        estimated_tokens=estimated_tokens
                    )
                
                # Run async function
                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(_embed_batch())
                loop.close()
                
                # Extract embeddings
                batch_embeddings = np.array([e.values for e in result.embeddings])
                all_embeddings.append(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error generating Gemini embeddings for batch {i//self.batch_size}: {e}")
                raise
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
        
        # Normalize if requested (3072 dim is already normalized)
        if normalize and embeddings.size > 0 and self.output_dimensionality != 3072:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
        return embeddings
    
    def _embed_content(self, texts: List[str], task_type: TaskType):
        """Internal method to embed content"""
        return self.client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config={
                "task_type": task_type.value,
                "output_dimensionality": self.output_dimensionality
            }
        )
    
    def get_dimension(self) -> int:
        return self.output_dimensionality
    
    def get_rate_limit_metrics(self) -> Dict[str, Any]:
        """Get current rate limit metrics"""
        return self.rate_limiter.get_metrics()


class MultimodalEmbedding(EmbeddingModel):
    """Multimodal embedding model for images, text, and video"""
    
    def __init__(self, model_name: str = "multimodalembedding"):
        if not VERTEX_AI_AVAILABLE:
            raise ImportError("Vertex AI not available. Install with: pip install google-cloud-aiplatform")
            
        self.model_name = model_name
        self.dimension = 1408  # Fixed dimension for multimodal embeddings
        
        # Initialize Vertex AI
        vertexai.init(project=settings.project_id, location=settings.location)
        
    def encode(
        self, 
        contents: Union[str, List[str], Dict[str, Any]], 
        **kwargs
    ) -> np.ndarray:
        """Encode multimodal content"""
        
        # TODO: Implement multimodal embedding logic
        # This would handle images, text, and video inputs
        logger.warning("Multimodal embedding not fully implemented yet")
        
        # Return mock embeddings for now
        if isinstance(contents, str):
            return np.random.randn(1, self.dimension)
        elif isinstance(contents, list):
            return np.random.randn(len(contents), self.dimension)
        else:
            return np.random.randn(1, self.dimension)
    
    def get_dimension(self) -> int:
        return self.dimension


class SentenceTransformerEmbedding(EmbeddingModel):
    """Sentence transformer embedding model"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("Sentence transformers not available. Install with: pip install sentence-transformers")
            
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
    def encode(
        self, 
        texts: Union[str, List[str]], 
        normalize: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode texts using sentence transformers"""
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            **kwargs
        )
        
        return embeddings
    
    def get_dimension(self) -> int:
        return self.dimension


class VertexAIEmbedding(EmbeddingModel):
    """Vertex AI text embedding model"""
    
    def __init__(self, model_name: str = "text-embedding-005"):
        if not VERTEX_AI_AVAILABLE:
            raise ImportError("Vertex AI not available. Install with: pip install google-cloud-aiplatform")
            
        self.model_name = model_name
        self.model = TextEmbeddingModel.from_pretrained(model_name)
        self.dimension = 768  # Standard dimension for Vertex AI embeddings
        
    def encode(
        self, 
        texts: Union[str, List[str]], 
        **kwargs
    ) -> np.ndarray:
        """Encode texts using Vertex AI"""
        
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = self.model.get_embeddings(texts)
        embedding_values = np.array([e.values for e in embeddings])
        
        return embedding_values
    
    def get_dimension(self) -> int:
        return self.dimension


class EmbeddingManager:
    """Manages multiple embedding providers and configurations"""
    
    def __init__(self, default_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS):
        self.default_provider = default_provider
        self.models = {}
        self._initialize_default_models()
        
    def _initialize_default_models(self):
        """Initialize default embedding models"""
        
        # Initialize sentence transformers if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.models[EmbeddingProvider.SENTENCE_TRANSFORMERS] = SentenceTransformerEmbedding()
                logger.info("Sentence transformers initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize sentence transformers: {e}")
                
        # Initialize Gemini if available
        if GENAI_AVAILABLE:
            try:
                self.models[EmbeddingProvider.GEMINI] = GeminiEmbedding()
                logger.info("Gemini embeddings initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini embeddings: {e}")
                
        # Initialize Vertex AI if available
        if VERTEX_AI_AVAILABLE and settings.use_vertex_ai:
            try:
                self.models[EmbeddingProvider.VERTEX_AI] = VertexAIEmbedding(
                    model_name=settings.embedding_model
                )
                logger.info("Vertex AI embeddings initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Vertex AI embeddings: {e}")
    
    def add_model(
        self, 
        provider: EmbeddingProvider, 
        model: EmbeddingModel
    ):
        """Add a custom embedding model"""
        self.models[provider] = model
        
    def get_model(
        self, 
        provider: Optional[EmbeddingProvider] = None
    ) -> EmbeddingModel:
        """Get embedding model by provider"""
        
        provider = provider or self.default_provider
        
        if provider not in self.models:
            raise ValueError(f"Provider {provider} not available. Available: {list(self.models.keys())}")
            
        return self.models[provider]
    
    def encode(
        self,
        texts: Union[str, List[str]],
        provider: Optional[EmbeddingProvider] = None,
        **kwargs
    ) -> np.ndarray:
        """Encode texts using specified provider"""
        
        model = self.get_model(provider)
        return model.encode(texts, **kwargs)
    
    def encode_with_task_type(
        self,
        queries: List[str],
        documents: List[str],
        provider: EmbeddingProvider = EmbeddingProvider.GEMINI,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Encode queries and documents with appropriate task types"""
        
        if provider != EmbeddingProvider.GEMINI:
            # For non-Gemini providers, use standard encoding
            return {
                "queries": self.encode(queries, provider, **kwargs),
                "documents": self.encode(documents, provider, **kwargs)
            }
            
        # For Gemini, use appropriate task types
        model = self.get_model(provider)
        
        query_embeddings = model.encode(
            queries, 
            task_type=TaskType.RETRIEVAL_QUERY,
            **kwargs
        )
        
        document_embeddings = model.encode(
            documents,
            task_type=TaskType.RETRIEVAL_DOCUMENT,
            **kwargs
        )
        
        return {
            "queries": query_embeddings,
            "documents": document_embeddings
        }
    
    def compute_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between embeddings"""
        return cosine_similarity(embeddings1, embeddings2)


# Global embedding manager instance
embedding_manager = EmbeddingManager()