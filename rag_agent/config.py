"""
Enhanced configuration settings for the RAG Agent.

These settings are used by the various RAG tools and support advanced features.
Vertex AI initialization is performed in the package's __init__.py
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Enhanced RAG Configuration Settings"""
    
    # API Configuration
    google_api_key: Optional[str] = Field(default=os.environ.get("GOOGLE_API_KEY"), env="GOOGLE_API_KEY")
    use_vertex_ai: bool = Field(False, env="GOOGLE_GENAI_USE_VERTEXAI")
    
    # Google Cloud Settings (for Vertex AI mode)
    project_id: str = Field(
        default=os.environ.get("GOOGLE_CLOUD_PROJECT", "charming-module-240007"),
        env="GOOGLE_CLOUD_PROJECT"
    )
    location: str = Field(
        default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        env="GOOGLE_CLOUD_LOCATION"
    )
    
    # Model Settings
    embedding_model: str = Field(default="text-embedding-005", env="EMBEDDING_MODEL")
    embedding_provider: str = Field(default="sentence_transformers", env="EMBEDDING_PROVIDER", pattern="^(gemini|vertex_ai|sentence_transformers|multimodal)$")
    gemini_embedding_model: str = Field(default="gemini-embedding-001", env="GEMINI_EMBEDDING_MODEL")
    gemini_output_dimensionality: int = Field(default=768, env="GEMINI_OUTPUT_DIM", ge=1, le=3072)
    gemini_task_type: str = Field(default="RETRIEVAL_DOCUMENT", env="GEMINI_TASK_TYPE")
    generation_model: str = Field(default="deepseek-coder", env="GENERATION_MODEL")
    reranking_model: str = Field(default="semantic-ranker-512@latest", env="RERANKING_MODEL")
    
    # Rate Limiting Settings
    rate_limit_tier: str = Field(default="free", env="RATE_LIMIT_TIER", pattern="^(free|tier_1|tier_2|tier_3)$")
    embedding_batch_size: int = Field(default=10, env="EMBEDDING_BATCH_SIZE", ge=1, le=100)
    rate_limit_retry_attempts: int = Field(default=3, env="RATE_LIMIT_RETRY_ATTEMPTS", ge=1, le=10)
    rate_limit_base_delay: float = Field(default=1.0, env="RATE_LIMIT_BASE_DELAY", ge=0.1, le=10.0)
    
    # Context Caching Settings
    enable_context_caching: bool = Field(default=True, env="ENABLE_CONTEXT_CACHING")
    context_cache_ttl: int = Field(default=3600, env="CONTEXT_CACHE_TTL", ge=300, le=86400)  # 5 min to 24 hours
    min_tokens_for_cache: int = Field(default=1024, env="MIN_TOKENS_FOR_CACHE", ge=256, le=10000)
    cache_corpus_documents: bool = Field(default=True, env="CACHE_CORPUS_DOCUMENTS")
    cache_query_context: bool = Field(default=True, env="CACHE_QUERY_CONTEXT")
    
    # Indexing Configuration
    indexing_strategy: str = Field(default="hybrid", pattern="^(dense|sparse|hybrid)$")
    chunk_size: int = Field(default=512, ge=128, le=2048)
    chunk_overlap: int = Field(default=100, ge=0, le=256)
    enable_late_chunking: bool = Field(default=True)
    enable_colbert: bool = Field(default=True)
    enable_layout_parser: bool = Field(default=True)
    enable_clustering: bool = Field(default=False)
    
    # Retrieval Configuration
    top_k_retrieval: int = Field(default=10, ge=1, le=100)
    enable_reranking: bool = Field(default=True)
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    distance_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Advanced Reranking Configuration
    enable_coherence_reranking: bool = Field(default=True, env="ENABLE_COHERENCE_RERANKING")
    enable_diversity_reranking: bool = Field(default=True, env="ENABLE_DIVERSITY_RERANKING")
    coherence_weight: float = Field(default=0.3, env="COHERENCE_WEIGHT", ge=0.0, le=1.0)
    diversity_weight: float = Field(default=0.2, env="DIVERSITY_WEIGHT", ge=0.0, le=1.0)
    diversity_lambda: float = Field(default=0.5, env="DIVERSITY_LAMBDA", ge=0.0, le=1.0)
    
    # Performance Settings
    enable_caching: bool = Field(default=True)
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    enable_context_pruning: bool = Field(default=True)
    enable_attention_pruning: bool = Field(default=False)
    enable_sentence_pruning: bool = Field(default=True)
    pruning_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    max_context_tokens: int = Field(default=8192, ge=1024, le=32768)
    
    # Advanced Features
    enable_query_expansion: bool = Field(default=True)
    enable_semantic_segmentation: bool = Field(default=True)
    enable_multi_stage_ranking: bool = Field(default=True)
    
    # Redis Configuration (for caching)
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    
    # Qdrant Configuration (Vector Database)
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_collection_prefix: str = Field(default="rag_corpus", env="QDRANT_COLLECTION_PREFIX")
    qdrant_timeout: float = Field(default=60.0, env="QDRANT_TIMEOUT")
    qdrant_batch_size: int = Field(default=100, env="QDRANT_BATCH_SIZE", ge=1, le=1000)
    
    # Document AI Configuration
    layout_processor_id: Optional[str] = Field(default=None, env="LAYOUT_PROCESSOR_ID")
    
    # Legacy RAG settings (for backward compatibility)
    DEFAULT_CHUNK_SIZE: int = 512
    DEFAULT_CHUNK_OVERLAP: int = 100
    DEFAULT_TOP_K: int = 3
    DEFAULT_DISTANCE_THRESHOLD: float = 0.5
    DEFAULT_EMBEDDING_MODEL: str = "publishers/google/models/text-embedding-005"
    DEFAULT_EMBEDDING_REQUESTS_PER_MIN: int = 1000
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from environment

# Create global settings instance
settings = Settings()

# Export legacy variables for backward compatibility
PROJECT_ID = settings.project_id
LOCATION = settings.location
DEFAULT_CHUNK_SIZE = settings.DEFAULT_CHUNK_SIZE
DEFAULT_CHUNK_OVERLAP = settings.DEFAULT_CHUNK_OVERLAP
DEFAULT_TOP_K = settings.DEFAULT_TOP_K
DEFAULT_DISTANCE_THRESHOLD = settings.DEFAULT_DISTANCE_THRESHOLD
DEFAULT_EMBEDDING_MODEL = settings.DEFAULT_EMBEDDING_MODEL
DEFAULT_EMBEDDING_REQUESTS_PER_MIN = settings.DEFAULT_EMBEDDING_REQUESTS_PER_MIN