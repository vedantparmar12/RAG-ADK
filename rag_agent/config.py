import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Enhanced RAG Configuration Settings"""
    
    # Google Cloud Settings
    project_id: str = Field(
        default=os.environ.get("GOOGLE_CLOUD_PROJECT", "charming-module-240007"),
        env="GOOGLE_CLOUD_PROJECT"
    )
    location: str = Field(
        default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        env="GOOGLE_CLOUD_LOCATION"
    )
    use_vertex_ai: bool = Field(True, env="GOOGLE_GENAI_USE_VERTEXAI")
    
    # Model Settings
    embedding_model: str = Field(default="text-embedding-005", env="EMBEDDING_MODEL")
    generation_model: str = Field(default="gemini-2.0-flash", env="GENERATION_MODEL")
    reranking_model: str = Field(default="semantic-ranker-512@latest", env="RERANKING_MODEL")
    
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
