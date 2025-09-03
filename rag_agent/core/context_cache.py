"""
Gemini API Context Caching for optimized embeddings and generation.
"""

import hashlib
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import logging
from enum import Enum

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from ..config import settings

logger = logging.getLogger(__name__)


class CacheType(Enum):
    """Types of cached content"""
    CORPUS_DOCUMENTS = "corpus_documents"
    SYSTEM_INSTRUCTIONS = "system_instructions"
    EMBEDDING_BATCH = "embedding_batch"
    QUERY_CONTEXT = "query_context"


@dataclass
class CacheEntry:
    """Represents a cached content entry"""
    cache_id: str
    cache_name: str  # Gemini cache name
    cache_type: CacheType
    model: str
    content_hash: str
    token_count: int
    created_at: datetime
    expire_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class GeminiContextCache:
    """Manages context caching for Gemini API calls"""
    
    def __init__(
        self,
        default_ttl_seconds: int = 3600,  # 1 hour default
        min_tokens_for_cache: int = 1024,  # Minimum tokens to cache
        cache_model: str = "models/gemini-1.5-flash-002"
    ):
        if not GENAI_AVAILABLE:
            raise ImportError("Google GenAI library required for context caching")
            
        self.client = genai.Client()
        self.default_ttl = default_ttl_seconds
        self.min_tokens = min_tokens_for_cache
        self.cache_model = cache_model
        
        # Local cache registry
        self.cache_registry: Dict[str, CacheEntry] = {}
        
        # Initialize by syncing with API
        self._sync_cache_registry()
        
    def _sync_cache_registry(self):
        """Sync local registry with Gemini API caches"""
        try:
            for cache in self.client.caches.list():
                # Extract metadata from display name if formatted correctly
                if cache.display_name and "|" in cache.display_name:
                    cache_type_str, content_hash = cache.display_name.split("|", 1)
                    try:
                        cache_type = CacheType(cache_type_str)
                        
                        entry = CacheEntry(
                            cache_id=f"{cache_type.value}_{content_hash[:8]}",
                            cache_name=cache.name,
                            cache_type=cache_type,
                            model=cache.model,
                            content_hash=content_hash,
                            token_count=cache.usage_metadata.total_token_count,
                            created_at=cache.create_time,
                            expire_time=cache.expire_time
                        )
                        
                        self.cache_registry[entry.cache_id] = entry
                        
                    except ValueError:
                        logger.warning(f"Unknown cache format: {cache.display_name}")
                        
        except Exception as e:
            logger.error(f"Failed to sync cache registry: {e}")
    
    def _generate_content_hash(self, content: Union[str, List[str], List[Any]]) -> str:
        """Generate hash for content to check if already cached"""
        if isinstance(content, list):
            content_str = "|".join(str(c) for c in content)
        else:
            content_str = str(content)
            
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def _estimate_tokens(self, content: Union[str, List[str]]) -> int:
        """Estimate token count for content"""
        if isinstance(content, list):
            total_chars = sum(len(str(c)) for c in content)
        else:
            total_chars = len(str(content))
            
        # Rough estimate: 1 token ≈ 4 characters
        return total_chars // 4
    
    def create_corpus_cache(
        self,
        corpus_id: str,
        documents: List[str],
        system_instruction: Optional[str] = None,
        ttl_seconds: Optional[int] = None
    ) -> Optional[CacheEntry]:
        """Create cache for corpus documents"""
        
        # Check token count
        estimated_tokens = self._estimate_tokens(documents)
        if estimated_tokens < self.min_tokens:
            logger.info(f"Document set too small for caching: {estimated_tokens} tokens")
            return None
            
        # Generate content hash
        content_hash = self._generate_content_hash(documents)
        cache_id = f"{CacheType.CORPUS_DOCUMENTS.value}_{content_hash[:8]}"
        
        # Check if already cached
        if cache_id in self.cache_registry:
            existing = self.cache_registry[cache_id]
            if existing.expire_time > datetime.now(timezone.utc):
                logger.info(f"Using existing cache: {cache_id}")
                return existing
                
        # Create new cache
        try:
            ttl = ttl_seconds or self.default_ttl
            
            # Build cache content
            contents = []
            for doc in documents:
                contents.append({"text": doc})
                
            cache_config = types.CreateCachedContentConfig(
                display_name=f"{CacheType.CORPUS_DOCUMENTS.value}|{content_hash}",
                system_instruction=system_instruction or (
                    "You are analyzing a corpus of documents. "
                    "Use these documents to answer questions accurately."
                ),
                contents=contents,
                ttl=f"{ttl}s"
            )
            
            cache = self.client.caches.create(
                model=self.cache_model,
                config=cache_config
            )
            
            # Create cache entry
            entry = CacheEntry(
                cache_id=cache_id,
                cache_name=cache.name,
                cache_type=CacheType.CORPUS_DOCUMENTS,
                model=cache.model,
                content_hash=content_hash,
                token_count=cache.usage_metadata.total_token_count,
                created_at=cache.create_time,
                expire_time=cache.expire_time,
                metadata={"corpus_id": corpus_id, "document_count": len(documents)}
            )
            
            self.cache_registry[cache_id] = entry
            logger.info(f"Created corpus cache: {cache_id} ({entry.token_count} tokens)")
            
            return entry
            
        except Exception as e:
            logger.error(f"Failed to create corpus cache: {e}")
            return None
    
    def create_embedding_cache(
        self,
        texts: List[str],
        embedding_instruction: str = "Generate embeddings for these texts",
        ttl_seconds: Optional[int] = None
    ) -> Optional[CacheEntry]:
        """Create cache for embedding generation"""
        
        estimated_tokens = self._estimate_tokens(texts)
        if estimated_tokens < self.min_tokens:
            return None
            
        content_hash = self._generate_content_hash(texts)
        cache_id = f"{CacheType.EMBEDDING_BATCH.value}_{content_hash[:8]}"
        
        if cache_id in self.cache_registry:
            existing = self.cache_registry[cache_id]
            if existing.expire_time > datetime.now(timezone.utc):
                return existing
                
        try:
            ttl = ttl_seconds or 1800  # 30 min for embeddings
            
            contents = [{"text": f"Text {i+1}: {text}"} for i, text in enumerate(texts)]
            
            cache_config = types.CreateCachedContentConfig(
                display_name=f"{CacheType.EMBEDDING_BATCH.value}|{content_hash}",
                system_instruction=embedding_instruction,
                contents=contents,
                ttl=f"{ttl}s"
            )
            
            cache = self.client.caches.create(
                model=self.cache_model,
                config=cache_config
            )
            
            entry = CacheEntry(
                cache_id=cache_id,
                cache_name=cache.name,
                cache_type=CacheType.EMBEDDING_BATCH,
                model=cache.model,
                content_hash=content_hash,
                token_count=cache.usage_metadata.total_token_count,
                created_at=cache.create_time,
                expire_time=cache.expire_time,
                metadata={"text_count": len(texts)}
            )
            
            self.cache_registry[cache_id] = entry
            logger.info(f"Created embedding cache: {cache_id} ({entry.token_count} tokens)")
            
            return entry
            
        except Exception as e:
            logger.error(f"Failed to create embedding cache: {e}")
            return None
    
    def create_query_context_cache(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        system_prompt: str,
        ttl_seconds: Optional[int] = None
    ) -> Optional[CacheEntry]:
        """Cache retrieved context for query processing"""
        
        # Extract text from chunks
        texts = [chunk.get("text", "") for chunk in retrieved_chunks]
        estimated_tokens = self._estimate_tokens(texts)
        
        if estimated_tokens < self.min_tokens:
            return None
            
        content_hash = self._generate_content_hash(texts + [system_prompt])
        cache_id = f"{CacheType.QUERY_CONTEXT.value}_{content_hash[:8]}"
        
        if cache_id in self.cache_registry:
            existing = self.cache_registry[cache_id]
            if existing.expire_time > datetime.now(timezone.utc):
                return existing
                
        try:
            ttl = ttl_seconds or 900  # 15 min for query context
            
            contents = []
            for i, chunk in enumerate(retrieved_chunks):
                contents.append({
                    "text": f"[Document {i+1}]\n{chunk.get('text', '')}\n"
                            f"Metadata: {chunk.get('metadata', {})}"
                })
                
            cache_config = types.CreateCachedContentConfig(
                display_name=f"{CacheType.QUERY_CONTEXT.value}|{content_hash}",
                system_instruction=system_prompt,
                contents=contents,
                ttl=f"{ttl}s"
            )
            
            cache = self.client.caches.create(
                model=self.cache_model,
                config=cache_config
            )
            
            entry = CacheEntry(
                cache_id=cache_id,
                cache_name=cache.name,
                cache_type=CacheType.QUERY_CONTEXT,
                model=cache.model,
                content_hash=content_hash,
                token_count=cache.usage_metadata.total_token_count,
                created_at=cache.create_time,
                expire_time=cache.expire_time,
                metadata={"chunk_count": len(retrieved_chunks)}
            )
            
            self.cache_registry[cache_id] = entry
            logger.info(f"Created query context cache: {cache_id} ({entry.token_count} tokens)")
            
            return entry
            
        except Exception as e:
            logger.error(f"Failed to create query context cache: {e}")
            return None
    
    def generate_with_cache(
        self,
        prompt: str,
        cache_entry: Optional[CacheEntry] = None,
        cache_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate content using cached context"""
        
        if not cache_entry and not cache_name:
            raise ValueError("Either cache_entry or cache_name must be provided")
            
        cache_to_use = cache_name or cache_entry.cache_name
        
        try:
            response = self.client.models.generate_content(
                model=self.cache_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    cached_content=cache_to_use,
                    **kwargs
                )
            )
            
            # Log token usage
            usage = response.usage_metadata
            logger.info(
                f"Generated with cache - Total tokens: {usage.total_token_count}, "
                f"Cached: {usage.cached_content_token_count}, "
                f"New: {usage.prompt_token_count}"
            )
            
            return {
                "text": response.text,
                "usage": {
                    "total_tokens": usage.total_token_count,
                    "cached_tokens": usage.cached_content_token_count,
                    "prompt_tokens": usage.prompt_token_count,
                    "completion_tokens": usage.candidates_token_count
                },
                "cache_used": cache_to_use
            }
            
        except Exception as e:
            logger.error(f"Failed to generate with cache: {e}")
            raise
    
    def update_cache_ttl(
        self,
        cache_id: str,
        new_ttl_seconds: Optional[int] = None,
        new_expire_time: Optional[datetime] = None
    ) -> bool:
        """Update cache TTL or expiration time"""
        
        if cache_id not in self.cache_registry:
            logger.error(f"Cache not found: {cache_id}")
            return False
            
        entry = self.cache_registry[cache_id]
        
        try:
            if new_ttl_seconds:
                config = types.UpdateCachedContentConfig(ttl=f"{new_ttl_seconds}s")
            elif new_expire_time:
                if new_expire_time.tzinfo is None:
                    new_expire_time = new_expire_time.replace(tzinfo=timezone.utc)
                config = types.UpdateCachedContentConfig(expire_time=new_expire_time)
            else:
                return False
                
            updated = self.client.caches.update(
                name=entry.cache_name,
                config=config
            )
            
            # Update local entry
            entry.expire_time = updated.expire_time
            logger.info(f"Updated cache TTL: {cache_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update cache: {e}")
            return False
    
    def delete_cache(self, cache_id: str) -> bool:
        """Delete a cache"""
        
        if cache_id not in self.cache_registry:
            return False
            
        entry = self.cache_registry[cache_id]
        
        try:
            self.client.caches.delete(entry.cache_name)
            del self.cache_registry[cache_id]
            logger.info(f"Deleted cache: {cache_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete cache: {e}")
            return False
    
    def cleanup_expired_caches(self) -> int:
        """Remove expired caches from registry"""
        
        current_time = datetime.now(timezone.utc)
        expired = []
        
        for cache_id, entry in self.cache_registry.items():
            if entry.expire_time < current_time:
                expired.append(cache_id)
                
        for cache_id in expired:
            del self.cache_registry[cache_id]
            
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired caches")
            
        return len(expired)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get statistics about cached content"""
        
        stats = {
            "total_caches": len(self.cache_registry),
            "total_cached_tokens": 0,
            "caches_by_type": {},
            "expiring_soon": []
        }
        
        current_time = datetime.now(timezone.utc)
        one_hour_later = current_time + timedelta(hours=1)
        
        for entry in self.cache_registry.values():
            stats["total_cached_tokens"] += entry.token_count
            
            # Count by type
            cache_type = entry.cache_type.value
            if cache_type not in stats["caches_by_type"]:
                stats["caches_by_type"][cache_type] = 0
            stats["caches_by_type"][cache_type] += 1
            
            # Check if expiring soon
            if entry.expire_time < one_hour_later:
                stats["expiring_soon"].append({
                    "cache_id": entry.cache_id,
                    "expires_in_minutes": int((entry.expire_time - current_time).total_seconds() / 60)
                })
                
        return stats
    
    def estimate_cache_cost_savings(
        self,
        token_count: int,
        cache_duration_hours: float = 1.0,
        requests_per_hour: int = 10
    ) -> Dict[str, float]:
        """Estimate cost savings from caching"""
        
        # Rough pricing estimates (check actual pricing)
        INPUT_TOKEN_COST = 0.00001  # Per token
        CACHED_TOKEN_COST = 0.0000025  # Per token (75% discount)
        STORAGE_COST_PER_HOUR = 0.000001  # Per token per hour
        
        # Without caching
        total_requests = int(cache_duration_hours * requests_per_hour)
        cost_without_cache = token_count * INPUT_TOKEN_COST * total_requests
        
        # With caching
        initial_cache_cost = token_count * INPUT_TOKEN_COST  # First request
        storage_cost = token_count * STORAGE_COST_PER_HOUR * cache_duration_hours
        subsequent_requests_cost = token_count * CACHED_TOKEN_COST * (total_requests - 1)
        cost_with_cache = initial_cache_cost + storage_cost + subsequent_requests_cost
        
        savings = cost_without_cache - cost_with_cache
        savings_percent = (savings / cost_without_cache) * 100 if cost_without_cache > 0 else 0
        
        return {
            "cost_without_cache": cost_without_cache,
            "cost_with_cache": cost_with_cache,
            "savings": savings,
            "savings_percent": savings_percent,
            "break_even_requests": 2  # Rough estimate
        }


# Global cache manager instance
context_cache_manager = None

def initialize_context_cache():
    """Initialize the global context cache manager"""
    global context_cache_manager
    
    if GENAI_AVAILABLE and settings.enable_context_caching:
        try:
            context_cache_manager = GeminiContextCache(
                default_ttl_seconds=settings.context_cache_ttl,
                min_tokens_for_cache=settings.min_tokens_for_cache
            )
            logger.info("Context cache manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize context cache: {e}")
            context_cache_manager = None
    else:
        logger.info("Context caching disabled or dependencies not available")