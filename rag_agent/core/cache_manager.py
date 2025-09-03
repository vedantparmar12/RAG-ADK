"""
Caching system for RAG query embeddings and search results.
Supports both in-memory and Redis backends with TTL.
"""

import json
import hashlib
import time
from typing import Dict, Any, Optional, List, Union
import logging
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)

# Optional Redis support
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not installed. Using in-memory cache only.")


class CacheBackend(ABC):
    """Abstract cache backend interface."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        pass


class MemoryCache(CacheBackend):
    """In-memory cache with TTL support."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        return time.time() > entry.get('expires_at', 0)
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        expired_keys = [
            key for key, entry in self.cache.items() 
            if self._is_expired(entry)
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def _evict_if_needed(self):
        """Evict oldest entries if cache is full."""
        if len(self.cache) >= self.max_size:
            # Remove oldest 20% of entries
            sorted_items = sorted(
                self.cache.items(), 
                key=lambda x: x[1].get('created_at', 0)
            )
            to_remove = int(len(sorted_items) * 0.2)
            for key, _ in sorted_items[:to_remove]:
                del self.cache[key]
    
    def get(self, key: str) -> Optional[Any]:
        self._cleanup_expired()
        entry = self.cache.get(key)
        if entry and not self._is_expired(entry):
            entry['access_count'] = entry.get('access_count', 0) + 1
            entry['last_accessed'] = time.time()
            return entry['value']
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        self._cleanup_expired()
        self._evict_if_needed()
        
        self.cache[key] = {
            'value': value,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'expires_at': time.time() + ttl,
            'access_count': 0
        }
        return True
    
    def delete(self, key: str) -> bool:
        return self.cache.pop(key, None) is not None
    
    def clear(self) -> bool:
        self.cache.clear()
        return True
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        self._cleanup_expired()
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': sum(e.get('access_count', 0) for e in self.cache.values()) / max(len(self.cache), 1)
        }


class RedisCache(CacheBackend):
    """Redis-based cache backend."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", prefix: str = "rag:"):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available")
        
        self.client = redis.from_url(redis_url)
        self.prefix = prefix
        
        try:
            self.client.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _make_key(self, key: str) -> str:
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        try:
            data = self.client.get(self._make_key(key))
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Redis get failed: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        try:
            data = json.dumps(value, default=self._json_serializer)
            self.client.setex(self._make_key(key), ttl, data)
            return True
        except Exception as e:
            logger.error(f"Redis set failed: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        try:
            return bool(self.client.delete(self._make_key(key)))
        except Exception as e:
            logger.error(f"Redis delete failed: {e}")
            return False
    
    def clear(self) -> bool:
        try:
            keys = self.client.keys(f"{self.prefix}*")
            if keys:
                self.client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Redis clear failed: {e}")
            return False
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy arrays."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object {obj} is not JSON serializable")


class RAGCacheManager:
    """Main cache manager for RAG operations."""
    
    def __init__(self, backend: CacheBackend, enable_embedding_cache: bool = True, 
                 enable_result_cache: bool = True):
        self.backend = backend
        self.enable_embedding_cache = enable_embedding_cache
        self.enable_result_cache = enable_result_cache
    
    def _make_embedding_key(self, text: str, model: str) -> str:
        """Generate cache key for embeddings."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"emb:{model}:{text_hash}"
    
    def _make_search_key(self, corpus_name: str, query: str, search_params: Dict) -> str:
        """Generate cache key for search results."""
        params_str = json.dumps(search_params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"search:{corpus_name}:{query_hash}:{params_hash}"
    
    def get_embedding(self, text: str, model: str) -> Optional[np.ndarray]:
        """Get cached embedding."""
        if not self.enable_embedding_cache:
            return None
        
        key = self._make_embedding_key(text, model)
        cached = self.backend.get(key)
        if cached:
            return np.array(cached)
        return None
    
    def set_embedding(self, text: str, model: str, embedding: np.ndarray, ttl: int = 86400) -> bool:
        """Cache embedding with 24h TTL."""
        if not self.enable_embedding_cache:
            return False
        
        key = self._make_embedding_key(text, model)
        return self.backend.set(key, embedding.tolist(), ttl)
    
    def get_search_results(self, corpus_name: str, query: str, search_params: Dict) -> Optional[List[Dict]]:
        """Get cached search results."""
        if not self.enable_result_cache:
            return None
        
        key = self._make_search_key(corpus_name, query, search_params)
        return self.backend.get(key)
    
    def set_search_results(self, corpus_name: str, query: str, search_params: Dict, 
                          results: List[Dict], ttl: int = 3600) -> bool:
        """Cache search results with 1h TTL."""
        if not self.enable_result_cache:
            return False
        
        key = self._make_search_key(corpus_name, query, search_params)
        return self.backend.set(key, results, ttl)
    
    def invalidate_corpus(self, corpus_name: str) -> bool:
        """Invalidate all cache entries for a corpus."""
        # For memory cache, we'd need to track keys by corpus
        # For Redis, we can use pattern matching
        if isinstance(self.backend, RedisCache):
            try:
                pattern = f"{self.backend.prefix}search:{corpus_name}:*"
                keys = self.backend.client.keys(pattern)
                if keys:
                    self.backend.client.delete(*keys)
                return True
            except Exception as e:
                logger.error(f"Failed to invalidate corpus cache: {e}")
        
        # For memory cache, we'd need a more sophisticated approach
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if hasattr(self.backend, 'stats'):
            return self.backend.stats()
        return {}


def create_cache_manager(use_redis: bool = False, redis_url: str = "redis://localhost:6379/0") -> RAGCacheManager:
    """Factory function to create cache manager."""
    try:
        if use_redis and REDIS_AVAILABLE:
            backend = RedisCache(redis_url)
            logger.info("Using Redis cache backend")
        else:
            backend = MemoryCache(max_size=2000)
            logger.info("Using in-memory cache backend")
        
        return RAGCacheManager(backend)
    except Exception as e:
        logger.warning(f"Failed to create cache backend: {e}. Using memory cache.")
        return RAGCacheManager(MemoryCache())


# Alias for backward compatibility
CacheManager = RAGCacheManager
