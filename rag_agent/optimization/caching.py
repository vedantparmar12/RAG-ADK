"""
Multi-level caching system for RAG with Redis and in-memory LRU cache.
"""

import redis
import hashlib
import json
import pickle
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import logging
import threading

from ..config import settings

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = []
        self.lock = threading.Lock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        
        with self.lock:
            if key not in self.cache:
                return None
                
            # Move to end (most recently used)
            self.order.remove(key)
            self.order.append(key)
            
            return self.cache[key]
    
    def put(self, key: str, value: Any):
        """Put value in cache"""
        
        with self.lock:
            if key in self.cache:
                # Update existing
                self.order.remove(key)
            elif len(self.cache) >= self.capacity:
                # Evict least recently used
                lru_key = self.order.pop(0)
                del self.cache[lru_key]
                
            self.cache[key] = value
            self.order.append(key)


class RetrievalCache:
    """Multi-level caching system for RAG"""
    
    def __init__(self, settings_obj: Optional[Any] = None):
        self.settings = settings_obj or settings
        self.setup_cache()
        
    def setup_cache(self):
        """Initialize cache backends"""
        # Redis for distributed caching
        if self.settings.enable_caching:
            try:
                self.redis_client = redis.Redis(
                    host=self.settings.redis_host,
                    port=self.settings.redis_port,
                    decode_responses=False,
                    socket_connect_timeout=5
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis cache connected successfully")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using in-memory cache only.")
                self.redis_client = None
        else:
            self.redis_client = None
        
        # In-memory LRU cache
        self.memory_cache = LRUCache(capacity=1000)
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
        
    def generate_cache_key(
        self,
        query: str,
        corpus_id: str,
        params: Dict[str, Any] = None
    ) -> str:
        """Generate cache key for query"""
        
        key_parts = [
            query,
            corpus_id,
            json.dumps(params or {}, sort_keys=True)
        ]
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(
        self,
        query: str,
        corpus_id: str,
        params: Dict[str, Any] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached results"""
        
        cache_key = self.generate_cache_key(query, corpus_id, params)
        
        # Check memory cache first
        result = self.memory_cache.get(cache_key)
        if result is not None:
            self.stats["hits"] += 1
            logger.debug(f"Memory cache hit for key: {cache_key}")
            return result
            
        # Check Redis if available
        if self.redis_client:
            try:
                redis_result = self.redis_client.get(cache_key)
                if redis_result:
                    result = pickle.loads(redis_result)
                    # Update memory cache
                    self.memory_cache.put(cache_key, result)
                    self.stats["hits"] += 1
                    logger.debug(f"Redis cache hit for key: {cache_key}")
                    return result
            except Exception as e:
                logger.error(f"Redis get error: {e}")
            
        self.stats["misses"] += 1
        return None
    
    def set(
        self,
        query: str,
        corpus_id: str,
        results: List[Dict[str, Any]],
        params: Dict[str, Any] = None,
        ttl: Optional[int] = None
    ):
        """Cache results"""
        
        if ttl is None:
            ttl = self.settings.cache_ttl
            
        cache_key = self.generate_cache_key(query, corpus_id, params)
        
        # Add to memory cache
        self.memory_cache.put(cache_key, results)
        
        # Add to Redis with TTL if available
        if self.redis_client:
            try:
                self.redis_client.setex(
                    cache_key,
                    ttl,
                    pickle.dumps(results)
                )
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        
        logger.debug(f"Cached results for key: {cache_key}")
    
    def invalidate_corpus(self, corpus_id: str):
        """Invalidate all cache entries for a corpus"""
        
        if self.redis_client:
            # Scan Redis for matching keys
            pattern = f"*|{corpus_id}|*"
            cursor = 0
            
            try:
                while True:
                    cursor, keys = self.redis_client.scan(
                        cursor=cursor,
                        match=pattern,
                        count=100
                    )
                    
                    if keys:
                        self.redis_client.delete(*keys)
                        logger.info(f"Invalidated {len(keys)} cache entries")
                        
                    if cursor == 0:
                        break
            except Exception as e:
                logger.error(f"Redis invalidation error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate,
            "memory_cache_size": len(self.memory_cache.cache),
            "redis_available": self.redis_client is not None
        }