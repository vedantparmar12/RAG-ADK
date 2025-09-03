"""
Semantic caching system with TTL for efficient query handling.
Optimized for Windows CPU operation.
"""

import logging
import json
import hashlib
import pickle
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from collections import OrderedDict
import threading
import sqlite3

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Container for cached query results"""
    query: str
    query_embedding: Optional[np.ndarray]
    answer: str
    context: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    accessed_at: datetime
    access_count: int
    ttl_seconds: int
    similarity_threshold: float

    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if self.ttl_seconds <= 0:
            return False  # No expiration
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds

    def update_access(self):
        """Update access timestamp and count"""
        self.accessed_at = datetime.now()
        self.access_count += 1

class SemanticCache:
    """
    Semantic cache with TTL and similarity matching.
    CPU-optimized for Windows.
    """
    
    def __init__(self,
                 cache_dir: Optional[str] = None,
                 max_cache_size: int = 1000,
                 default_ttl: int = 3600,
                 similarity_threshold: float = 0.85,
                 use_embeddings: bool = True,
                 embedding_model: Optional[Any] = None,
                 persistence: bool = True):
        """
        Initialize semantic cache.
        
        Args:
            cache_dir: Directory for cache storage
            max_cache_size: Maximum number of entries
            default_ttl: Default TTL in seconds
            similarity_threshold: Minimum similarity for cache hit
            use_embeddings: Whether to use semantic similarity
            embedding_model: Model for generating embeddings
            persistence: Whether to persist cache to disk
        """
        self.max_cache_size = max_cache_size
        self.default_ttl = default_ttl
        self.similarity_threshold = similarity_threshold
        self.use_embeddings = use_embeddings
        self.embedding_model = embedding_model
        self.persistence = persistence
        
        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".rag_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.embeddings_index: Dict[str, np.ndarray] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }
        
        # Setup persistence
        if self.persistence:
            self.db_path = self.cache_dir / "semantic_cache.db"
            self._init_database()
            self._load_cache()
        
        # Start cleanup thread
        self.cleanup_interval = 300  # 5 minutes
        self._start_cleanup_thread()
        
        logger.info(f"Semantic cache initialized at {self.cache_dir}")
    
    def _init_database(self):
        """Initialize SQLite database for persistence"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                cache_key TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                context TEXT NOT NULL,
                metadata TEXT,
                created_at REAL NOT NULL,
                accessed_at REAL NOT NULL,
                access_count INTEGER DEFAULT 1,
                ttl_seconds INTEGER,
                similarity_threshold REAL,
                embedding BLOB
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache_entries(accessed_at)
        """)
        
        conn.commit()
        conn.close()
    
    def _load_cache(self):
        """Load cache from database"""
        if not self.db_path.exists():
            return
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Load entries ordered by access time
            cursor.execute("""
                SELECT * FROM cache_entries 
                ORDER BY accessed_at DESC 
                LIMIT ?
            """, (self.max_cache_size,))
            
            for row in cursor.fetchall():
                cache_key = row[0]
                
                # Deserialize embedding
                embedding = None
                if row[10]:
                    embedding = pickle.loads(row[10])
                
                entry = CacheEntry(
                    query=row[1],
                    query_embedding=embedding,
                    answer=row[2],
                    context=json.loads(row[3]),
                    metadata=json.loads(row[4]) if row[4] else {},
                    created_at=datetime.fromtimestamp(row[5]),
                    accessed_at=datetime.fromtimestamp(row[6]),
                    access_count=row[7],
                    ttl_seconds=row[8],
                    similarity_threshold=row[9]
                )
                
                # Check if expired
                if not entry.is_expired():
                    self.cache[cache_key] = entry
                    if embedding is not None:
                        self.embeddings_index[cache_key] = embedding
            
            conn.close()
            logger.info(f"Loaded {len(self.cache)} entries from cache")
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    
    def _save_entry(self, cache_key: str, entry: CacheEntry):
        """Save cache entry to database"""
        if not self.persistence:
            return
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Serialize embedding
            embedding_blob = None
            if entry.query_embedding is not None:
                embedding_blob = pickle.dumps(entry.query_embedding)
            
            cursor.execute("""
                INSERT OR REPLACE INTO cache_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_key,
                entry.query,
                entry.answer,
                json.dumps(entry.context),
                json.dumps(entry.metadata),
                entry.created_at.timestamp(),
                entry.accessed_at.timestamp(),
                entry.access_count,
                entry.ttl_seconds,
                entry.similarity_threshold,
                embedding_blob
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save cache entry: {e}")
    
    def get(self, query: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached result for query.
        
        Args:
            query: Query to look up
            context: Optional context for matching
            
        Returns:
            Cached result or None
        """
        with self.lock:
            # Generate cache key
            cache_key = self._generate_cache_key(query, context)
            
            # Check exact match first
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                
                # Check expiration
                if entry.is_expired():
                    self._remove_entry(cache_key)
                    self.stats["expirations"] += 1
                    self.stats["misses"] += 1
                    return None
                
                # Update access info
                entry.update_access()
                self._save_entry(cache_key, entry)
                
                # Move to end (most recently used)
                self.cache.move_to_end(cache_key)
                
                self.stats["hits"] += 1
                return {
                    "answer": entry.answer,
                    "context": entry.context,
                    "metadata": {
                        **entry.metadata,
                        "cache_hit": True,
                        "cache_type": "exact",
                        "access_count": entry.access_count
                    }
                }
            
            # Try semantic similarity if enabled
            if self.use_embeddings and self.embedding_model:
                result = self._semantic_search(query, context)
                if result:
                    self.stats["hits"] += 1
                    return result
            
            self.stats["misses"] += 1
            return None
    
    def put(self, 
            query: str,
            answer: str,
            context: List[str],
            metadata: Optional[Dict[str, Any]] = None,
            ttl: Optional[int] = None) -> str:
        """
        Store result in cache.
        
        Args:
            query: Query that was answered
            answer: Generated answer
            context: Context used
            metadata: Optional metadata
            ttl: Optional TTL override
            
        Returns:
            Cache key
        """
        with self.lock:
            # Generate cache key
            cache_key = self._generate_cache_key(query, metadata)
            
            # Generate embedding if enabled
            embedding = None
            if self.use_embeddings and self.embedding_model:
                try:
                    embedding = self._generate_embedding(query)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding: {e}")
            
            # Create cache entry
            entry = CacheEntry(
                query=query,
                query_embedding=embedding,
                answer=answer,
                context=context,
                metadata=metadata or {},
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                access_count=1,
                ttl_seconds=ttl or self.default_ttl,
                similarity_threshold=self.similarity_threshold
            )
            
            # Check cache size and evict if needed
            if len(self.cache) >= self.max_cache_size:
                self._evict_lru()
            
            # Store entry
            self.cache[cache_key] = entry
            if embedding is not None:
                self.embeddings_index[cache_key] = embedding
            
            # Persist to database
            self._save_entry(cache_key, entry)
            
            return cache_key
    
    def _semantic_search(self, query: str, context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Search cache using semantic similarity.
        
        Args:
            query: Query to search for
            context: Optional context
            
        Returns:
            Best matching cached result or None
        """
        if not self.embeddings_index:
            return None
        
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            if query_embedding is None:
                return None
            
            # Calculate similarities
            similarities = []
            for cache_key, cached_embedding in self.embeddings_index.items():
                if cache_key not in self.cache:
                    continue
                
                entry = self.cache[cache_key]
                if entry.is_expired():
                    continue
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, cached_embedding)
                
                if similarity >= entry.similarity_threshold:
                    similarities.append((cache_key, similarity))
            
            if not similarities:
                return None
            
            # Get best match
            best_key, best_similarity = max(similarities, key=lambda x: x[1])
            entry = self.cache[best_key]
            
            # Update access info
            entry.update_access()
            self._save_entry(best_key, entry)
            self.cache.move_to_end(best_key)
            
            return {
                "answer": entry.answer,
                "context": entry.context,
                "metadata": {
                    **entry.metadata,
                    "cache_hit": True,
                    "cache_type": "semantic",
                    "similarity": best_similarity,
                    "original_query": entry.query,
                    "access_count": entry.access_count
                }
            }
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return None
    
    def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None
        """
        if not self.embedding_model:
            return None
        
        try:
            # Use the provided embedding model
            # This is a placeholder - integrate with your actual embedding model
            # For CPU optimization, use a small model
            
            # Dummy embedding for demonstration
            # Replace with: embedding = self.embedding_model.encode(text)
            embedding = np.random.randn(384).astype(np.float32)
            
            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0.0-1.0)
        """
        # Vectors should already be normalized
        return float(np.dot(vec1, vec2))
    
    def _generate_cache_key(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate unique cache key.
        
        Args:
            query: Query text
            context: Optional context
            
        Returns:
            Cache key
        """
        # Include relevant context in key
        key_parts = [query]
        
        if context:
            # Add relevant context fields
            if "corpus_id" in context:
                key_parts.append(f"corpus:{context['corpus_id']}")
            if "filters" in context:
                key_parts.append(f"filters:{json.dumps(context['filters'], sort_keys=True)}")
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.cache:
            return
        
        # Get oldest entry (first in OrderedDict)
        oldest_key = next(iter(self.cache))
        self._remove_entry(oldest_key)
        self.stats["evictions"] += 1
    
    def _remove_entry(self, cache_key: str):
        """Remove entry from cache"""
        if cache_key in self.cache:
            del self.cache[cache_key]
        
        if cache_key in self.embeddings_index:
            del self.embeddings_index[cache_key]
        
        # Remove from database
        if self.persistence:
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache_entries WHERE cache_key = ?", (cache_key,))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Failed to remove cache entry from database: {e}")
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_worker():
            while True:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired()
        
        thread = threading.Thread(target=cleanup_worker, daemon=True)
        thread.start()
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        with self.lock:
            expired_keys = []
            
            for cache_key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(cache_key)
            
            for key in expired_keys:
                self._remove_entry(key)
                self.stats["expirations"] += 1
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def clear(self):
        """Clear entire cache"""
        with self.lock:
            self.cache.clear()
            self.embeddings_index.clear()
            
            if self.persistence:
                try:
                    conn = sqlite3.connect(str(self.db_path))
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM cache_entries")
                    conn.commit()
                    conn.close()
                except Exception as e:
                    logger.error(f"Failed to clear database: {e}")
            
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                **self.stats,
                "size": len(self.cache),
                "hit_rate": hit_rate,
                "total_requests": total_requests
            }