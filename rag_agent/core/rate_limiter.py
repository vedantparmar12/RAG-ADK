"""
Rate limiting and request management for API calls.
"""

import time
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import logging
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class RateLimitTier(Enum):
    """API usage tiers with different rate limits"""
    FREE = "free"
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


@dataclass
class RateLimitConfig:
    """Configuration for rate limits per model and tier"""
    rpm: int  # Requests per minute
    tpm: int  # Tokens per minute
    rpd: int  # Requests per day
    
    # Optional limits
    ipm: Optional[int] = None  # Images per minute
    sessions: Optional[int] = None  # Concurrent sessions (for Live API)


# Gemini API rate limits by model and tier
RATE_LIMITS = {
    "gemini-embedding-001": {
        RateLimitTier.FREE: RateLimitConfig(rpm=100, tpm=30000, rpd=1000),
        RateLimitTier.TIER_1: RateLimitConfig(rpm=1000, tpm=1000000, rpd=50000),
        RateLimitTier.TIER_2: RateLimitConfig(rpm=2000, tpm=2000000, rpd=100000),
        RateLimitTier.TIER_3: RateLimitConfig(rpm=4000, tpm=4000000, rpd=200000)
    },
    "gemini-2.5-flash": {
        RateLimitTier.FREE: RateLimitConfig(rpm=10, tpm=250000, rpd=250),
        RateLimitTier.TIER_1: RateLimitConfig(rpm=60, tpm=2000000, rpd=10000),
        RateLimitTier.TIER_2: RateLimitConfig(rpm=120, tpm=4000000, rpd=25000),
        RateLimitTier.TIER_3: RateLimitConfig(rpm=240, tpm=10000000, rpd=50000)
    },
    "gemini-2.0-flash": {
        RateLimitTier.FREE: RateLimitConfig(rpm=15, tpm=1000000, rpd=200),
        RateLimitTier.TIER_1: RateLimitConfig(rpm=100, tpm=5000000, rpd=10000),
        RateLimitTier.TIER_2: RateLimitConfig(rpm=200, tpm=10000000, rpd=25000),
        RateLimitTier.TIER_3: RateLimitConfig(rpm=400, tpm=20000000, rpd=50000)
    }
}


@dataclass
class RequestMetrics:
    """Track request metrics for rate limiting"""
    requests_per_minute: deque = field(default_factory=lambda: deque(maxlen=60))
    tokens_per_minute: deque = field(default_factory=lambda: deque(maxlen=60))
    daily_requests: int = 0
    daily_tokens: int = 0
    last_reset: datetime = field(default_factory=datetime.now)
    
    def add_request(self, tokens: int = 0):
        """Record a new request"""
        current_time = time.time()
        self.requests_per_minute.append(current_time)
        if tokens > 0:
            self.tokens_per_minute.append((current_time, tokens))
        self.daily_requests += 1
        self.daily_tokens += tokens
        
    def get_current_rpm(self) -> int:
        """Get requests in the last minute"""
        current_time = time.time()
        cutoff = current_time - 60
        return sum(1 for t in self.requests_per_minute if t > cutoff)
    
    def get_current_tpm(self) -> int:
        """Get tokens in the last minute"""
        current_time = time.time()
        cutoff = current_time - 60
        return sum(tokens for t, tokens in self.tokens_per_minute if t > cutoff)
    
    def reset_daily_counters(self):
        """Reset daily counters (called at midnight Pacific)"""
        self.daily_requests = 0
        self.daily_tokens = 0
        self.last_reset = datetime.now()


class RateLimiter:
    """Manages rate limiting for API calls"""
    
    def __init__(
        self,
        model: str,
        tier: RateLimitTier = RateLimitTier.FREE,
        retry_attempts: int = 3,
        base_delay: float = 1.0
    ):
        self.model = model
        self.tier = tier
        self.retry_attempts = retry_attempts
        self.base_delay = base_delay
        
        # Get rate limits for model
        self.limits = RATE_LIMITS.get(model, {}).get(tier)
        if not self.limits:
            # Default conservative limits
            self.limits = RateLimitConfig(rpm=10, tpm=10000, rpd=100)
            
        self.metrics = RequestMetrics()
        self._lock = threading.Lock()
        
        # Start background task for daily reset
        self._start_daily_reset_timer()
        
    def _start_daily_reset_timer(self):
        """Schedule daily reset at midnight Pacific"""
        def reset_loop():
            while True:
                # Calculate seconds until midnight Pacific
                now = datetime.now()
                midnight = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                sleep_seconds = (midnight - now).total_seconds()
                
                time.sleep(sleep_seconds)
                with self._lock:
                    self.metrics.reset_daily_counters()
                logger.info(f"Reset daily counters for {self.model}")
                
        reset_thread = threading.Thread(target=reset_loop, daemon=True)
        reset_thread.start()
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)"""
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4
    
    def can_make_request(self, estimated_tokens: int = 0) -> bool:
        """Check if request can be made within rate limits"""
        with self._lock:
            current_rpm = self.metrics.get_current_rpm()
            current_tpm = self.metrics.get_current_tpm()
            
            # Check all limits
            if current_rpm >= self.limits.rpm:
                return False
            if estimated_tokens > 0 and current_tpm + estimated_tokens > self.limits.tpm:
                return False
            if self.metrics.daily_requests >= self.limits.rpd:
                return False
                
            return True
    
    def wait_if_needed(self, estimated_tokens: int = 0) -> float:
        """Calculate wait time if rate limited"""
        if self.can_make_request(estimated_tokens):
            return 0.0
            
        with self._lock:
            current_rpm = self.metrics.get_current_rpm()
            current_tpm = self.metrics.get_current_tpm()
            
            wait_times = []
            
            # Calculate wait for RPM limit
            if current_rpm >= self.limits.rpm:
                # Find oldest request in the minute window
                if self.metrics.requests_per_minute:
                    oldest = min(self.metrics.requests_per_minute)
                    wait_time = 60 - (time.time() - oldest) + 0.1
                    wait_times.append(wait_time)
                    
            # Calculate wait for TPM limit
            if estimated_tokens > 0 and current_tpm + estimated_tokens > self.limits.tpm:
                # Need to wait for some tokens to expire
                tokens_to_free = (current_tpm + estimated_tokens) - self.limits.tpm
                # Find when enough tokens will be freed
                current_time = time.time()
                cutoff = current_time - 60
                accumulated = 0
                
                for t, tokens in sorted(self.metrics.tokens_per_minute, key=lambda x: x[0]):
                    if t > cutoff:
                        accumulated += tokens
                        if accumulated >= tokens_to_free:
                            wait_time = 60 - (current_time - t) + 0.1
                            wait_times.append(wait_time)
                            break
                            
            # Daily limit - need to wait until reset
            if self.metrics.daily_requests >= self.limits.rpd:
                now = datetime.now()
                midnight = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                wait_seconds = (midnight - now).total_seconds()
                wait_times.append(wait_seconds)
                
            return max(wait_times) if wait_times else 1.0
    
    def record_request(self, tokens: int = 0):
        """Record that a request was made"""
        with self._lock:
            self.metrics.add_request(tokens)
            
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        estimated_tokens: int = 0,
        **kwargs
    ) -> Any:
        """Execute function with rate limiting and retry logic"""
        
        last_error = None
        
        for attempt in range(self.retry_attempts):
            # Wait if rate limited
            wait_time = self.wait_if_needed(estimated_tokens)
            if wait_time > 0:
                logger.info(f"Rate limited, waiting {wait_time:.1f}s (attempt {attempt + 1})")
                await asyncio.sleep(wait_time)
                
            try:
                # Record request
                self.record_request(estimated_tokens)
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                    
                return result
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if it's a rate limit error
                if any(term in error_str for term in ['rate limit', 'quota', 'too many requests', '429']):
                    # Exponential backoff
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"Rate limit error, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    # Non-rate limit error, re-raise
                    raise
                    
        # All retries exhausted
        raise last_error or Exception("All retry attempts failed")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current rate limit metrics"""
        with self._lock:
            return {
                "model": self.model,
                "tier": self.tier.value,
                "current_rpm": self.metrics.get_current_rpm(),
                "current_tpm": self.metrics.get_current_tpm(),
                "daily_requests": self.metrics.daily_requests,
                "daily_tokens": self.metrics.daily_tokens,
                "limits": {
                    "rpm": self.limits.rpm,
                    "tpm": self.limits.tpm,
                    "rpd": self.limits.rpd
                },
                "utilization": {
                    "rpm_percent": (self.metrics.get_current_rpm() / self.limits.rpm) * 100,
                    "tpm_percent": (self.metrics.get_current_tpm() / self.limits.tpm) * 100,
                    "rpd_percent": (self.metrics.daily_requests / self.limits.rpd) * 100
                }
            }


class BatchProcessor:
    """Process requests in batches to optimize rate limits"""
    
    def __init__(self, rate_limiter: RateLimiter, batch_size: int = 10):
        self.rate_limiter = rate_limiter
        self.batch_size = batch_size
        self.queue = asyncio.Queue()
        self.results = {}
        
    async def add_request(self, request_id: str, func: Callable, *args, **kwargs):
        """Add request to batch queue"""
        await self.queue.put((request_id, func, args, kwargs))
        
    async def process_batch(self):
        """Process a batch of requests"""
        batch = []
        
        # Collect batch
        for _ in range(self.batch_size):
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                batch.append(item)
            except asyncio.TimeoutError:
                break
                
        if not batch:
            return
            
        # Estimate total tokens for batch
        total_tokens = sum(
            self.rate_limiter.estimate_tokens(str(args) + str(kwargs))
            for _, _, args, kwargs in batch
        )
        
        # Execute batch with rate limiting
        results = await self.rate_limiter.execute_with_retry(
            self._execute_batch,
            batch,
            estimated_tokens=total_tokens
        )
        
        # Store results
        for (request_id, _, _, _), result in zip(batch, results):
            self.results[request_id] = result
            
    async def _execute_batch(self, batch: List[tuple]) -> List[Any]:
        """Execute all functions in batch"""
        tasks = []
        
        for _, func, args, kwargs in batch:
            if asyncio.iscoroutinefunction(func):
                tasks.append(func(*args, **kwargs))
            else:
                # Wrap sync function in coroutine
                tasks.append(asyncio.to_thread(func, *args, **kwargs))
                
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_result(self, request_id: str) -> Any:
        """Get result for request ID"""
        return self.results.get(request_id)