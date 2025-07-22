"""
Mock components for testing without heavy ML dependencies.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MockSentenceTransformer:
    """Mock sentence transformer for testing"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        logger.info(f"MockSentenceTransformer initialized with {model_name}")
        
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Return random embeddings for testing"""
        # Return random 384-dimensional embeddings
        return np.random.randn(len(texts), 384)

class MockFAISS:
    """Mock FAISS index for testing"""
    
    @staticmethod
    def IndexFlatIP(dimension: int):
        """Create mock index"""
        
        class MockIndex:
            def __init__(self, dim):
                self.dimension = dim
                self.vectors = []
                
            def add(self, vectors):
                self.vectors.extend(vectors)
                
            def search(self, query, k):
                # Return mock results
                n_results = min(k, len(self.vectors))
                indices = np.arange(n_results).reshape(1, -1)
                scores = np.random.rand(1, n_results)
                return scores, indices
                
        return MockIndex(dimension)
    
    @staticmethod
    def IndexIVFFlat(quantizer, dimension, nlist):
        """Create mock IVF index"""
        return MockFAISS.IndexFlatIP(dimension)

# Replace imports if actual libraries are not available
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.warning("sentence-transformers not available, using mock")
    SentenceTransformer = MockSentenceTransformer

try:
    import faiss
except ImportError:
    logger.warning("faiss not available, using mock")
    faiss = MockFAISS