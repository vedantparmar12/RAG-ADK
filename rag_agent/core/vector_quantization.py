"""
Vector Quantization Module for Embedding Compression
Reduces memory usage and improves search speed while maintaining quality
"""

import logging
import pickle
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import os

import numpy as np
import faiss
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class QuantizationType(Enum):
    """Types of vector quantization"""
    PRODUCT_QUANTIZATION = "pq"
    SCALAR_QUANTIZATION = "sq"
    BINARY_QUANTIZATION = "binary"
    HIERARCHICAL_QUANTIZATION = "hierarchical"
    LEARNED_QUANTIZATION = "learned"

@dataclass
class QuantizationConfig:
    """Configuration for vector quantization"""
    method: QuantizationType
    n_bits: int = 8
    n_subvectors: Optional[int] = None
    codebook_size: Optional[int] = None
    compression_ratio: float = 8.0
    training_iterations: int = 100
    batch_size: int = 1000

@dataclass
class QuantizedIndex:
    """Quantized vector index"""
    quantizer: Any
    codebook: Optional[np.ndarray]
    compressed_vectors: np.ndarray
    metadata: Dict[str, Any]
    config: QuantizationConfig

class ProductQuantizer:
    """Product Quantization implementation"""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.n_subvectors = config.n_subvectors or 8
        self.n_bits = config.n_bits
        self.codebook_size = config.codebook_size or (2 ** self.n_bits)
        
        self.codebooks = []
        self.trained = False
        
    def train(self, vectors: np.ndarray) -> None:
        """Train product quantizer on vector data"""
        
        logger.info(f"Training product quantizer with {len(vectors)} vectors")
        
        n_vectors, dim = vectors.shape
        
        if dim % self.n_subvectors != 0:
            # Pad vectors to make divisible
            padding_size = self.n_subvectors - (dim % self.n_subvectors)
            vectors = np.pad(vectors, ((0, 0), (0, padding_size)), mode='constant')
            dim = vectors.shape[1]
        
        subvector_dim = dim // self.n_subvectors
        self.subvector_dim = subvector_dim
        
        # Train codebook for each subvector
        self.codebooks = []
        
        for i in range(self.n_subvectors):
            start_idx = i * subvector_dim
            end_idx = (i + 1) * subvector_dim
            subvectors = vectors[:, start_idx:end_idx]
            
            # Use MiniBatchKMeans for large datasets
            if n_vectors > 100000:
                kmeans = MiniBatchKMeans(
                    n_clusters=self.codebook_size,
                    batch_size=self.config.batch_size,
                    max_iter=self.config.training_iterations,
                    random_state=42,
                    n_init=3
                )
            else:
                kmeans = KMeans(
                    n_clusters=self.codebook_size,
                    max_iter=self.config.training_iterations,
                    random_state=42,
                    n_init=10
                )
            
            kmeans.fit(subvectors)
            self.codebooks.append(kmeans.cluster_centers_)
            
            logger.info(f"Trained codebook {i+1}/{self.n_subvectors}")
        
        self.trained = True
        logger.info("Product quantization training completed")
    
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors using trained quantizer"""
        
        if not self.trained:
            raise ValueError("Quantizer must be trained before encoding")
        
        n_vectors, dim = vectors.shape
        
        # Pad if necessary
        if dim != self.subvector_dim * self.n_subvectors:
            padding_size = (self.subvector_dim * self.n_subvectors) - dim
            if padding_size > 0:
                vectors = np.pad(vectors, ((0, 0), (0, padding_size)), mode='constant')
        
        # Encode each subvector
        codes = np.zeros((n_vectors, self.n_subvectors), dtype=np.uint8)
        
        for i in range(self.n_subvectors):
            start_idx = i * self.subvector_dim
            end_idx = (i + 1) * self.subvector_dim
            subvectors = vectors[:, start_idx:end_idx]
            
            # Find nearest codebook entry
            distances = np.linalg.norm(
                subvectors[:, np.newaxis, :] - self.codebooks[i][np.newaxis, :, :],
                axis=2
            )
            codes[:, i] = np.argmin(distances, axis=1)
        
        return codes
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode quantized vectors"""
        
        if not self.trained:
            raise ValueError("Quantizer must be trained before decoding")
        
        n_vectors, n_subvectors = codes.shape
        reconstructed = np.zeros((n_vectors, self.subvector_dim * n_subvectors))
        
        for i in range(self.n_subvectors):
            start_idx = i * self.subvector_dim
            end_idx = (i + 1) * self.subvector_dim
            reconstructed[:, start_idx:end_idx] = self.codebooks[i][codes[:, i]]
        
        return reconstructed
    
    def get_compression_ratio(self) -> float:
        """Calculate compression ratio"""
        
        original_size = self.subvector_dim * self.n_subvectors * 32  # float32
        compressed_size = self.n_subvectors * self.n_bits
        
        return original_size / compressed_size

class ScalarQuantizer:
    """Scalar Quantization implementation"""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.n_bits = config.n_bits
        self.n_levels = 2 ** self.n_bits
        
        self.scale = None
        self.offset = None
        self.trained = False
    
    def train(self, vectors: np.ndarray) -> None:
        """Train scalar quantizer"""
        
        logger.info(f"Training scalar quantizer with {len(vectors)} vectors")
        
        # Calculate min/max for each dimension
        self.min_vals = np.min(vectors, axis=0)
        self.max_vals = np.max(vectors, axis=0)
        
        # Calculate scale and offset
        self.scale = (self.max_vals - self.min_vals) / (self.n_levels - 1)
        self.offset = self.min_vals
        
        # Avoid division by zero
        self.scale = np.maximum(self.scale, 1e-8)
        
        self.trained = True
        logger.info("Scalar quantization training completed")
    
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors using scalar quantization"""
        
        if not self.trained:
            raise ValueError("Quantizer must be trained before encoding")
        
        # Normalize to [0, n_levels-1]
        normalized = (vectors - self.offset) / self.scale
        
        # Quantize
        quantized = np.round(np.clip(normalized, 0, self.n_levels - 1))
        
        return quantized.astype(np.uint8 if self.n_bits <= 8 else np.uint16)
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode quantized vectors"""
        
        if not self.trained:
            raise ValueError("Quantizer must be trained before decoding")
        
        # Dequantize
        return codes.astype(np.float32) * self.scale + self.offset

class BinaryQuantizer:
    """Binary Quantization using LSH"""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.n_bits = config.n_bits or 64
        
        self.random_projections = None
        self.trained = False
    
    def train(self, vectors: np.ndarray) -> None:
        """Train binary quantizer"""
        
        logger.info(f"Training binary quantizer with {len(vectors)} vectors")
        
        n_vectors, dim = vectors.shape
        
        # Generate random projection matrix
        self.random_projections = np.random.randn(dim, self.n_bits)
        
        # Normalize projections
        self.random_projections /= np.linalg.norm(self.random_projections, axis=0)
        
        self.trained = True
        logger.info("Binary quantization training completed")
    
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors as binary codes"""
        
        if not self.trained:
            raise ValueError("Quantizer must be trained before encoding")
        
        # Project vectors
        projections = np.dot(vectors, self.random_projections)
        
        # Convert to binary (0/1)
        binary_codes = (projections > 0).astype(np.uint8)
        
        return binary_codes
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Approximate decode (reconstruction is lossy)"""
        
        if not self.trained:
            raise ValueError("Quantizer must be trained before decoding")
        
        # Convert binary to -1/1
        signed_codes = codes.astype(np.float32) * 2 - 1
        
        # Approximate reconstruction
        reconstructed = np.dot(signed_codes, self.random_projections.T)
        
        return reconstructed

class LearnedQuantizer(nn.Module):
    """Neural network-based learned quantization"""
    
    def __init__(self, config: QuantizationConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.n_bits = config.n_bits
        self.codebook_size = config.codebook_size or (2 ** self.n_bits)
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, self.codebook_size)
        )
        
        # Learnable codebook
        self.codebook = nn.Parameter(torch.randn(self.codebook_size, input_dim))
        
        self.trained = False
    
    def forward(self, x):
        """Forward pass through quantizer"""
        
        # Get logits for codebook selection
        logits = self.encoder(x)
        
        # Soft assignment (differentiable)
        soft_assignment = torch.softmax(logits, dim=-1)
        
        # Quantized representation
        quantized = torch.matmul(soft_assignment, self.codebook)
        
        return quantized, logits, soft_assignment
    
    def train_quantizer(self, vectors: np.ndarray, epochs: int = 100):
        """Train the learned quantizer"""
        
        logger.info(f"Training learned quantizer with {len(vectors)} vectors")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        # Convert to tensor
        vectors_tensor = torch.FloatTensor(vectors).to(device)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        # Training loop
        batch_size = self.config.batch_size
        n_batches = len(vectors) // batch_size
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            # Shuffle data
            perm = torch.randperm(len(vectors))
            vectors_tensor = vectors_tensor[perm]
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                batch = vectors_tensor[start_idx:end_idx]
                
                optimizer.zero_grad()
                
                # Forward pass
                quantized, logits, soft_assignment = self.forward(batch)
                
                # Reconstruction loss
                recon_loss = torch.mse_loss(quantized, batch)
                
                # Commitment loss (encourage discrete assignments)
                commitment_loss = torch.mean((soft_assignment.max(dim=1)[0] - 1.0) ** 2)
                
                loss = recon_loss + 0.1 * commitment_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / n_batches
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.trained = True
        logger.info("Learned quantization training completed")
    
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors using learned quantizer"""
        
        if not self.trained:
            raise ValueError("Quantizer must be trained before encoding")
        
        device = next(self.parameters()).device
        vectors_tensor = torch.FloatTensor(vectors).to(device)
        
        with torch.no_grad():
            _, logits, _ = self.forward(vectors_tensor)
            codes = torch.argmax(logits, dim=-1)
        
        return codes.cpu().numpy()
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode using learned quantizer"""
        
        if not self.trained:
            raise ValueError("Quantizer must be trained before decoding")
        
        device = next(self.parameters()).device
        codes_tensor = torch.LongTensor(codes).to(device)
        
        with torch.no_grad():
            reconstructed = self.codebook[codes_tensor]
        
        return reconstructed.cpu().numpy()

class VectorQuantizationManager:
    """Main manager for vector quantization"""
    
    def __init__(self):
        self.quantizers: Dict[str, Any] = {}
        self.indices: Dict[str, QuantizedIndex] = {}
    
    def create_quantizer(
        self,
        name: str,
        config: QuantizationConfig,
        vectors: Optional[np.ndarray] = None
    ) -> None:
        """Create and optionally train a quantizer"""
        
        if config.method == QuantizationType.PRODUCT_QUANTIZATION:
            quantizer = ProductQuantizer(config)
        elif config.method == QuantizationType.SCALAR_QUANTIZATION:
            quantizer = ScalarQuantizer(config)
        elif config.method == QuantizationType.BINARY_QUANTIZATION:
            quantizer = BinaryQuantizer(config)
        elif config.method == QuantizationType.LEARNED_QUANTIZATION:
            if vectors is None:
                raise ValueError("Learned quantization requires training vectors")
            quantizer = LearnedQuantizer(config, vectors.shape[1])
        else:
            raise ValueError(f"Unsupported quantization method: {config.method}")
        
        # Train if vectors provided
        if vectors is not None:
            if config.method == QuantizationType.LEARNED_QUANTIZATION:
                quantizer.train_quantizer(vectors)
            else:
                quantizer.train(vectors)
        
        self.quantizers[name] = quantizer
        
        logger.info(f"Created quantizer '{name}' with method {config.method.value}")
    
    def quantize_vectors(
        self,
        quantizer_name: str,
        vectors: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> QuantizedIndex:
        """Quantize vectors using specified quantizer"""
        
        if quantizer_name not in self.quantizers:
            raise ValueError(f"Quantizer '{quantizer_name}' not found")
        
        quantizer = self.quantizers[quantizer_name]
        
        # Encode vectors
        compressed_vectors = quantizer.encode(vectors)
        
        # Create quantized index
        index = QuantizedIndex(
            quantizer=quantizer,
            codebook=getattr(quantizer, 'codebooks', None) or getattr(quantizer, 'codebook', None),
            compressed_vectors=compressed_vectors,
            metadata=metadata or {},
            config=quantizer.config
        )
        
        # Store index
        index_name = f"{quantizer_name}_index"
        self.indices[index_name] = index
        
        # Calculate compression stats
        original_size = vectors.nbytes
        compressed_size = compressed_vectors.nbytes
        compression_ratio = original_size / compressed_size
        
        logger.info(f"Quantized {len(vectors)} vectors. Compression ratio: {compression_ratio:.2f}x")
        
        return index
    
    def search_quantized(
        self,
        index_name: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        exact_reranking: bool = True
    ) -> List[Tuple[int, float]]:
        """Search in quantized index"""
        
        if index_name not in self.indices:
            raise ValueError(f"Index '{index_name}' not found")
        
        index = self.indices[index_name]
        quantizer = index.quantizer
        
        # Quantize query
        query_quantized = quantizer.encode(query_vector.reshape(1, -1))
        
        if index.config.method == QuantizationType.BINARY_QUANTIZATION:
            # Hamming distance for binary codes
            compressed_vectors = index.compressed_vectors
            query_code = query_quantized[0]
            
            # XOR and count bits
            distances = np.sum(compressed_vectors != query_code, axis=1)
            similarities = 1.0 / (1.0 + distances)  # Convert to similarity
            
        else:
            # Reconstruct for similarity calculation
            query_reconstructed = quantizer.decode(query_quantized)[0]
            compressed_reconstructed = quantizer.decode(index.compressed_vectors)
            
            # Cosine similarity
            query_norm = np.linalg.norm(query_reconstructed)
            doc_norms = np.linalg.norm(compressed_reconstructed, axis=1)
            
            similarities = np.dot(compressed_reconstructed, query_reconstructed) / (doc_norms * query_norm)
        
        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        return results
    
    def save_index(self, index_name: str, file_path: str) -> None:
        """Save quantized index to disk"""
        
        if index_name not in self.indices:
            raise ValueError(f"Index '{index_name}' not found")
        
        index = self.indices[index_name]
        
        with open(file_path, 'wb') as f:
            pickle.dump(index, f)
        
        logger.info(f"Saved index '{index_name}' to {file_path}")
    
    def load_index(self, index_name: str, file_path: str) -> None:
        """Load quantized index from disk"""
        
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            index = pickle.load(f)
        
        self.indices[index_name] = index
        
        # Also store the quantizer
        quantizer_name = index_name.replace('_index', '')
        self.quantizers[quantizer_name] = index.quantizer
        
        logger.info(f"Loaded index '{index_name}' from {file_path}")
    
    def get_compression_stats(self, index_name: str) -> Dict[str, Any]:
        """Get compression statistics for an index"""
        
        if index_name not in self.indices:
            raise ValueError(f"Index '{index_name}' not found")
        
        index = self.indices[index_name]
        
        # Estimate original size (assume float32)
        n_vectors, compressed_dim = index.compressed_vectors.shape
        
        if index.config.method == QuantizationType.PRODUCT_QUANTIZATION:
            quantizer = index.quantizer
            original_dim = quantizer.subvector_dim * quantizer.n_subvectors
            original_size = n_vectors * original_dim * 4  # float32
        else:
            # Estimate from compressed dimensions
            original_size = n_vectors * compressed_dim * 4 * (32 / index.config.n_bits)
        
        compressed_size = index.compressed_vectors.nbytes
        
        if index.codebook is not None:
            if isinstance(index.codebook, list):
                codebook_size = sum(cb.nbytes for cb in index.codebook)
            else:
                codebook_size = index.codebook.nbytes
        else:
            codebook_size = 0
        
        total_compressed_size = compressed_size + codebook_size
        compression_ratio = original_size / total_compressed_size
        
        return {
            'original_size_bytes': original_size,
            'compressed_size_bytes': compressed_size,
            'codebook_size_bytes': codebook_size,
            'total_size_bytes': total_compressed_size,
            'compression_ratio': compression_ratio,
            'n_vectors': n_vectors,
            'method': index.config.method.value,
            'n_bits': index.config.n_bits
        }

# Global instance
vector_quantizer = VectorQuantizationManager()

def get_vector_quantizer() -> VectorQuantizationManager:
    """Get the global vector quantization manager"""
    return vector_quantizer