"""
Qdrant client implementation for RAG operations.
Provides a unified interface for vector storage and retrieval using Qdrant.
"""

import json
import logging
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        VectorParams, Distance, HnswConfigDiff, SearchParams,
        PointStruct, Filter, FieldCondition, MatchValue
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from ..config import settings

logger = logging.getLogger(__name__)


class QdrantRAGClient:
    """Qdrant client for RAG operations with corpus management"""
    
    def __init__(self, 
                 url: str = None, 
                 api_key: str = None, 
                 timeout: float = None,
                 collection_prefix: str = None):
        """
        Initialize Qdrant client
        
        Args:
            url: Qdrant server URL
            api_key: API key for cloud Qdrant
            timeout: Request timeout in seconds
            collection_prefix: Prefix for collection names
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client not installed. Install with: pip install qdrant-client")
        
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        self.timeout = timeout or settings.qdrant_timeout
        self.collection_prefix = collection_prefix or settings.qdrant_collection_prefix
        
        # Initialize client
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=self.timeout
        )
        
        logger.info(f"Initialized Qdrant client: {self.url}")
    
    def _get_collection_name(self, corpus_name: str) -> str:
        """Generate collection name for corpus"""
        # Create deterministic collection name
        safe_name = corpus_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        return f"{self.collection_prefix}_{safe_name}"
    
    def create_corpus_collection(self, 
                                corpus_name: str, 
                                vector_dim: int,
                                distance: str = "Cosine") -> bool:
        """
        Create a new collection for a corpus
        
        Args:
            corpus_name: Name of the corpus
            vector_dim: Dimension of vectors
            distance: Distance metric (Cosine, Euclidean, Dot)
        
        Returns:
            True if successful
        """
        collection_name = self._get_collection_name(corpus_name)
        
        try:
            # Map distance string to Qdrant enum
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclidean": Distance.EUCLID,
                "Dot": Distance.DOT
            }
            distance_metric = distance_map.get(distance, Distance.COSINE)
            
            # Check if collection exists
            if self.client.collection_exists(collection_name):
                logger.info(f"Collection {collection_name} already exists")
                return True
            
            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_dim,
                    distance=distance_metric
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=128
                )
            )
            
            logger.info(f"Created Qdrant collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            return False
    
    def delete_corpus_collection(self, corpus_name: str) -> bool:
        """
        Delete a corpus collection
        
        Args:
            corpus_name: Name of the corpus
            
        Returns:
            True if successful
        """
        collection_name = self._get_collection_name(corpus_name)
        
        try:
            if self.client.collection_exists(collection_name):
                self.client.delete_collection(collection_name)
                logger.info(f"Deleted Qdrant collection: {collection_name}")
                return True
            else:
                logger.warning(f"Collection {collection_name} does not exist")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False
    
    def upsert_documents(self, 
                        corpus_name: str,
                        documents: List[Dict[str, Any]]) -> bool:
        """
        Upsert documents to a corpus collection
        
        Args:
            corpus_name: Name of the corpus
            documents: List of document dictionaries with 'id', 'vector', and metadata
            
        Returns:
            True if successful
        """
        collection_name = self._get_collection_name(corpus_name)
        
        try:
            # Prepare points for Qdrant
            points = []
            
            for doc in documents:
                point = PointStruct(
                    id=doc.get('id') or str(hash(doc.get('text', ''))),
                    vector=doc['vector'],
                    payload={
                        'text': doc.get('text', ''),
                        'document_id': doc.get('document_id', ''),
                        'chunk_index': doc.get('chunk_index', 0),
                        'metadata': doc.get('metadata', {}),
                        'timestamp': doc.get('timestamp', datetime.now().isoformat())
                    }
                )
                points.append(point)
            
            # Batch upsert
            batch_size = settings.qdrant_batch_size
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
            
            logger.info(f"Upserted {len(documents)} documents to {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert documents to {collection_name}: {e}")
            return False
    
    def search_vectors(self, 
                      corpus_name: str,
                      query_vector: List[float],
                      top_k: int = 10,
                      filter_conditions: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search vectors in a corpus collection
        
        Args:
            corpus_name: Name of the corpus
            query_vector: Query vector
            top_k: Number of results to return
            filter_conditions: Optional filter conditions
            
        Returns:
            List of search results
        """
        collection_name = self._get_collection_name(corpus_name)
        
        try:
            # Prepare filter if provided
            query_filter = None
            if filter_conditions:
                # Simple filter implementation - can be extended
                conditions = []
                for key, value in filter_conditions.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                if conditions:
                    query_filter = Filter(must=conditions)
            
            # Perform search
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                search_params=SearchParams(hnsw_ef=128)
            )
            
            # Format results
            results = []
            for result in search_result:
                results.append({
                    'id': result.id,
                    'score': result.score,
                    'text': result.payload.get('text', ''),
                    'document_id': result.payload.get('document_id', ''),
                    'chunk_index': result.payload.get('chunk_index', 0),
                    'metadata': result.payload.get('metadata', {}),
                    'timestamp': result.payload.get('timestamp', '')
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search in {collection_name}: {e}")
            return []
    
    def delete_document(self, 
                       corpus_name: str, 
                       document_id: str) -> bool:
        """
        Delete all chunks of a document from collection
        
        Args:
            corpus_name: Name of the corpus
            document_id: ID of the document to delete
            
        Returns:
            True if successful
        """
        collection_name = self._get_collection_name(corpus_name)
        
        try:
            # Delete points with matching document_id
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
            
            result = self.client.delete(
                collection_name=collection_name,
                points_selector=filter_condition
            )
            
            logger.info(f"Deleted document {document_id} from {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id} from {collection_name}: {e}")
            return False
    
    def get_collection_info(self, corpus_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a corpus collection
        
        Args:
            corpus_name: Name of the corpus
            
        Returns:
            Collection information or None
        """
        collection_name = self._get_collection_name(corpus_name)
        
        try:
            if not self.client.collection_exists(collection_name):
                return None
                
            info = self.client.get_collection(collection_name)
            return {
                'name': collection_name,
                'vectors_count': info.vectors_count,
                'indexed_vectors_count': info.indexed_vectors_count,
                'points_count': info.points_count,
                'segments_count': info.segments_count,
                'status': info.status,
                'optimizer_status': info.optimizer_status,
                'config': {
                    'vector_size': info.config.params.vectors.size,
                    'distance': info.config.params.vectors.distance,
                    'hnsw_config': {
                        'm': info.config.hnsw_config.m,
                        'ef_construct': info.config.hnsw_config.ef_construct
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info for {collection_name}: {e}")
            return None
    
    def list_collections(self) -> List[str]:
        """
        List all corpus collections
        
        Returns:
            List of corpus names
        """
        try:
            collections = self.client.get_collections()
            corpus_names = []
            
            for collection in collections.collections:
                if collection.name.startswith(self.collection_prefix):
                    # Extract corpus name from collection name
                    corpus_name = collection.name[len(self.collection_prefix) + 1:]
                    corpus_names.append(corpus_name)
            
            return corpus_names
            
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def scroll_points(self, 
                     corpus_name: str, 
                     limit: int = 100,
                     offset: Optional[str] = None) -> Tuple[List[Dict], Optional[str]]:
        """
        Scroll through points in a collection
        
        Args:
            corpus_name: Name of the corpus
            limit: Maximum number of points to return
            offset: Offset for pagination
            
        Returns:
            Tuple of (points, next_offset)
        """
        collection_name = self._get_collection_name(corpus_name)
        
        try:
            result = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points = []
            for point in result[0]:
                points.append({
                    'id': point.id,
                    'text': point.payload.get('text', ''),
                    'document_id': point.payload.get('document_id', ''),
                    'chunk_index': point.payload.get('chunk_index', 0),
                    'metadata': point.payload.get('metadata', {}),
                    'timestamp': point.payload.get('timestamp', '')
                })
            
            next_offset = result[1] if len(result) > 1 else None
            return points, next_offset
            
        except Exception as e:
            logger.error(f"Failed to scroll points in {collection_name}: {e}")
            return [], None
    
    def get_corpus_stats(self, corpus_name: str) -> Dict[str, Any]:
        """
        Get statistics for a corpus
        
        Args:
            corpus_name: Name of the corpus
            
        Returns:
            Statistics dictionary
        """
        collection_name = self._get_collection_name(corpus_name)
        
        try:
            if not self.client.collection_exists(collection_name):
                return {
                    'exists': False,
                    'points_count': 0,
                    'vectors_count': 0
                }
            
            info = self.client.get_collection(collection_name)
            return {
                'exists': True,
                'points_count': info.points_count,
                'vectors_count': info.vectors_count,
                'indexed_vectors_count': info.indexed_vectors_count,
                'segments_count': info.segments_count,
                'status': info.status
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats for {collection_name}: {e}")
            return {'exists': False, 'error': str(e)}
    
    def health_check(self) -> bool:
        """
        Check if Qdrant server is healthy
        
        Returns:
            True if healthy
        """
        try:
            # Try to get collections as a health check
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
    
    def close(self):
        """Close the Qdrant client connection"""
        try:
            if hasattr(self.client, 'close'):
                self.client.close()
        except Exception as e:
            logger.warning(f"Error closing Qdrant client: {e}")


# Global Qdrant client instance
_qdrant_client: Optional[QdrantRAGClient] = None


def get_qdrant_client() -> QdrantRAGClient:
    """Get or create the global Qdrant client instance"""
    global _qdrant_client
    
    if _qdrant_client is None:
        _qdrant_client = QdrantRAGClient()
    
    return _qdrant_client


def reset_qdrant_client():
    """Reset the global Qdrant client instance"""
    global _qdrant_client
    
    if _qdrant_client:
        _qdrant_client.close()
        _qdrant_client = None
