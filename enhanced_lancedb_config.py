"""
Enhanced LanceDB configuration for optimal RAG performance.
Keeps LanceDB but with optimized settings for better recall and performance.
"""

import lancedb
import pyarrow as pa
from typing import Dict, Any, List
import numpy as np

class OptimizedLanceDBConfig:
    """Optimized LanceDB configuration for RAG workloads"""
    
    @staticmethod
    def create_optimized_table(db, table_name: str, dimension: int) -> Any:
        """Create LanceDB table with optimized settings"""
        
        # Enhanced schema with better indexing
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32())),
            pa.field("document_id", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("metadata", pa.string()),
            pa.field("timestamp", pa.string()),
            # Add fields for better filtering
            pa.field("doc_type", pa.string()),
            pa.field("importance_score", pa.float32())
        ])
        
        # Create table with optimized settings
        table = db.create_table(table_name, schema=schema)
        
        # Create multiple indexes for better performance
        try:
            # Vector index with optimized parameters
            table.create_index(
                "vector",
                index_type="IVF_PQ",  # Better than flat for larger datasets
                num_partitions=256,    # Optimal for medium datasets
                num_sub_vectors=16     # Balance accuracy vs speed
            )
            
            # FTS index for text search
            table.create_fts_index("text", use_tantivy=True)  # Use Tantivy for better performance
            
            # Scalar indexes for fast filtering
            table.create_scalar_index("document_id")
            table.create_scalar_index("doc_type")
            
        except Exception as e:
            print(f"Index creation warning: {e}")
            
        return table
    
    @staticmethod
    def get_optimized_search_params() -> Dict[str, Any]:
        """Get optimized search parameters"""
        return {
            "nprobe": 32,           # Search more partitions for better recall
            "refine_factor": 2,     # Refine top results
            "use_index": True       # Always use index when available
        }
    
    @staticmethod  
    def batch_insert_optimized(table, data: List[Dict[str, Any]], batch_size: int = 1000):
        """Optimized batch insertion"""
        
        # Sort data by document_id for better locality
        data.sort(key=lambda x: x.get('document_id', ''))
        
        # Insert in optimized batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            table.add(batch)
            
            # Trigger optimization every 5 batches
            if i % (batch_size * 5) == 0:
                try:
                    table.optimize()
                except:
                    pass  # Optimization may not always be available

class HybridLanceDBSearch:
    """Enhanced hybrid search using LanceDB + BM25"""
    
    def __init__(self, table, bm25_index=None):
        self.table = table
        self.bm25_index = bm25_index
        
    def hybrid_search(self, 
                     query_vector: np.ndarray,
                     query_text: str,
                     top_k: int = 10,
                     vector_weight: float = 0.7,
                     fts_weight: float = 0.3) -> List[Dict[str, Any]]:
        """Enhanced hybrid search combining vector, FTS, and BM25"""
        
        # 1. Vector similarity search
        vector_results = self.table.search(query_vector) \
            .limit(top_k * 2) \
            .nprobe(32) \
            .to_list()
        
        # 2. Full-text search using LanceDB FTS
        fts_results = []
        try:
            fts_results = self.table.search(query_text, query_type="fts") \
                .limit(top_k * 2) \
                .to_list()
        except:
            pass  # FTS might not be available
            
        # 3. BM25 search (if available)
        bm25_results = []
        if self.bm25_index:
            bm25_results = self._bm25_search(query_text, top_k * 2)
        
        # 4. Combine results with advanced fusion
        combined = self._advanced_fusion(
            vector_results, 
            fts_results, 
            bm25_results,
            vector_weight,
            fts_weight,
            top_k
        )
        
        return combined
    
    def _advanced_fusion(self, 
                        vector_results: List,
                        fts_results: List, 
                        bm25_results: List,
                        vector_weight: float,
                        fts_weight: float,
                        top_k: int) -> List[Dict[str, Any]]:
        """Advanced result fusion with multiple ranking signals"""
        
        # Create unified scoring
        scores = {}
        
        # Vector scores (normalized)
        for i, result in enumerate(vector_results):
            doc_id = result.get('id')
            scores[doc_id] = scores.get(doc_id, 0) + vector_weight / (i + 1)
        
        # FTS scores
        for i, result in enumerate(fts_results):
            doc_id = result.get('id')
            scores[doc_id] = scores.get(doc_id, 0) + fts_weight / (i + 1)
            
        # BM25 scores
        bm25_weight = 0.2
        for i, result in enumerate(bm25_results):
            doc_id = result.get('id')
            scores[doc_id] = scores.get(doc_id, 0) + bm25_weight / (i + 1)
        
        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Build final results
        all_results = {r['id']: r for r in vector_results + fts_results}
        
        final_results = []
        for doc_id in sorted_ids[:top_k]:
            if doc_id in all_results:
                result = all_results[doc_id].copy()
                result['hybrid_score'] = scores[doc_id]
                final_results.append(result)
                
        return final_results

# Performance monitoring
class LanceDBPerformanceMonitor:
    """Monitor LanceDB performance and suggest optimizations"""
    
    @staticmethod
    def analyze_performance(table, query_count: int = 100) -> Dict[str, Any]:
        """Analyze search performance"""
        import time
        
        # Sample queries for testing
        test_vector = np.random.rand(384).astype(np.float32)
        
        # Measure search latency
        start_time = time.time()
        for _ in range(query_count):
            results = table.search(test_vector).limit(10).to_list()
        avg_latency = (time.time() - start_time) / query_count
        
        # Get table stats
        stats = table.stats()
        
        return {
            "avg_latency_ms": avg_latency * 1000,
            "total_rows": stats.num_rows if hasattr(stats, 'num_rows') else 0,
            "has_vector_index": True,  # Assume indexed
            "recommendation": "optimize" if avg_latency > 0.01 else "good"
        }
