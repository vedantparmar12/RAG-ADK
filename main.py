"""
Enhanced Vertex AI RAG Agent with FastAPI interface.
Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import logging
import numpy as np
from datetime import datetime

from rag_agent.config import settings
from rag_agent.agents.root_agent import EnhancedRAGRootAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced Vertex AI RAG Agent",
    description="Advanced RAG system with hybrid search, ColBERT retrieval, and intelligent chunking",
    version="2.0.0"
)

class QueryRequest(BaseModel):
    """Query request with advanced options"""
    query: str
    corpus_id: str
    
    # Retrieval options
    top_k: Optional[int] = Field(10, ge=1, le=100)
    retrieval_strategy: Optional[str] = Field("hybrid", pattern="^(dense|sparse|hybrid|colbert)$")
    alpha: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    
    # Performance options
    use_cache: Optional[bool] = True
    enable_pruning: Optional[bool] = True
    max_context_tokens: Optional[int] = Field(8192, ge=1024, le=32768)
    
    # Advanced options
    enable_reranking: Optional[bool] = True
    enable_query_expansion: Optional[bool] = False
    return_chunks: Optional[bool] = False
    
class CorpusRequest(BaseModel):
    """Corpus creation request"""
    name: str
    description: str
    
    # Indexing configuration
    indexing_strategy: Optional[str] = Field("hybrid", pattern="^(dense|sparse|hybrid)$")
    chunk_size: Optional[int] = Field(512, ge=128, le=2048)
    chunk_overlap: Optional[int] = Field(64, ge=0, le=256)
    
    # Advanced features
    enable_layout_parser: Optional[bool] = True
    enable_late_chunking: Optional[bool] = True
    enable_colbert: Optional[bool] = True

class DocumentRequest(BaseModel):
    """Document addition request"""
    uris: List[str] = Field(..., description="List of document URIs (GCS, Drive, etc.)")
    use_layout_parser: Optional[bool] = True
    enable_late_chunking: Optional[bool] = True

# Initialize RAG system
try:
    rag_system = EnhancedRAGRootAgent(settings)
    logger.info("Enhanced RAG system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}")
    rag_system = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "version": "2.0.0",
        "rag_enabled": rag_system is not None,
        "settings": {
            "indexing_strategy": settings.indexing_strategy,
            "enable_colbert": settings.enable_colbert,
            "enable_caching": settings.enable_caching
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if rag_system else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "rag_system": "online" if rag_system else "offline",
            "caching": "enabled" if settings.enable_caching else "disabled",
            "colbert": "enabled" if settings.enable_colbert else "disabled"
        }
    }

@app.post("/query", response_model=Dict[str, Any])
async def query_documents(request: QueryRequest):
    """Execute RAG query with advanced options"""
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        # Process query
        response = await rag_system.process_query(
            query=request.query,
            corpus_id=request.corpus_id,
            top_k=request.top_k,
            retrieval_strategy=request.retrieval_strategy,
            alpha=request.alpha,
            use_cache=request.use_cache,
            enable_pruning=request.enable_pruning,
            max_context_tokens=request.max_context_tokens,
            enable_reranking=request.enable_reranking,
            enable_query_expansion=request.enable_query_expansion
        )
        
        # Format response
        formatted_response = {
            "query": request.query,
            "answer": response.get("response", "No response generated"),
            "confidence": response.get("confidence", 0.0),
            "metadata": {
                "chunks_retrieved": len(response.get("chunks_retrieved", [])),
                "retrieval_time": response.get("metadata", {}).get("retrieval_time"),
                "generation_time": response.get("metadata", {}).get("generation_time"),
                "cache_hit": response.get("metadata", {}).get("cache_hit", False),
                "pruning_stats": response.get("metadata", {}).get("pruning_stats")
            }
        }
        
        if request.return_chunks:
            formatted_response["chunks"] = response.get("chunks_retrieved", [])
            
        return formatted_response
        
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/corpus", response_model=Dict[str, Any])
async def create_corpus(request: CorpusRequest):
    """Create new corpus with configuration"""
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        result = await rag_system.create_corpus(
            corpus_name=request.name,
            description=request.description,
            indexing_strategy=request.indexing_strategy,
            chunk_size=request.chunk_size,
            enable_colbert=request.enable_colbert
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Corpus creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/corpus/{corpus_id}/documents")
async def add_documents(
    corpus_id: str,
    request: DocumentRequest,
    background_tasks: BackgroundTasks
):
    """Add documents to corpus"""
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    # Process in background
    background_tasks.add_task(
        process_documents_async,
        corpus_id,
        request.uris,
        request.use_layout_parser,
        request.enable_late_chunking
    )
    
    return {
        "status": "processing",
        "corpus_id": corpus_id,
        "document_count": len(request.uris),
        "message": "Documents are being processed in the background"
    }

@app.get("/corpus/{corpus_id}/stats")
async def get_corpus_stats(corpus_id: str):
    """Get corpus statistics"""
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        stats = await rag_system.get_corpus_stats(corpus_id)
        return stats
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/stats")
async def get_cache_stats():
    """Get caching statistics"""
    
    if not rag_system or not rag_system.cache:
        return {"cache_enabled": False}
        
    return rag_system.cache.get_stats()

@app.post("/cache/invalidate/{corpus_id}")
async def invalidate_cache(corpus_id: str):
    """Invalidate cache for a corpus"""
    
    if not rag_system or not rag_system.cache:
        return {"status": "cache_disabled"}
        
    rag_system.cache.invalidate_corpus(corpus_id)
    
    return {
        "status": "success",
        "corpus_id": corpus_id,
        "message": "Cache invalidated successfully"
    }

@app.post("/evaluate")
async def evaluate_configuration(
    test_queries: List[str],
    corpus_id: str,
    config_a: Dict[str, Any],
    config_b: Dict[str, Any]
):
    """A/B test different configurations"""
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    results = {
        "config_a": [],
        "config_b": [],
        "comparison": {}
    }
    
    for query in test_queries:
        try:
            # Test config A
            response_a = await rag_system.process_query(
                query=query,
                corpus_id=corpus_id,
                **config_a
            )
            results["config_a"].append({
                "query": query,
                "retrieval_time": response_a.get("metadata", {}).get("retrieval_time", 0),
                "quality_score": response_a.get("confidence", 0.0)
            })
            
            # Test config B
            response_b = await rag_system.process_query(
                query=query,
                corpus_id=corpus_id,
                **config_b
            )
            results["config_b"].append({
                "query": query,
                "retrieval_time": response_b.get("metadata", {}).get("retrieval_time", 0),
                "quality_score": response_b.get("confidence", 0.0)
            })
        except Exception as e:
            logger.error(f"Evaluation error for query '{query}': {e}")
    
    # Calculate comparison metrics
    if results["config_a"] and results["config_b"]:
        results["comparison"] = {
            "avg_retrieval_time_a": np.mean([r["retrieval_time"] for r in results["config_a"]]),
            "avg_retrieval_time_b": np.mean([r["retrieval_time"] for r in results["config_b"]]),
            "avg_quality_a": np.mean([r["quality_score"] for r in results["config_a"]]),
            "avg_quality_b": np.mean([r["quality_score"] for r in results["config_b"]])
        }
    
    return results

async def process_documents_async(
    corpus_id: str,
    uris: List[str],
    use_layout_parser: bool,
    enable_late_chunking: bool
):
    """Background task for document processing"""
    try:
        result = await rag_system.add_documents(
            corpus_id=corpus_id,
            source_uris=uris,
            use_layout_parser=use_layout_parser,
            enable_late_chunking=enable_late_chunking
        )
        logger.info(f"Processed {result['processed_count']} documents for corpus {corpus_id}")
    except Exception as e:
        logger.error(f"Document processing error: {e}")

if __name__ == "__main__":
    # Run with: python main.py
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )