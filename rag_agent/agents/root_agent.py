from google.adk import Agent
from google.adk.agents import LlmAgent
from typing import List, Dict, Any, Optional
import logging
import asyncio
import time

from ..config import settings
from ..core.indexing import HybridIndexer, CorpusManager
from ..core.document_processor import DocumentProcessor
from ..core.retrieval import HybridRetriever
from ..optimization.context_pruning import ContextPruner
from ..optimization.caching import RetrievalCache
from ..core.context_cache import context_cache_manager, CacheType

logger = logging.getLogger(__name__)


class EnhancedRAGRootAgent:
    """Root orchestrator agent for Enhanced RAG system"""
    
    def __init__(self, settings_obj: Optional[Any] = None):
        self.settings = settings_obj or settings
        self.setup_components()
        self.setup_sub_agents()
        
    def setup_components(self):
        """Initialize core components"""
        self.indexer = HybridIndexer(self.settings)
        self.corpus_manager = CorpusManager(self.settings)
        self.document_processor = DocumentProcessor(self.settings)
        self.retriever = HybridRetriever(self.settings)
        self.context_pruner = ContextPruner(self.settings)
        
        if self.settings.enable_caching:
            self.cache = RetrievalCache(self.settings)
        else:
            self.cache = None
        
    def setup_sub_agents(self):
        """Initialize specialized sub-agents"""
        
        # Indexing Agent - Handles document processing and indexing
        self.indexing_agent = LlmAgent(
            name="IndexingAgent",
            model=self.settings.generation_model,
            description="Manages document ingestion, parsing, and indexing",
            instruction="""You are an expert at processing and indexing documents.
            Your responsibilities include:
            1. Parsing documents using appropriate strategies
            2. Creating optimal chunk configurations
            3. Generating embeddings using selected strategies
            4. Managing corpus creation and updates
            """,
            tools=[
                self.create_corpus_tool(),
                self.add_documents_tool(),
                self.analyze_corpus_tool()
            ]
        )
        
        # Search Agent - Handles retrieval and ranking
        self.search_agent = LlmAgent(
            name="SearchAgent",
            model=self.settings.generation_model,
            description="Performs intelligent document retrieval and ranking",
            instruction="""You are a search specialist optimizing retrieval.
            Your tasks include:
            1. Executing hybrid searches across indices
            2. Applying ColBERT-style retrieval when needed
            3. Reranking results for relevance
            4. Managing retrieval caching
            """,
            tools=[
                self.hybrid_search_tool(),
                self.colbert_retrieval_tool(),
                self.rerank_results_tool()
            ]
        )
        
        # Generation Agent - Handles response generation
        self.generation_agent = LlmAgent(
            name="GenerationAgent",
            model=self.settings.generation_model,
            description="Generates accurate, grounded responses",
            instruction="""You generate precise answers from retrieved context.
            Your approach:
            1. Analyze retrieved chunks for relevance
            2. Apply context pruning when needed
            3. Generate factual, well-sourced responses
            4. Include citations and confidence scores
            """,
            tools=[
                self.prune_context_tool(),
                self.generate_response_tool()
            ]
        )
        
        # Root Agent - Orchestrates the entire system
        self.root_agent = Agent(
            name="EnhancedRAGOrchestrator",
            model=self.settings.generation_model,
            description="Orchestrates advanced RAG operations",
            instruction="""You coordinate the Enhanced RAG system.
            For queries, you:
            1. Analyze query intent and complexity
            2. Route to appropriate sub-agents
            3. Optimize retrieval strategy
            4. Ensure high-quality responses
            """,
            sub_agents=[
                self.indexing_agent,
                self.search_agent,
                self.generation_agent
            ]
        )
    
    def create_corpus_tool(self):
        """Tool for creating a new corpus with advanced configuration"""
        
        def create_corpus(
            corpus_name: str,
            description: str,
            indexing_strategy: str = "hybrid",
            chunk_size: int = 512,
            enable_colbert: bool = True,
            embedding_provider: str = "sentence_transformers",
            embedding_config: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """Create a new corpus with specified configuration"""
            
            try:
                corpus = self.corpus_manager.create_corpus(
                    name=corpus_name,
                    description=description,
                    indexing_strategy=indexing_strategy,
                    chunk_size=chunk_size,
                    enable_colbert=enable_colbert,
                    embedding_provider=embedding_provider,
                    embedding_config=embedding_config
                )
                
                # Store index reference
                self.retriever.set_index(corpus["id"], corpus["index"])
                
                return {
                    "status": "success",
                    "corpus_id": corpus["id"],
                    "configuration": {
                        "indexing_strategy": indexing_strategy,
                        "embedding_provider": embedding_provider,
                        "chunk_size": chunk_size,
                        "enable_colbert": enable_colbert
                    }
                }
            except Exception as e:
                logger.error(f"Corpus creation failed: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        return create_corpus
    
    def add_documents_tool(self):
        """Tool for adding documents with intelligent processing"""
        
        def add_documents(
            corpus_id: str,
            source_uris: List[str],
            use_layout_parser: bool = True,
            enable_late_chunking: bool = True
        ) -> Dict[str, Any]:
            """Add documents to corpus with advanced processing"""
            
            try:
                results = self.document_processor.process_documents(
                    corpus_id=corpus_id,
                    uris=source_uris,
                    use_layout_parser=use_layout_parser,
                    enable_late_chunking=enable_late_chunking
                )
                
                # Invalidate cache for this corpus
                if self.cache:
                    self.cache.invalidate_corpus(corpus_id)
                
                return {
                    "status": "success",
                    "processed_files": results["processed_count"],
                    "chunks_created": results["total_chunks"],
                    "processing_time": results["processing_time"],
                    "errors": results.get("errors", [])
                }
            except Exception as e:
                logger.error(f"Document processing failed: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        return add_documents
    
    def analyze_corpus_tool(self):
        """Tool for analyzing corpus statistics"""
        
        def analyze_corpus(corpus_id: str) -> Dict[str, Any]:
            """Get detailed corpus statistics"""
            
            try:
                config = self.corpus_manager.get_config(corpus_id)
                
                return {
                    "status": "success",
                    "corpus_id": corpus_id,
                    "metadata": config["metadata"],
                    "indexing_info": {
                        "has_dense_index": config["has_dense_index"],
                        "has_sparse_index": config["has_sparse_index"],
                        "has_colbert_index": config["has_colbert_index"]
                    }
                }
            except Exception as e:
                logger.error(f"Corpus analysis failed: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        return analyze_corpus
    
    def hybrid_search_tool(self):
        """Tool for performing hybrid search"""
        
        def hybrid_search(
            query: str,
            corpus_id: str,
            top_k: int = 10,
            alpha: float = 0.5,
            use_cache: bool = True
        ) -> List[Dict[str, Any]]:
            """Execute hybrid search combining dense and sparse retrieval"""
            
            try:
                results = self.retriever.search(
                    query=query,
                    corpus_id=corpus_id,
                    top_k=top_k,
                    alpha=alpha,
                    use_cache=use_cache
                )
                
                return results
            except Exception as e:
                logger.error(f"Hybrid search failed: {e}")
                return []
        
        return hybrid_search
    
    def colbert_retrieval_tool(self):
        """Tool for ColBERT-specific retrieval"""
        
        def colbert_retrieval(
            query: str,
            corpus_id: str,
            top_k: int = 10
        ) -> List[Dict[str, Any]]:
            """Execute ColBERT retrieval"""
            
            # This is handled within hybrid search when enabled
            return self.hybrid_search_tool()(
                query=query,
                corpus_id=corpus_id,
                top_k=top_k,
                alpha=0.0,  # Pure ColBERT
                use_cache=True
            )
        
        return colbert_retrieval
    
    def rerank_results_tool(self):
        """Tool for reranking search results"""
        
        def rerank_results(
            query: str,
            results: List[Dict[str, Any]],
            top_k: int = 10
        ) -> List[Dict[str, Any]]:
            """Rerank search results for better relevance"""
            
            # Simple reranking based on score
            # In production, use a cross-encoder model
            sorted_results = sorted(
                results,
                key=lambda x: x.get("score", 0),
                reverse=True
            )
            
            return sorted_results[:top_k]
        
        return rerank_results
    
    def prune_context_tool(self):
        """Tool for intelligent context pruning"""
        
        def prune_context(
            query: str,
            chunks: List[Dict[str, Any]],
            max_tokens: int = 8192
        ) -> Dict[str, Any]:
            """Prune context to most relevant information"""
            
            try:
                pruned_chunks, stats = self.context_pruner.prune_context(
                    query=query,
                    chunks=chunks,
                    max_tokens=max_tokens
                )
                
                return {
                    "chunks": pruned_chunks,
                    "stats": stats
                }
            except Exception as e:
                logger.error(f"Context pruning failed: {e}")
                return {
                    "chunks": chunks[:10],  # Fallback to simple truncation
                    "stats": {"error": str(e)}
                }
        
        return prune_context
    
    def generate_response_tool(self):
        """Tool for generating final response"""
        
        def generate_response(
            query: str,
            context_chunks: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            """Generate response from context"""
            
            # Format context
            context = "\n\n".join([
                f"[{i+1}] {chunk.get('text', '')}"
                for i, chunk in enumerate(context_chunks)
            ])
            
            # Simple response generation
            # In production, use a more sophisticated approach
            response = f"Based on the retrieved information:\n\n{context[:2000]}..."
            
            return {
                "response": response,
                "confidence": 0.85,  # Placeholder
                "citations": [i+1 for i in range(len(context_chunks))]
            }
        
        return generate_response
    
    async def process_query(
        self,
        query: str,
        corpus_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Main entry point for query processing"""
        
        start_time = time.time()
        
        # Search for relevant chunks
        retrieval_start = time.time()
        search_results = self.hybrid_search_tool()(
            query=query,
            corpus_id=corpus_id,
            top_k=kwargs.get("top_k", 10),
            alpha=kwargs.get("alpha", 0.5),
            use_cache=kwargs.get("use_cache", True)
        )
        retrieval_time = time.time() - retrieval_start
        
        # Prune context if enabled
        if kwargs.get("enable_pruning", True):
            pruning_result = self.prune_context_tool()(
                query=query,
                chunks=search_results,
                max_tokens=kwargs.get("max_context_tokens", 8192)
            )
            context_chunks = pruning_result["chunks"]
            pruning_stats = pruning_result["stats"]
        else:
            context_chunks = search_results[:kwargs.get("top_k", 10)]
            pruning_stats = None
        
        # Generate response
        generation_start = time.time()
        response_data = self.generate_response_tool()(
            query=query,
            context_chunks=context_chunks
        )
        generation_time = time.time() - generation_start
        
        # Compile final response
        final_response = {
            "query": query,
            "chunks_retrieved": context_chunks,
            "response": response_data["response"],
            "confidence": response_data.get("confidence", 0.0),
            "metadata": {
                "total_time": time.time() - start_time,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "cache_hit": False,  # TODO: Track actual cache hits
                "pruning_stats": pruning_stats
            }
        }
        
        return final_response
    
    async def create_corpus(self, **kwargs) -> Dict[str, Any]:
        """Create a new corpus"""
        return self.create_corpus_tool()(**kwargs)
    
    async def add_documents(self, **kwargs) -> Dict[str, Any]:
        """Add documents to corpus"""
        return self.add_documents_tool()(**kwargs)
    
    async def get_corpus_stats(self, corpus_id: str) -> Dict[str, Any]:
        """Get corpus statistics"""
        return self.analyze_corpus_tool()(corpus_id)
