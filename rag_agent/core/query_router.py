"""
Smart query routing system that determines the best approach for answering queries.
Optimized for Windows CPU operation.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass
import re
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries and their routing strategy"""
    FACTUAL_RETRIEVAL = "factual_retrieval"  # Needs RAG
    CONVERSATIONAL = "conversational"  # Direct LLM
    ANALYTICAL = "analytical"  # RAG + reasoning
    SUMMARIZATION = "summarization"  # RAG for context
    COMPARISON = "comparison"  # Multi-doc RAG
    DEFINITION = "definition"  # Can use direct LLM
    PROCEDURAL = "procedural"  # Step-by-step, may need RAG
    CREATIVE = "creative"  # Direct LLM
    MULTIMODAL = "multimodal"  # Needs image processing

@dataclass
class RoutingDecision:
    """Container for routing decision"""
    query_type: QueryType
    use_rag: bool
    use_direct_llm: bool
    use_multimodal: bool
    confidence: float
    reasoning: str
    suggested_models: List[str]
    retrieval_params: Optional[Dict[str, Any]] = None

class QueryRouter:
    """
    Intelligent query router that determines the best approach for answering queries.
    CPU-optimized for Windows.
    """
    
    def __init__(self, 
                 ollama_client: Optional[Any] = None,
                 routing_model: str = "llama3.2",
                 use_llm_routing: bool = False):
        """
        Initialize query router.
        
        Args:
            ollama_client: Optional Ollama client for LLM-based routing
            routing_model: Model to use for routing decisions
            use_llm_routing: Whether to use LLM for routing (CPU-intensive)
        """
        self.ollama_client = ollama_client
        self.routing_model = routing_model
        self.use_llm_routing = use_llm_routing
        
        # Pattern-based routing rules (CPU-efficient)
        self.routing_patterns = {
            QueryType.FACTUAL_RETRIEVAL: [
                r"what (is|are|was|were)",
                r"when did",
                r"where (is|was|did)",
                r"who (is|was|were)",
                r"find (information|details|facts)",
                r"tell me about",
                r"explain the (concept|idea|theory)",
                r"according to",
                r"based on (the|our) (document|data|information)"
            ],
            QueryType.CONVERSATIONAL: [
                r"^(hi|hello|hey|greetings)",
                r"how are you",
                r"thank you",
                r"goodbye",
                r"nice to (meet|chat)",
                r"^(yes|no|okay|sure)$"
            ],
            QueryType.ANALYTICAL: [
                r"analyze",
                r"evaluate",
                r"assess",
                r"examine",
                r"investigate",
                r"what (caused|led to|resulted in)",
                r"why (did|does|is)",
                r"how does .* affect",
                r"relationship between",
                r"correlation"
            ],
            QueryType.SUMMARIZATION: [
                r"summar(ize|y)",
                r"brief(ly)?",
                r"overview",
                r"main points",
                r"key (findings|takeaways|insights)",
                r"gist",
                r"tl;?dr"
            ],
            QueryType.COMPARISON: [
                r"compar(e|ison)",
                r"differ(ence)?",
                r"similar(ity)?",
                r"versus|vs\.?",
                r"contrast",
                r"better|worse",
                r"advantages?.* disadvantages?"
            ],
            QueryType.DEFINITION: [
                r"define",
                r"what does .* mean",
                r"meaning of",
                r"definition of",
                r"^what is [a-z]+$"
            ],
            QueryType.PROCEDURAL: [
                r"how (to|do|can)",
                r"steps to",
                r"procedure",
                r"process (of|for)",
                r"guide (to|for)",
                r"instructions",
                r"tutorial"
            ],
            QueryType.CREATIVE: [
                r"create",
                r"generate",
                r"write (a|an)",
                r"compose",
                r"design",
                r"imagine",
                r"invent",
                r"come up with"
            ],
            QueryType.MULTIMODAL: [
                r"(image|picture|photo|diagram|chart|graph)",
                r"show me",
                r"visual",
                r"what does .* look like",
                r"describe the (image|picture)",
                r"in the (figure|illustration)"
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for query_type, patterns in self.routing_patterns.items():
            self.compiled_patterns[query_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
        
        logger.info(f"Query router initialized (LLM routing: {use_llm_routing})")
    
    def route_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """
        Determine the best routing for a query.
        
        Args:
            query: User query
            context: Optional context (conversation history, metadata)
            
        Returns:
            RoutingDecision with recommended approach
        """
        # Use pattern-based routing for CPU efficiency
        if not self.use_llm_routing or not self.ollama_client:
            return self._pattern_based_routing(query, context)
        else:
            # Use LLM-based routing (more accurate but CPU-intensive)
            return self._llm_based_routing(query, context)
    
    def _pattern_based_routing(self, query: str, context: Optional[Dict[str, Any]]) -> RoutingDecision:
        """
        Fast pattern-based routing for CPU efficiency.
        
        Args:
            query: User query
            context: Optional context
            
        Returns:
            RoutingDecision
        """
        # Score each query type
        scores = {}
        for query_type, patterns in self.compiled_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern.search(query):
                    score += 1
            scores[query_type] = score
        
        # Find best match
        if not any(scores.values()):
            # Default to factual retrieval if no patterns match
            best_type = QueryType.FACTUAL_RETRIEVAL
            confidence = 0.3
        else:
            best_type = max(scores, key=scores.get)
            max_score = scores[best_type]
            confidence = min(max_score / 3.0, 1.0)  # Normalize confidence
        
        # Determine routing based on query type
        routing_config = self._get_routing_config(best_type, query)
        
        return RoutingDecision(
            query_type=best_type,
            use_rag=routing_config["use_rag"],
            use_direct_llm=routing_config["use_direct_llm"],
            use_multimodal=routing_config["use_multimodal"],
            confidence=confidence,
            reasoning=routing_config["reasoning"],
            suggested_models=routing_config["models"],
            retrieval_params=routing_config.get("retrieval_params")
        )
    
    def _llm_based_routing(self, query: str, context: Optional[Dict[str, Any]]) -> RoutingDecision:
        """
        LLM-based routing for better accuracy (CPU-intensive).
        
        Args:
            query: User query
            context: Optional context
            
        Returns:
            RoutingDecision
        """
        prompt = f"""Analyze this query and determine the best approach to answer it.

Query: "{query}"

Classify the query type and routing strategy:
1. Query Type: [factual_retrieval, conversational, analytical, summarization, comparison, definition, procedural, creative, multimodal]
2. Use RAG: [true/false] - Should we search documents?
3. Use Direct LLM: [true/false] - Can the LLM answer directly?
4. Confidence: [0.0-1.0] - How confident are you?
5. Reasoning: Brief explanation

Respond in JSON format:
{{
    "query_type": "...",
    "use_rag": true/false,
    "use_direct_llm": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "..."
}}"""

        try:
            response = self.ollama_client.generate(
                model=self.routing_model,
                prompt=prompt,
                temperature=0.1,  # Low temperature for consistency
                max_tokens=200
            )
            
            # Parse JSON response
            result = json.loads(response)
            
            query_type = QueryType[result["query_type"].upper()]
            routing_config = self._get_routing_config(query_type, query)
            
            return RoutingDecision(
                query_type=query_type,
                use_rag=result["use_rag"],
                use_direct_llm=result["use_direct_llm"],
                use_multimodal=routing_config["use_multimodal"],
                confidence=result["confidence"],
                reasoning=result["reasoning"],
                suggested_models=routing_config["models"],
                retrieval_params=routing_config.get("retrieval_params")
            )
            
        except Exception as e:
            logger.warning(f"LLM routing failed, falling back to pattern-based: {e}")
            return self._pattern_based_routing(query, context)
    
    def _get_routing_config(self, query_type: QueryType, query: str) -> Dict[str, Any]:
        """
        Get routing configuration for a query type.
        
        Args:
            query_type: Type of query
            query: Original query
            
        Returns:
            Routing configuration
        """
        configs = {
            QueryType.FACTUAL_RETRIEVAL: {
                "use_rag": True,
                "use_direct_llm": False,
                "use_multimodal": False,
                "reasoning": "Factual query requires document retrieval",
                "models": ["llama3.2", "mistral"],
                "retrieval_params": {"top_k": 5, "similarity_threshold": 0.7}
            },
            QueryType.CONVERSATIONAL: {
                "use_rag": False,
                "use_direct_llm": True,
                "use_multimodal": False,
                "reasoning": "Conversational query can be handled directly",
                "models": ["llama3.2", "phi3"],
                "retrieval_params": None
            },
            QueryType.ANALYTICAL: {
                "use_rag": True,
                "use_direct_llm": True,
                "use_multimodal": False,
                "reasoning": "Analytical query needs context and reasoning",
                "models": ["llama3.2", "qwen2.5"],
                "retrieval_params": {"top_k": 10, "similarity_threshold": 0.6}
            },
            QueryType.SUMMARIZATION: {
                "use_rag": True,
                "use_direct_llm": True,
                "use_multimodal": False,
                "reasoning": "Summarization requires document context",
                "models": ["llama3.2", "mistral"],
                "retrieval_params": {"top_k": 15, "similarity_threshold": 0.5}
            },
            QueryType.COMPARISON: {
                "use_rag": True,
                "use_direct_llm": True,
                "use_multimodal": False,
                "reasoning": "Comparison needs multiple document sources",
                "models": ["llama3.2", "qwen2.5"],
                "retrieval_params": {"top_k": 10, "similarity_threshold": 0.6}
            },
            QueryType.DEFINITION: {
                "use_rag": False,
                "use_direct_llm": True,
                "use_multimodal": False,
                "reasoning": "Definition can often be provided directly",
                "models": ["llama3.2", "phi3"],
                "retrieval_params": None
            },
            QueryType.PROCEDURAL: {
                "use_rag": True,
                "use_direct_llm": True,
                "use_multimodal": False,
                "reasoning": "Procedural query may need specific instructions",
                "models": ["llama3.2", "mistral"],
                "retrieval_params": {"top_k": 7, "similarity_threshold": 0.65}
            },
            QueryType.CREATIVE: {
                "use_rag": False,
                "use_direct_llm": True,
                "use_multimodal": False,
                "reasoning": "Creative tasks are best handled directly",
                "models": ["llama3.2", "qwen2.5"],
                "retrieval_params": None
            },
            QueryType.MULTIMODAL: {
                "use_rag": True,
                "use_direct_llm": True,
                "use_multimodal": True,
                "reasoning": "Query involves visual content",
                "models": ["llava", "bakllava"],  # Vision models for Ollama
                "retrieval_params": {"top_k": 5, "include_images": True}
            }
        }
        
        return configs.get(query_type, configs[QueryType.FACTUAL_RETRIEVAL])
    
    def should_decompose_query(self, query: str) -> Tuple[bool, Optional[List[str]]]:
        """
        Determine if query should be decomposed into sub-queries.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (should_decompose, sub_queries)
        """
        # Check for complex query indicators
        complex_indicators = [
            r" and ",
            r" or ",
            r"first.*then",
            r"after that",
            r"multiple|several|various",
            r"\?.*\?",  # Multiple questions
            r"additionally|furthermore|moreover"
        ]
        
        is_complex = any(
            re.search(pattern, query, re.IGNORECASE) 
            for pattern in complex_indicators
        )
        
        if not is_complex:
            return False, None
        
        # Simple decomposition based on conjunctions
        sub_queries = []
        
        # Split on "and" or "or"
        parts = re.split(r'\s+(?:and|or)\s+', query, flags=re.IGNORECASE)
        if len(parts) > 1:
            # Preserve question context
            question_prefix = ""
            if query.lower().startswith(("what", "how", "why", "when", "where", "who")):
                question_prefix = query.split()[0] + " "
            
            for part in parts:
                if not part.strip().startswith(("what", "how", "why", "when", "where", "who")):
                    sub_queries.append(question_prefix + part.strip() + "?")
                else:
                    sub_queries.append(part.strip() + "?")
        
        return len(sub_queries) > 1, sub_queries if len(sub_queries) > 1 else None
    
    def optimize_for_cpu(self, decision: RoutingDecision) -> RoutingDecision:
        """
        Optimize routing decision for CPU-only operation.
        
        Args:
            decision: Original routing decision
            
        Returns:
            CPU-optimized routing decision
        """
        # Reduce retrieval parameters for CPU
        if decision.retrieval_params:
            decision.retrieval_params["top_k"] = min(decision.retrieval_params.get("top_k", 5), 5)
            decision.retrieval_params["batch_size"] = 1  # Single item processing
        
        # Prefer lighter models for CPU
        cpu_friendly_models = ["phi3", "llama3.2", "gemma2"]
        decision.suggested_models = [
            model for model in cpu_friendly_models 
            if model in decision.suggested_models or not decision.suggested_models
        ] or cpu_friendly_models[:1]
        
        return decision