"""
Query decomposition system for breaking complex queries into manageable sub-queries.
Optimized for Windows CPU operation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
import json
from enum import Enum

logger = logging.getLogger(__name__)

class DecompositionStrategy(Enum):
    """Strategies for query decomposition"""
    SEQUENTIAL = "sequential"  # Questions that build on each other
    PARALLEL = "parallel"  # Independent questions
    HIERARCHICAL = "hierarchical"  # Main question with sub-questions
    CONDITIONAL = "conditional"  # Questions with conditions

@dataclass
class SubQuery:
    """Container for a decomposed sub-query"""
    query: str
    query_id: str
    parent_id: Optional[str]
    strategy: DecompositionStrategy
    dependencies: List[str]  # IDs of queries this depends on
    weight: float  # Importance weight
    metadata: Dict[str, Any]

@dataclass 
class DecomposedQuery:
    """Container for decomposed query results"""
    original_query: str
    sub_queries: List[SubQuery]
    strategy: DecompositionStrategy
    requires_aggregation: bool
    aggregation_prompt: Optional[str]

class QueryDecomposer:
    """
    Decomposes complex queries into simpler sub-queries.
    CPU-optimized for Windows.
    """
    
    def __init__(self,
                 ollama_client: Optional[Any] = None,
                 decomposition_model: str = "llama3.2",
                 use_llm_decomposition: bool = False,
                 max_sub_queries: int = 5):
        """
        Initialize query decomposer.
        
        Args:
            ollama_client: Optional Ollama client for LLM-based decomposition
            decomposition_model: Model to use for decomposition
            use_llm_decomposition: Whether to use LLM (CPU-intensive)
            max_sub_queries: Maximum number of sub-queries to generate
        """
        self.ollama_client = ollama_client
        self.decomposition_model = decomposition_model
        self.use_llm_decomposition = use_llm_decomposition
        self.max_sub_queries = max_sub_queries
        
        # Decomposition patterns for CPU-efficient processing
        self.decomposition_patterns = {
            "multi_aspect": {
                "patterns": [
                    r"(.*?)\s+and\s+(.*?)(?:\s+and\s+|$)",
                    r"(.*?),\s+(.*?)(?:,\s+|$)",
                    r"both\s+(.*?)\s+and\s+(.*)",
                    r"(.*?)\s+as well as\s+(.*)"
                ],
                "strategy": DecompositionStrategy.PARALLEL
            },
            "sequential": {
                "patterns": [
                    r"first\s+(.*?)[,.]?\s+then\s+(.*)",
                    r"(.*?)\s+followed by\s+(.*)",
                    r"after\s+(.*?)[,.]?\s+(.*)",
                    r"before\s+(.*?)[,.]?\s+(.*)"
                ],
                "strategy": DecompositionStrategy.SEQUENTIAL
            },
            "comparative": {
                "patterns": [
                    r"compare\s+(.*?)\s+(?:with|and|to)\s+(.*)",
                    r"difference between\s+(.*?)\s+and\s+(.*)",
                    r"(.*?)\s+versus\s+(.*)",
                    r"(.*?)\s+vs\.?\s+(.*)"
                ],
                "strategy": DecompositionStrategy.PARALLEL
            },
            "conditional": {
                "patterns": [
                    r"if\s+(.*?)[,.]?\s+then\s+(.*)",
                    r"when\s+(.*?)[,.]?\s+(.*)",
                    r"(.*?)\s+only if\s+(.*)",
                    r"(.*?)\s+unless\s+(.*)"
                ],
                "strategy": DecompositionStrategy.CONDITIONAL
            },
            "hierarchical": {
                "patterns": [
                    r"(.*?)\s+including\s+(.*)",
                    r"(.*?)\s+such as\s+(.*)",
                    r"(.*?)\s+especially\s+(.*)",
                    r"(.*?)\s+particularly\s+(.*)"
                ],
                "strategy": DecompositionStrategy.HIERARCHICAL
            }
        }
        
        logger.info(f"Query decomposer initialized (LLM: {use_llm_decomposition})")
    
    def decompose(self, query: str, context: Optional[Dict[str, Any]] = None) -> DecomposedQuery:
        """
        Decompose a complex query into sub-queries.
        
        Args:
            query: Complex query to decompose
            context: Optional context
            
        Returns:
            DecomposedQuery with sub-queries
        """
        # Check if decomposition is needed
        if not self._needs_decomposition(query):
            return DecomposedQuery(
                original_query=query,
                sub_queries=[],
                strategy=DecompositionStrategy.PARALLEL,
                requires_aggregation=False,
                aggregation_prompt=None
            )
        
        # Use appropriate decomposition method
        if self.use_llm_decomposition and self.ollama_client:
            return self._llm_decomposition(query, context)
        else:
            return self._pattern_decomposition(query, context)
    
    def _needs_decomposition(self, query: str) -> bool:
        """
        Check if query needs decomposition.
        
        Args:
            query: Query to check
            
        Returns:
            True if decomposition is beneficial
        """
        # Length check
        if len(query.split()) < 10:
            return False
        
        # Complexity indicators
        complexity_indicators = [
            r"\band\b",
            r"\bor\b",
            r",",
            r"multiple|several|various",
            r"first.*then",
            r"compare|versus|difference",
            r"if.*then",
            r"including|such as"
        ]
        
        complexity_score = sum(
            1 for pattern in complexity_indicators
            if re.search(pattern, query, re.IGNORECASE)
        )
        
        return complexity_score >= 2
    
    def _pattern_decomposition(self, query: str, context: Optional[Dict[str, Any]]) -> DecomposedQuery:
        """
        Pattern-based decomposition for CPU efficiency.
        
        Args:
            query: Query to decompose
            context: Optional context
            
        Returns:
            DecomposedQuery
        """
        sub_queries = []
        best_strategy = DecompositionStrategy.PARALLEL
        
        # Try each decomposition pattern
        for pattern_type, config in self.decomposition_patterns.items():
            for pattern in config["patterns"]:
                matches = re.findall(pattern, query, re.IGNORECASE)
                if matches:
                    best_strategy = config["strategy"]
                    
                    # Extract sub-queries from matches
                    for match in matches:
                        if isinstance(match, tuple):
                            for sub_query in match:
                                if sub_query and len(sub_query.strip()) > 5:
                                    sub_queries.append(sub_query.strip())
                        else:
                            if match and len(match.strip()) > 5:
                                sub_queries.append(match.strip())
                    
                    if sub_queries:
                        break
            
            if sub_queries:
                break
        
        # Fallback: split on conjunctions
        if not sub_queries:
            parts = re.split(r'\s+(?:and|or|but)\s+', query, flags=re.IGNORECASE)
            sub_queries = [p.strip() for p in parts if len(p.strip()) > 5]
            
            if len(sub_queries) <= 1:
                # No decomposition possible
                return DecomposedQuery(
                    original_query=query,
                    sub_queries=[],
                    strategy=DecompositionStrategy.PARALLEL,
                    requires_aggregation=False,
                    aggregation_prompt=None
                )
        
        # Limit number of sub-queries
        sub_queries = sub_queries[:self.max_sub_queries]
        
        # Create SubQuery objects
        sub_query_objects = []
        for i, sq in enumerate(sub_queries):
            # Ensure sub-query is a complete question
            sq = self._ensure_question_format(sq, query)
            
            sub_query_obj = SubQuery(
                query=sq,
                query_id=f"sq_{i+1}",
                parent_id=None,
                strategy=best_strategy,
                dependencies=[] if best_strategy == DecompositionStrategy.PARALLEL else [f"sq_{i}"] if i > 0 else [],
                weight=1.0 / len(sub_queries),
                metadata={"index": i, "pattern_type": pattern_type if sub_queries else "conjunction"}
            )
            sub_query_objects.append(sub_query_obj)
        
        # Create aggregation prompt
        aggregation_prompt = self._create_aggregation_prompt(query, sub_query_objects)
        
        return DecomposedQuery(
            original_query=query,
            sub_queries=sub_query_objects,
            strategy=best_strategy,
            requires_aggregation=True,
            aggregation_prompt=aggregation_prompt
        )
    
    def _llm_decomposition(self, query: str, context: Optional[Dict[str, Any]]) -> DecomposedQuery:
        """
        LLM-based decomposition for better accuracy (CPU-intensive).
        
        Args:
            query: Query to decompose
            context: Optional context
            
        Returns:
            DecomposedQuery
        """
        prompt = f"""Decompose this complex query into simpler sub-queries that can be answered independently.

Original Query: "{query}"

Rules:
1. Create 2-{self.max_sub_queries} focused sub-queries
2. Each sub-query should be answerable on its own
3. Together they should cover all aspects of the original query
4. Make each sub-query a complete question

Respond in JSON format:
{{
    "sub_queries": [
        {{"query": "...", "weight": 0.0-1.0}},
        ...
    ],
    "strategy": "parallel|sequential|hierarchical|conditional",
    "requires_aggregation": true/false
}}"""

        try:
            response = self.ollama_client.generate(
                model=self.decomposition_model,
                prompt=prompt,
                temperature=0.3,
                max_tokens=300
            )
            
            # Parse JSON response
            result = json.loads(response)
            
            # Create SubQuery objects
            sub_query_objects = []
            for i, sq_data in enumerate(result["sub_queries"]):
                sub_query_obj = SubQuery(
                    query=sq_data["query"],
                    query_id=f"sq_{i+1}",
                    parent_id=None,
                    strategy=DecompositionStrategy[result["strategy"].upper()],
                    dependencies=[],
                    weight=sq_data.get("weight", 1.0 / len(result["sub_queries"])),
                    metadata={"index": i, "source": "llm"}
                )
                sub_query_objects.append(sub_query_obj)
            
            aggregation_prompt = self._create_aggregation_prompt(query, sub_query_objects)
            
            return DecomposedQuery(
                original_query=query,
                sub_queries=sub_query_objects,
                strategy=DecompositionStrategy[result["strategy"].upper()],
                requires_aggregation=result["requires_aggregation"],
                aggregation_prompt=aggregation_prompt
            )
            
        except Exception as e:
            logger.warning(f"LLM decomposition failed, using pattern-based: {e}")
            return self._pattern_decomposition(query, context)
    
    def _ensure_question_format(self, sub_query: str, original_query: str) -> str:
        """
        Ensure sub-query is formatted as a complete question.
        
        Args:
            sub_query: Sub-query to format
            original_query: Original query for context
            
        Returns:
            Formatted question
        """
        # If already a question, return as-is
        if sub_query.strip().endswith("?"):
            return sub_query
        
        # Extract question word from original if present
        question_words = ["what", "how", "why", "when", "where", "who", "which"]
        original_lower = original_query.lower()
        
        for qw in question_words:
            if original_lower.startswith(qw):
                # Check if sub-query already has a question word
                if not any(sub_query.lower().startswith(q) for q in question_words):
                    return f"{qw.capitalize()} {sub_query}?"
                break
        
        # Add question mark if missing
        if not sub_query.endswith("?"):
            sub_query += "?"
        
        return sub_query
    
    def _create_aggregation_prompt(self, original_query: str, sub_queries: List[SubQuery]) -> str:
        """
        Create prompt for aggregating sub-query answers.
        
        Args:
            original_query: Original query
            sub_queries: List of sub-queries
            
        Returns:
            Aggregation prompt
        """
        sub_query_list = "\n".join([f"- {sq.query}" for sq in sub_queries])
        
        return f"""Based on the answers to these sub-questions:
{sub_query_list}

Provide a comprehensive answer to the original question:
"{original_query}"

Synthesize the information from all sub-answers into a coherent response."""
    
    def execute_decomposed_query(self, 
                                decomposed: DecomposedQuery,
                                query_executor: Any) -> Dict[str, Any]:
        """
        Execute decomposed queries and aggregate results.
        
        Args:
            decomposed: Decomposed query
            query_executor: Function to execute individual queries
            
        Returns:
            Aggregated results
        """
        if not decomposed.sub_queries:
            # No decomposition, execute original
            return query_executor(decomposed.original_query)
        
        # Execute sub-queries based on strategy
        sub_results = []
        
        if decomposed.strategy == DecompositionStrategy.PARALLEL:
            # Execute all in parallel (simulated for CPU)
            for sq in decomposed.sub_queries:
                result = query_executor(sq.query)
                sub_results.append({
                    "query": sq.query,
                    "answer": result,
                    "weight": sq.weight
                })
        
        elif decomposed.strategy == DecompositionStrategy.SEQUENTIAL:
            # Execute in sequence, passing context
            context = {}
            for sq in decomposed.sub_queries:
                result = query_executor(sq.query, context=context)
                sub_results.append({
                    "query": sq.query,
                    "answer": result,
                    "weight": sq.weight
                })
                # Update context with result
                context[sq.query_id] = result
        
        else:
            # Default execution
            for sq in decomposed.sub_queries:
                result = query_executor(sq.query)
                sub_results.append({
                    "query": sq.query,
                    "answer": result,
                    "weight": sq.weight
                })
        
        # Aggregate results
        if decomposed.requires_aggregation and decomposed.aggregation_prompt:
            aggregated = self._aggregate_results(
                decomposed.aggregation_prompt,
                sub_results
            )
            return {
                "answer": aggregated,
                "sub_results": sub_results,
                "strategy": decomposed.strategy.value
            }
        else:
            # Return all sub-results
            return {
                "answer": sub_results,
                "strategy": decomposed.strategy.value
            }
    
    def _aggregate_results(self, aggregation_prompt: str, sub_results: List[Dict[str, Any]]) -> str:
        """
        Aggregate sub-query results into final answer.
        
        Args:
            aggregation_prompt: Prompt for aggregation
            sub_results: Results from sub-queries
            
        Returns:
            Aggregated answer
        """
        # Format sub-results for aggregation
        results_text = "\n\n".join([
            f"Question: {r['query']}\nAnswer: {r['answer']}"
            for r in sub_results
        ])
        
        final_prompt = f"{aggregation_prompt}\n\nSub-answers:\n{results_text}"
        
        if self.ollama_client:
            try:
                response = self.ollama_client.generate(
                    model=self.decomposition_model,
                    prompt=final_prompt,
                    temperature=0.3
                )
                return response
            except Exception as e:
                logger.error(f"Aggregation failed: {e}")
        
        # Fallback: concatenate answers
        return "\n\n".join([r['answer'] for r in sub_results])