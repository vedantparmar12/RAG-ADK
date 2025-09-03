"""
Advanced Query Rewriting Module with HyDE and Query Expansion
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from ..config import settings

logger = logging.getLogger(__name__)

@dataclass
class RewrittenQuery:
    """Container for rewritten query results"""
    original_query: str
    hyde_hypothetical_document: str
    expanded_queries: List[str]
    sub_queries: List[str]
    query_embedding: np.ndarray
    metadata: Dict[str, Any]

class HyDEGenerator:
    """Hypothetical Document Embeddings (HyDE) generator"""
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the T5 model for document generation"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            logger.info(f"HyDE model initialized: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize HyDE model: {e}")
            raise
    
    def generate_hypothetical_document(self, query: str, domain: str = "general") -> str:
        """Generate a hypothetical document that would answer the query"""
        
        domain_prompts = {
            "general": f"Write a comprehensive document that answers this question: {query}",
            "technical": f"Write a detailed technical document explaining: {query}",
            "medical": f"Write a medical document that addresses: {query}",
            "legal": f"Write a legal document that explains: {query}",
            "academic": f"Write an academic paper section that covers: {query}"
        }
        
        prompt = domain_prompts.get(domain, domain_prompts["general"])
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=300,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.2
                )
            
            hypothetical_doc = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return hypothetical_doc
            
        except Exception as e:
            logger.error(f"Error generating hypothetical document: {e}")
            return query  # Fallback to original query

class QueryExpander:
    """Query expansion using semantic similarity and term variations"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.synonym_cache = {}
        
    @lru_cache(maxsize=1000)
    def expand_query(self, query: str, num_expansions: int = 3) -> List[str]:
        """Expand query with semantic variations"""
        
        expansions = []
        
        # 1. Synonym-based expansion
        expansions.extend(self._synonym_expansion(query))
        
        # 2. Paraphrase-based expansion
        expansions.extend(self._paraphrase_expansion(query))
        
        # 3. Context-aware expansion
        expansions.extend(self._context_expansion(query))
        
        # Remove duplicates and limit
        unique_expansions = list(dict.fromkeys([query] + expansions))
        return unique_expansions[:num_expansions + 1]
    
    def _synonym_expansion(self, query: str) -> List[str]:
        """Generate synonym-based query variations"""
        expansions = []
        
        # Simple keyword replacement patterns
        replacements = {
            r'\bhow to\b': 'method for',
            r'\bwhat is\b': 'definition of',
            r'\bwhy\b': 'reason for',
            r'\bwhen\b': 'time of',
            r'\bwhere\b': 'location of',
            r'\bwhich\b': 'type of'
        }
        
        for pattern, replacement in replacements.items():
            if re.search(pattern, query, re.IGNORECASE):
                expanded = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
                if expanded != query:
                    expansions.append(expanded)
        
        return expansions
    
    def _paraphrase_expansion(self, query: str) -> List[str]:
        """Generate paraphrased versions of the query"""
        expansions = []
        
        # Template-based paraphrasing
        if query.startswith("how"):
            expansions.append(f"what are the steps to {query[4:]}")
            expansions.append(f"guide for {query[4:]}")
        
        if "?" in query:
            statement = query.replace("?", "")
            expansions.append(f"information about {statement}")
            expansions.append(f"explain {statement}")
        
        return expansions
    
    def _context_expansion(self, query: str) -> List[str]:
        """Add contextual terms based on query analysis"""
        expansions = []
        
        # Domain-specific term additions
        tech_terms = ["implementation", "configuration", "setup", "troubleshooting"]
        business_terms = ["strategy", "process", "workflow", "best practices"]
        
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["code", "programming", "software", "api"]):
            for term in tech_terms:
                expansions.append(f"{query} {term}")
        
        if any(term in query_lower for term in ["business", "management", "strategy"]):
            for term in business_terms:
                expansions.append(f"{query} {term}")
        
        return expansions

class QueryDecomposer:
    """Decompose complex queries into sub-queries"""
    
    def __init__(self):
        self.conjunction_patterns = [
            r'\band\b', r'\bor\b', r'\bbut\b', r'\bhowever\b', 
            r'\balso\b', r'\bmoreover\b', r'\bfurthermore\b'
        ]
        
    def decompose_query(self, query: str) -> List[str]:
        """Break down complex queries into simpler sub-queries"""
        sub_queries = []
        
        # 1. Split on conjunctions
        sub_queries.extend(self._split_on_conjunctions(query))
        
        # 2. Extract question components
        sub_queries.extend(self._extract_question_components(query))
        
        # 3. Identify implicit sub-questions
        sub_queries.extend(self._identify_implicit_questions(query))
        
        # Clean and filter
        cleaned_queries = [q.strip() for q in sub_queries if len(q.strip()) > 10]
        
        # Remove duplicates while preserving order
        unique_queries = []
        seen = set()
        for q in cleaned_queries:
            if q.lower() not in seen:
                unique_queries.append(q)
                seen.add(q.lower())
        
        return unique_queries if unique_queries else [query]
    
    def _split_on_conjunctions(self, query: str) -> List[str]:
        """Split query on logical conjunctions"""
        parts = [query]
        
        for pattern in self.conjunction_patterns:
            new_parts = []
            for part in parts:
                splits = re.split(pattern, part, flags=re.IGNORECASE)
                new_parts.extend([s.strip() for s in splits])
            parts = new_parts
        
        return [p for p in parts if len(p) > 5]
    
    def _extract_question_components(self, query: str) -> List[str]:
        """Extract different question aspects"""
        components = []
        
        # Multiple question words in one query
        question_words = ["what", "how", "why", "when", "where", "which", "who"]
        found_questions = []
        
        for word in question_words:
            if word in query.lower():
                # Try to extract the question part
                pattern = rf'\b{word}\b[^.?!]*[.?!]?'
                matches = re.findall(pattern, query, re.IGNORECASE)
                found_questions.extend(matches)
        
        components.extend([q.strip() for q in found_questions if len(q.strip()) > 10])
        
        return components
    
    def _identify_implicit_questions(self, query: str) -> List[str]:
        """Identify implicit sub-questions within the query"""
        implicit = []
        
        # Look for comparative structures
        if "difference between" in query.lower():
            parts = query.lower().split("difference between")[-1]
            if " and " in parts:
                items = parts.split(" and ")
                for item in items:
                    implicit.append(f"What is {item.strip()}?")
        
        # Look for process descriptions
        if any(word in query.lower() for word in ["step", "process", "procedure"]):
            implicit.append(f"What are the steps involved in {query.lower().replace('steps', '').replace('process', '').strip()}?")
        
        return implicit

class AdvancedQueryRewriter:
    """Main query rewriting orchestrator"""
    
    def __init__(self):
        self.hyde_generator = HyDEGenerator()
        self.query_expander = QueryExpander()
        self.query_decomposer = QueryDecomposer()
        
        # Initialize embedding model for query embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    async def rewrite_query(
        self,
        query: str,
        enable_hyde: bool = True,
        enable_expansion: bool = True,
        enable_decomposition: bool = True,
        domain: str = "general",
        expansion_count: int = 3
    ) -> RewrittenQuery:
        """Main query rewriting method"""
        
        metadata = {
            "original_length": len(query),
            "enable_hyde": enable_hyde,
            "enable_expansion": enable_expansion,
            "enable_decomposition": enable_decomposition,
            "domain": domain
        }
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Generate hypothetical document if enabled
            hyde_doc = ""
            if enable_hyde:
                hyde_doc = await self._generate_hyde_async(query, domain)
                metadata["hyde_length"] = len(hyde_doc)
            
            # Expand query if enabled
            expanded_queries = [query]
            if enable_expansion:
                expanded_queries = self.query_expander.expand_query(query, expansion_count)
                metadata["expansion_count"] = len(expanded_queries)
            
            # Decompose query if enabled
            sub_queries = [query]
            if enable_decomposition:
                sub_queries = self.query_decomposer.decompose_query(query)
                metadata["sub_query_count"] = len(sub_queries)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            metadata["processing_time"] = asyncio.get_event_loop().time() - start_time
            
            return RewrittenQuery(
                original_query=query,
                hyde_hypothetical_document=hyde_doc,
                expanded_queries=expanded_queries,
                sub_queries=sub_queries,
                query_embedding=query_embedding,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in query rewriting: {e}")
            # Return minimal rewritten query on error
            return RewrittenQuery(
                original_query=query,
                hyde_hypothetical_document="",
                expanded_queries=[query],
                sub_queries=[query],
                query_embedding=self.embedding_model.encode(query),
                metadata={"error": str(e)}
            )
    
    async def _generate_hyde_async(self, query: str, domain: str) -> str:
        """Async wrapper for HyDE generation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.hyde_generator.generate_hypothetical_document, 
            query, 
            domain
        )
    
    def batch_rewrite_queries(
        self,
        queries: List[str],
        **kwargs
    ) -> List[RewrittenQuery]:
        """Batch process multiple queries"""
        return [
            asyncio.run(self.rewrite_query(query, **kwargs)) 
            for query in queries
        ]

# Global instance
query_rewriter = AdvancedQueryRewriter()

def get_query_rewriter() -> AdvancedQueryRewriter:
    """Get the global query rewriter instance"""
    return query_rewriter