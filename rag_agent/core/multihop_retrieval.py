"""
Multi-hop Retrieval System
Performs iterative, chained retrieval to answer complex questions
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class HopType(Enum):
    """Types of retrieval hops"""
    INITIAL = "initial"
    FOLLOWUP = "followup"  
    BRIDGE = "bridge"
    VERIFICATION = "verification"
    SYNTHESIS = "synthesis"

@dataclass
class RetrievalHop:
    """Single hop in multi-hop retrieval"""
    hop_id: str
    hop_number: int
    hop_type: HopType
    query: str
    retrieved_docs: List[Dict[str, Any]]
    reasoning: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievalPath:
    """Complete path through multi-hop retrieval"""
    path_id: str
    original_query: str
    hops: List[RetrievalHop]
    final_answer_docs: List[Dict[str, Any]]
    path_confidence: float
    reasoning_chain: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MultiHopResult:
    """Result of multi-hop retrieval"""
    original_query: str
    paths: List[RetrievalPath]
    best_path: RetrievalPath
    aggregated_documents: List[Dict[str, Any]]
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class QueryDecomposer:
    """Decompose complex queries into sub-questions for multi-hop"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        
        # Question patterns for decomposition
        self.complex_patterns = {
            'comparison': ['compare', 'versus', 'difference between', 'better than', 'contrast'],
            'causal': ['why', 'because', 'cause', 'reason', 'leads to', 'results in'],
            'temporal': ['when', 'before', 'after', 'during', 'timeline', 'history'],
            'compositional': ['made of', 'consists of', 'parts', 'components', 'structure'],
            'conditional': ['if', 'when', 'unless', 'provided that', 'in case'],
            'multi_entity': ['and', 'both', 'all', 'multiple', 'several']
        }
    
    def decompose_query(self, query: str) -> List[Dict[str, Any]]:
        """Decompose complex query into sub-questions"""
        
        query_lower = query.lower()
        sub_questions = []
        
        # Detect query type
        query_type = self._detect_query_type(query_lower)
        
        if query_type == 'comparison':
            sub_questions = self._decompose_comparison(query)
        elif query_type == 'causal':
            sub_questions = self._decompose_causal(query)
        elif query_type == 'temporal':
            sub_questions = self._decompose_temporal(query)
        elif query_type == 'compositional':
            sub_questions = self._decompose_compositional(query)
        elif query_type == 'conditional':
            sub_questions = self._decompose_conditional(query)
        elif query_type == 'multi_entity':
            sub_questions = self._decompose_multi_entity(query)
        else:
            # Default decomposition
            sub_questions = self._default_decomposition(query)
        
        return sub_questions
    
    def _detect_query_type(self, query: str) -> str:
        """Detect the type of complex query"""
        
        for query_type, patterns in self.complex_patterns.items():
            if any(pattern in query for pattern in patterns):
                return query_type
        
        return 'default'
    
    def _decompose_comparison(self, query: str) -> List[Dict[str, Any]]:
        """Decompose comparison queries"""
        sub_questions = []
        
        # Extract entities being compared
        comparison_words = ['versus', 'vs', 'compared to', 'difference between']
        entities = []
        
        for word in comparison_words:
            if word in query.lower():
                parts = query.lower().split(word)
                if len(parts) >= 2:
                    # Extract potential entities
                    left = parts[0].strip().split()[-3:]  # Last 3 words
                    right = parts[1].strip().split()[:3]   # First 3 words
                    entities.extend([' '.join(left), ' '.join(right)])
        
        # Create sub-questions
        if len(entities) >= 2:
            for entity in entities[:2]:  # Limit to 2 main entities
                sub_questions.append({
                    'query': f"What are the characteristics of {entity}?",
                    'type': 'definition',
                    'priority': 0.8,
                    'entity': entity
                })
            
            sub_questions.append({
                'query': f"How do {entities[0]} and {entities[1]} differ?",
                'type': 'comparison',
                'priority': 1.0,
                'entities': entities[:2]
            })
        
        return sub_questions
    
    def _decompose_causal(self, query: str) -> List[Dict[str, Any]]:
        """Decompose causal queries"""
        sub_questions = []
        
        # Extract cause and effect
        if 'why' in query.lower():
            effect = query.replace('why', '').replace('?', '').strip()
            sub_questions.append({
                'query': f"What causes {effect}?",
                'type': 'cause',
                'priority': 1.0,
                'effect': effect
            })
            
            sub_questions.append({
                'query': f"What are the mechanisms behind {effect}?",
                'type': 'mechanism',
                'priority': 0.7,
                'effect': effect
            })
        
        return sub_questions
    
    def _decompose_temporal(self, query: str) -> List[Dict[str, Any]]:
        """Decompose temporal queries"""
        sub_questions = []
        
        if 'when' in query.lower():
            event = query.replace('when', '').replace('?', '').strip()
            sub_questions.append({
                'query': f"What is the timeline of {event}?",
                'type': 'timeline',
                'priority': 1.0,
                'event': event
            })
            
            sub_questions.append({
                'query': f"What are the key dates related to {event}?",
                'type': 'dates',
                'priority': 0.8,
                'event': event
            })
        
        return sub_questions
    
    def _decompose_compositional(self, query: str) -> List[Dict[str, Any]]:
        """Decompose compositional queries"""
        sub_questions = []
        
        composition_words = ['made of', 'consists of', 'parts', 'components']
        for word in composition_words:
            if word in query.lower():
                entity = query.lower().replace(word, '').replace('what', '').replace('?', '').strip()
                sub_questions.append({
                    'query': f"What are the main components of {entity}?",
                    'type': 'components',
                    'priority': 1.0,
                    'entity': entity
                })
                
                sub_questions.append({
                    'query': f"How is {entity} structured?",
                    'type': 'structure',
                    'priority': 0.8,
                    'entity': entity
                })
                break
        
        return sub_questions
    
    def _decompose_conditional(self, query: str) -> List[Dict[str, Any]]:
        """Decompose conditional queries"""
        sub_questions = []
        
        if 'if' in query.lower():
            parts = query.lower().split('if')
            if len(parts) >= 2:
                condition = parts[1].strip()
                outcome = parts[0].strip()
                
                sub_questions.append({
                    'query': f"What happens when {condition}?",
                    'type': 'conditional_outcome',
                    'priority': 1.0,
                    'condition': condition,
                    'outcome': outcome
                })
                
                sub_questions.append({
                    'query': f"Under what conditions does {outcome} occur?",
                    'type': 'conditions',
                    'priority': 0.8,
                    'outcome': outcome
                })
        
        return sub_questions
    
    def _decompose_multi_entity(self, query: str) -> List[Dict[str, Any]]:
        """Decompose multi-entity queries"""
        sub_questions = []
        
        # Simple entity extraction based on 'and'
        if ' and ' in query.lower():
            parts = query.lower().split(' and ')
            entities = [part.strip() for part in parts]
            
            for entity in entities[:3]:  # Limit to 3 entities
                sub_questions.append({
                    'query': f"What is {entity}?",
                    'type': 'definition',
                    'priority': 0.7,
                    'entity': entity
                })
            
            # Add relationship question
            if len(entities) >= 2:
                sub_questions.append({
                    'query': f"How are {entities[0]} and {entities[1]} related?",
                    'type': 'relationship',
                    'priority': 0.9,
                    'entities': entities[:2]
                })
        
        return sub_questions
    
    def _default_decomposition(self, query: str) -> List[Dict[str, Any]]:
        """Default decomposition for simple queries"""
        return [{
            'query': query,
            'type': 'simple',
            'priority': 1.0,
            'original': True
        }]

class PathPlanner:
    """Plan retrieval paths for multi-hop queries"""
    
    def __init__(self):
        self.max_hops = 5
        self.min_confidence = 0.3
    
    def plan_retrieval_paths(
        self,
        original_query: str,
        sub_questions: List[Dict[str, Any]],
        max_paths: int = 3
    ) -> List[List[Dict[str, Any]]]:
        """Plan multiple retrieval paths"""
        
        paths = []
        
        # Sort sub-questions by priority
        sorted_questions = sorted(sub_questions, key=lambda x: x.get('priority', 0.5), reverse=True)
        
        # Create primary path (highest priority questions)
        primary_path = []
        for i, q in enumerate(sorted_questions[:self.max_hops]):
            primary_path.append({
                'hop_number': i,
                'query': q['query'],
                'type': q.get('type', 'default'),
                'priority': q.get('priority', 0.5),
                'reasoning': f"Step {i+1}: {q.get('type', 'default')} question"
            })
        
        paths.append(primary_path)
        
        # Create alternative paths by reordering or focusing on different aspects
        if len(sorted_questions) > 2:
            # Path 2: Reverse order (sometimes bottom-up works better)
            alt_path_1 = []
            reversed_questions = sorted_questions[:self.max_hops][::-1]
            for i, q in enumerate(reversed_questions):
                alt_path_1.append({
                    'hop_number': i,
                    'query': q['query'],
                    'type': q.get('type', 'default'),
                    'priority': q.get('priority', 0.5),
                    'reasoning': f"Alternative Step {i+1}: Bottom-up approach"
                })
            paths.append(alt_path_1)
            
            # Path 3: Focus on specific query types
            specific_types = ['definition', 'cause', 'comparison']
            for query_type in specific_types:
                type_questions = [q for q in sorted_questions if q.get('type') == query_type]
                if len(type_questions) >= 2:
                    type_path = []
                    for i, q in enumerate(type_questions[:self.max_hops]):
                        type_path.append({
                            'hop_number': i,
                            'query': q['query'],
                            'type': q.get('type', 'default'),
                            'priority': q.get('priority', 0.5),
                            'reasoning': f"Focused Step {i+1}: {query_type} focus"
                        })
                    paths.append(type_path)
                    break
        
        return paths[:max_paths]

class MultiHopRetriever:
    """Main multi-hop retrieval orchestrator"""
    
    def __init__(self, base_retriever=None):
        self.base_retriever = base_retriever
        self.query_decomposer = QueryDecomposer()
        self.path_planner = PathPlanner()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Retrieval configuration
        self.max_hops = 5
        self.hop_timeout = 30.0  # seconds
        self.confidence_threshold = 0.4
        
    async def retrieve_multihop(
        self,
        query: str,
        corpus_id: str,
        max_paths: int = 2,
        top_k_per_hop: int = 5
    ) -> MultiHopResult:
        """Execute multi-hop retrieval"""
        
        logger.info(f"Starting multi-hop retrieval for: {query}")
        
        # Step 1: Decompose query
        sub_questions = self.query_decomposer.decompose_query(query)
        
        if not sub_questions:
            # Fallback to single-hop
            return await self._single_hop_fallback(query, corpus_id, top_k_per_hop)
        
        # Step 2: Plan retrieval paths
        planned_paths = self.path_planner.plan_retrieval_paths(
            query, sub_questions, max_paths
        )
        
        # Step 3: Execute paths
        executed_paths = []
        
        for i, path_plan in enumerate(planned_paths):
            try:
                path = await self._execute_path(
                    f"path_{i}",
                    query,
                    path_plan,
                    corpus_id,
                    top_k_per_hop
                )
                executed_paths.append(path)
            except Exception as e:
                logger.error(f"Error executing path {i}: {e}")
                continue
        
        if not executed_paths:
            # Fallback to single-hop
            return await self._single_hop_fallback(query, corpus_id, top_k_per_hop)
        
        # Step 4: Select best path
        best_path = self._select_best_path(executed_paths)
        
        # Step 5: Aggregate results
        aggregated_docs = self._aggregate_documents(executed_paths)
        
        # Step 6: Calculate overall confidence and reasoning
        overall_confidence = self._calculate_overall_confidence(executed_paths)
        reasoning = self._generate_reasoning(best_path)
        
        return MultiHopResult(
            original_query=query,
            paths=executed_paths,
            best_path=best_path,
            aggregated_documents=aggregated_docs,
            confidence=overall_confidence,
            reasoning=reasoning,
            metadata={
                'num_paths': len(executed_paths),
                'total_hops': sum(len(path.hops) for path in executed_paths),
                'decomposed_questions': len(sub_questions)
            }
        )
    
    async def _execute_path(
        self,
        path_id: str,
        original_query: str,
        path_plan: List[Dict[str, Any]],
        corpus_id: str,
        top_k_per_hop: int
    ) -> RetrievalPath:
        """Execute a single retrieval path"""
        
        hops = []
        cumulative_context = []
        
        for hop_step in path_plan:
            hop_query = hop_step['query']
            hop_number = hop_step['hop_number']
            
            # Enhance query with context from previous hops
            if cumulative_context:
                context_summary = self._summarize_context(cumulative_context)
                enhanced_query = f"{hop_query} Context: {context_summary}"
            else:
                enhanced_query = hop_query
            
            # Execute retrieval for this hop
            try:
                retrieved_docs = await self._execute_single_hop(
                    enhanced_query,
                    corpus_id,
                    top_k_per_hop
                )
                
                # Calculate hop confidence
                hop_confidence = self._calculate_hop_confidence(retrieved_docs, hop_query)
                
                hop = RetrievalHop(
                    hop_id=f"{path_id}_hop_{hop_number}",
                    hop_number=hop_number,
                    hop_type=HopType.FOLLOWUP if hop_number > 0 else HopType.INITIAL,
                    query=enhanced_query,
                    retrieved_docs=retrieved_docs,
                    reasoning=hop_step.get('reasoning', ''),
                    confidence=hop_confidence,
                    metadata={
                        'original_query': hop_query,
                        'query_type': hop_step.get('type', 'default')
                    }
                )
                
                hops.append(hop)
                
                # Add to cumulative context
                if retrieved_docs:
                    cumulative_context.extend([doc.get('content', '')[:200] for doc in retrieved_docs[:3]])
                
                # Early stopping if confidence is too low
                if hop_confidence < self.confidence_threshold and hop_number > 0:
                    logger.info(f"Early stopping path {path_id} at hop {hop_number} due to low confidence")
                    break
                    
            except Exception as e:
                logger.error(f"Error in hop {hop_number} of path {path_id}: {e}")
                break
        
        # Determine final answer documents
        final_docs = []
        if hops:
            # Use documents from the last hop
            final_docs = hops[-1].retrieved_docs
        
        # Calculate path confidence
        path_confidence = np.mean([hop.confidence for hop in hops]) if hops else 0.0
        
        # Generate reasoning chain
        reasoning_chain = " -> ".join([hop.reasoning for hop in hops])
        
        return RetrievalPath(
            path_id=path_id,
            original_query=original_query,
            hops=hops,
            final_answer_docs=final_docs,
            path_confidence=float(path_confidence),
            reasoning_chain=reasoning_chain,
            metadata={
                'num_hops': len(hops),
                'total_docs': sum(len(hop.retrieved_docs) for hop in hops)
            }
        )
    
    async def _execute_single_hop(
        self,
        query: str,
        corpus_id: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Execute a single retrieval hop"""
        
        if self.base_retriever:
            # Use the provided base retriever
            return await self.base_retriever.retrieve(query, corpus_id, top_k)
        else:
            # Mock implementation for standalone usage
            return [
                {
                    'content': f"Mock document for query: {query}",
                    'score': 0.8,
                    'id': f"mock_doc_{hash(query) % 1000}",
                    'metadata': {'source': 'mock'}
                }
            ]
    
    def _summarize_context(self, context_pieces: List[str], max_length: int = 200) -> str:
        """Summarize cumulative context"""
        
        combined_context = " ".join(context_pieces)
        if len(combined_context) <= max_length:
            return combined_context
        
        # Simple truncation (could be improved with summarization model)
        return combined_context[:max_length] + "..."
    
    def _calculate_hop_confidence(self, docs: List[Dict[str, Any]], query: str) -> float:
        """Calculate confidence for a single hop"""
        
        if not docs:
            return 0.0
        
        # Average retrieval scores
        scores = [doc.get('score', 0.5) for doc in docs]
        avg_score = np.mean(scores)
        
        # Boost confidence if multiple high-quality results
        if len(docs) >= 3 and avg_score > 0.7:
            return min(avg_score * 1.1, 1.0)
        
        return float(avg_score)
    
    def _select_best_path(self, paths: List[RetrievalPath]) -> RetrievalPath:
        """Select the best retrieval path"""
        
        if not paths:
            raise ValueError("No paths to select from")
        
        # Score paths based on multiple criteria
        scored_paths = []
        
        for path in paths:
            score = 0.0
            
            # Path confidence (40%)
            score += path.path_confidence * 0.4
            
            # Number of successful hops (30%)
            successful_hops = sum(1 for hop in path.hops if hop.confidence > self.confidence_threshold)
            hop_score = successful_hops / max(len(path.hops), 1)
            score += hop_score * 0.3
            
            # Quality of final documents (30%)
            if path.final_answer_docs:
                doc_scores = [doc.get('score', 0.5) for doc in path.final_answer_docs]
                avg_doc_score = np.mean(doc_scores)
                score += avg_doc_score * 0.3
            
            scored_paths.append((path, score))
        
        # Return path with highest score
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        return scored_paths[0][0]
    
    def _aggregate_documents(self, paths: List[RetrievalPath]) -> List[Dict[str, Any]]:
        """Aggregate documents from all paths"""
        
        all_docs = []
        seen_ids = set()
        
        # Collect documents from all paths
        for path in paths:
            for hop in path.hops:
                for doc in hop.retrieved_docs:
                    doc_id = doc.get('id', f"doc_{hash(doc.get('content', ''))}")
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        # Add path information to document
                        enhanced_doc = doc.copy()
                        enhanced_doc['retrieval_paths'] = enhanced_doc.get('retrieval_paths', [])
                        enhanced_doc['retrieval_paths'].append({
                            'path_id': path.path_id,
                            'hop_number': hop.hop_number,
                            'hop_confidence': hop.confidence
                        })
                        all_docs.append(enhanced_doc)
        
        # Sort by relevance score
        all_docs.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        
        return all_docs
    
    def _calculate_overall_confidence(self, paths: List[RetrievalPath]) -> float:
        """Calculate overall confidence across all paths"""
        
        if not paths:
            return 0.0
        
        # Weight paths by their individual confidence
        weighted_confidences = []
        for path in paths:
            # Weight by path confidence and number of successful hops
            weight = path.path_confidence * len(path.hops)
            weighted_confidences.append(path.path_confidence * weight)
        
        if not weighted_confidences:
            return 0.0
        
        return float(np.mean(weighted_confidences))
    
    def _generate_reasoning(self, best_path: RetrievalPath) -> str:
        """Generate human-readable reasoning for the retrieval process"""
        
        reasoning = f"Multi-hop retrieval completed with {len(best_path.hops)} hops:\n"
        
        for i, hop in enumerate(best_path.hops):
            reasoning += f"{i+1}. {hop.reasoning} (confidence: {hop.confidence:.2f})\n"
        
        reasoning += f"\nFinal result confidence: {best_path.path_confidence:.2f}"
        
        return reasoning
    
    async def _single_hop_fallback(
        self,
        query: str,
        corpus_id: str,
        top_k: int
    ) -> MultiHopResult:
        """Fallback to single-hop retrieval"""
        
        logger.info("Falling back to single-hop retrieval")
        
        # Execute single retrieval
        docs = await self._execute_single_hop(query, corpus_id, top_k)
        
        # Create single hop
        hop = RetrievalHop(
            hop_id="fallback_hop_0",
            hop_number=0,
            hop_type=HopType.INITIAL,
            query=query,
            retrieved_docs=docs,
            reasoning="Single-hop fallback retrieval",
            confidence=self._calculate_hop_confidence(docs, query)
        )
        
        # Create single path
        path = RetrievalPath(
            path_id="fallback_path",
            original_query=query,
            hops=[hop],
            final_answer_docs=docs,
            path_confidence=hop.confidence,
            reasoning_chain="Single-hop fallback",
            metadata={'fallback': True}
        )
        
        return MultiHopResult(
            original_query=query,
            paths=[path],
            best_path=path,
            aggregated_documents=docs,
            confidence=hop.confidence,
            reasoning="Fallback to single-hop retrieval",
            metadata={'fallback': True}
        )

# Global instance
multihop_retriever = MultiHopRetriever()

def get_multihop_retriever() -> MultiHopRetriever:
    """Get the global multi-hop retriever instance"""
    return multihop_retriever