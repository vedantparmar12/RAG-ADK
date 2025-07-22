"""
Intelligent context pruning to optimize token usage and improve response quality.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
import logging
import re
import contextlib

from ..config import settings

logger = logging.getLogger(__name__)


class ContextPruner:
    """Implements intelligent context pruning"""
    
    def __init__(self, settings_obj: Optional[Any] = None):
        self.settings = settings_obj or settings
        self.setup_models()
        
    def setup_models(self):
        """Initialize pruning models"""
        if self.settings.enable_context_pruning and TRANSFORMERS_AVAILABLE:
            try:
                # Use a general-purpose model for relevance scoring
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
                self.pruning_model = AutoModelForSequenceClassification.from_pretrained(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.pruning_model.to(self.device)
                logger.info("Context pruning model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load pruning model: {e}")
                self.settings.enable_context_pruning = False
        elif self.settings.enable_context_pruning:
            logger.warning("Transformers not available, disabling context pruning")
            self.settings.enable_context_pruning = False
                
    def prune_context(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        max_tokens: int = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Prune context to most relevant information"""
        
        if max_tokens is None:
            max_tokens = self.settings.max_context_tokens
            
        if not self.settings.enable_context_pruning:
            # Return all chunks if pruning is disabled
            return chunks[:10], {"pruning_enabled": False}
            
        # Calculate relevance scores
        relevance_scores = self._calculate_relevance_scores(
            query,
            chunks
        )
        
        # Apply attention-based pruning if enabled
        if self.settings.enable_attention_pruning:
            attention_weights = self._calculate_attention_weights(
                query,
                chunks
            )
            relevance_scores = self._combine_scores(
                relevance_scores,
                attention_weights
            )
            
        # Select chunks within token limit
        selected_chunks = []
        current_tokens = 0
        pruning_stats = {
            "original_chunks": len(chunks),
            "original_tokens": sum(self._estimate_tokens(c.get("text", "")) for c in chunks),
            "pruned_chunks": 0,
            "pruned_tokens": 0,
            "compression_ratio": 0.0
        }
        
        # Sort by relevance
        scored_chunks = list(zip(chunks, relevance_scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        for chunk, score in scored_chunks:
            chunk_text = chunk.get("text", "")
            chunk_tokens = self._estimate_tokens(chunk_text)
            
            if current_tokens + chunk_tokens <= max_tokens:
                # Add chunk
                selected_chunks.append({
                    **chunk,
                    "relevance_score": float(score)
                })
                current_tokens += chunk_tokens
            else:
                # Try sentence-level pruning
                if self.settings.enable_sentence_pruning:
                    pruned_chunk = self._prune_chunk_sentences(
                        query,
                        chunk,
                        max_tokens - current_tokens
                    )
                    if pruned_chunk:
                        selected_chunks.append(pruned_chunk)
                        current_tokens += pruned_chunk["token_count"]
                        
        # Calculate stats
        pruning_stats["pruned_chunks"] = len(selected_chunks)
        pruning_stats["pruned_tokens"] = current_tokens
        pruning_stats["compression_ratio"] = 1 - (
            current_tokens / pruning_stats["original_tokens"]
        ) if pruning_stats["original_tokens"] > 0 else 0
        
        return selected_chunks, pruning_stats
    
    def _calculate_relevance_scores(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Calculate relevance scores for chunks"""
        
        if not self.settings.enable_context_pruning:
            # Return uniform scores if model not loaded
            return np.ones(len(chunks))
            
        scores = []
        
        with torch.no_grad() if TRANSFORMERS_AVAILABLE else contextlib.nullcontext():
            for chunk in chunks:
                chunk_text = chunk.get("text", "")
                
                # Prepare input
                inputs = self.tokenizer(
                    query,
                    chunk_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                # Get relevance score
                outputs = self.pruning_model(**inputs)
                score = torch.sigmoid(outputs.logits[0]).cpu().item()
                scores.append(score)
                
        return np.array(scores)
    
    def _calculate_attention_weights(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Calculate attention-based weights for chunks"""
        
        # Simple implementation based on query term overlap
        query_terms = set(query.lower().split())
        weights = []
        
        for chunk in chunks:
            chunk_text = chunk.get("text", "").lower()
            chunk_terms = set(chunk_text.split())
            
            # Calculate overlap
            overlap = len(query_terms.intersection(chunk_terms))
            weight = overlap / len(query_terms) if query_terms else 0
            weights.append(weight)
            
        return np.array(weights)
    
    def _combine_scores(
        self,
        relevance_scores: np.ndarray,
        attention_weights: np.ndarray,
        alpha: float = 0.7
    ) -> np.ndarray:
        """Combine relevance scores with attention weights"""
        
        # Normalize weights
        if attention_weights.max() > 0:
            attention_weights = attention_weights / attention_weights.max()
            
        # Combine with weighted average
        combined = alpha * relevance_scores + (1 - alpha) * attention_weights
        
        return combined
    
    def _prune_chunk_sentences(
        self,
        query: str,
        chunk: Dict[str, Any],
        max_tokens: int
    ) -> Optional[Dict[str, Any]]:
        """Prune chunk at sentence level"""
        
        chunk_text = chunk.get("text", "")
        sentences = self._split_sentences(chunk_text)
        
        if not sentences:
            return None
            
        sentence_scores = []
        
        # Score each sentence
        for sentence in sentences:
            score = self._score_sentence(query, sentence)
            sentence_scores.append(score)
            
        # Select top sentences within limit
        scored_sentences = list(zip(sentences, sentence_scores))
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        selected_sentences = []
        current_tokens = 0
        
        for sentence, score in scored_sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            if current_tokens + sentence_tokens <= max_tokens:
                selected_sentences.append(sentence)
                current_tokens += sentence_tokens
                
        if not selected_sentences:
            return None
            
        # Create pruned chunk
        pruned_text = " ".join(selected_sentences)
        
        return {
            **chunk,
            "text": pruned_text,
            "token_count": current_tokens,
            "pruned": True,
            "original_text": chunk_text
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _score_sentence(self, query: str, sentence: str) -> float:
        """Score individual sentence relevance"""
        
        # Simple scoring based on term overlap
        query_terms = set(query.lower().split())
        sentence_terms = set(sentence.lower().split())
        
        if not query_terms:
            return 0.0
            
        overlap = len(query_terms.intersection(sentence_terms))
        score = overlap / len(query_terms)
        
        return score
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Rough estimation: 1 token per 4 characters
        return len(text) // 4