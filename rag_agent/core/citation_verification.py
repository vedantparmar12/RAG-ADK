"""
Advanced Citation and Verification Module
Provides proper source attribution and fact verification
"""

import logging
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

logger = logging.getLogger(__name__)

class CitationType(Enum):
    """Types of citations"""
    DIRECT_QUOTE = "direct_quote"
    PARAPHRASE = "paraphrase"
    INFERENCE = "inference"
    FACTUAL_CLAIM = "factual_claim"
    STATISTIC = "statistic"

class VerificationStatus(Enum):
    """Verification status levels"""
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"

@dataclass
class Citation:
    """Structured citation information"""
    id: str
    text: str
    source_id: str
    source_title: str
    source_excerpt: str
    page_number: Optional[int] = None
    confidence_score: float = 0.0
    citation_type: CitationType = CitationType.PARAPHRASE
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SourceDocument:
    """Source document for citation"""
    id: str
    title: str
    content: str
    author: Optional[str] = None
    publication_date: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VerificationResult:
    """Result of fact verification"""
    claim: str
    status: VerificationStatus
    confidence: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    sources: List[str]
    reasoning: str

@dataclass
class CitedResponse:
    """Response with citations and verification"""
    text: str
    citations: List[Citation]
    verification_results: List[VerificationResult]
    overall_credibility: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class CitationExtractor:
    """Extract and identify citable statements from text"""
    
    def __init__(self):
        # Initialize spaCy for NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except IOError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Patterns for identifying citation-worthy content
        self.citation_patterns = {
            CitationType.DIRECT_QUOTE: [
                r'"([^"]+)"',
                r"'([^']+)'",
                r'"([^"]+)"'
            ],
            CitationType.STATISTIC: [
                r'\b\d+(\.\d+)?%',
                r'\b\d{1,3}(,\d{3})*(\.\d+)?',
                r'\$\d{1,3}(,\d{3})*(\.\d{2})?',
                r'\b\d+(\.\d+)?\s*(million|billion|trillion|thousand)'
            ],
            CitationType.FACTUAL_CLAIM: [
                r'\b(studies show|research indicates|according to|based on)',
                r'\b(it was found that|results show|data reveals)',
                r'\b(experts believe|scientists say|researchers conclude)'
            ]
        }
    
    def extract_citable_statements(self, text: str) -> List[Dict[str, Any]]:
        """Extract statements that should be cited"""
        
        statements = []
        
        if self.nlp:
            doc = self.nlp(text)
            
            # Extract sentences
            for sent in doc.sents:
                sent_text = sent.text.strip()
                
                # Identify citation type
                citation_type = self._classify_statement(sent_text)
                
                # Extract entities and key phrases
                entities = [(ent.text, ent.label_) for ent in sent.ents]
                
                # Calculate importance score
                importance = self._calculate_importance(sent, entities)
                
                if importance > 0.3:  # Threshold for citation-worthy content
                    statements.append({
                        'text': sent_text,
                        'start_char': sent.start_char,
                        'end_char': sent.end_char,
                        'citation_type': citation_type,
                        'entities': entities,
                        'importance': importance
                    })
        else:
            # Fallback: simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            for i, sent in enumerate(sentences):
                sent = sent.strip()
                if len(sent) > 20:  # Minimum length
                    citation_type = self._classify_statement(sent)
                    statements.append({
                        'text': sent,
                        'start_char': i * 50,  # Rough estimate
                        'end_char': (i + 1) * 50,
                        'citation_type': citation_type,
                        'entities': [],
                        'importance': 0.5
                    })
        
        return statements
    
    def _classify_statement(self, text: str) -> CitationType:
        """Classify the type of statement for citation purposes"""
        
        text_lower = text.lower()
        
        # Check for direct quotes
        if any(re.search(pattern, text) for pattern in self.citation_patterns[CitationType.DIRECT_QUOTE]):
            return CitationType.DIRECT_QUOTE
        
        # Check for statistics
        if any(re.search(pattern, text) for pattern in self.citation_patterns[CitationType.STATISTIC]):
            return CitationType.STATISTIC
        
        # Check for factual claims
        if any(re.search(pattern, text_lower) for pattern in self.citation_patterns[CitationType.FACTUAL_CLAIM]):
            return CitationType.FACTUAL_CLAIM
        
        # Check for inference indicators
        inference_words = ['therefore', 'thus', 'hence', 'consequently', 'as a result']
        if any(word in text_lower for word in inference_words):
            return CitationType.INFERENCE
        
        return CitationType.PARAPHRASE
    
    def _calculate_importance(self, sentence_doc, entities: List[Tuple[str, str]]) -> float:
        """Calculate importance score for citation"""
        
        score = 0.0
        
        # Base score for all sentences
        score += 0.2
        
        # Boost for entities
        score += len(entities) * 0.1
        
        # Boost for specific entity types
        important_entities = ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY', 'PERCENT']
        for _, label in entities:
            if label in important_entities:
                score += 0.15
        
        # Boost for numbers and statistics
        if re.search(r'\d+', sentence_doc.text):
            score += 0.2
        
        # Boost for superlatives and strong claims
        strong_words = ['most', 'best', 'worst', 'largest', 'smallest', 'first', 'last', 'only', 'never', 'always']
        if any(word in sentence_doc.text.lower() for word in strong_words):
            score += 0.25
        
        return min(score, 1.0)

class SourceMatcher:
    """Match citations to source documents"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.sources: Dict[str, SourceDocument] = {}
        self.source_embeddings: Dict[str, np.ndarray] = {}
    
    def add_sources(self, sources: List[SourceDocument]):
        """Add source documents to the matcher"""
        
        logger.info(f"Adding {len(sources)} source documents")
        
        for source in sources:
            self.sources[source.id] = source
            
            # Create embeddings for source content
            # Split into chunks for better matching
            chunks = self._chunk_source(source.content)
            chunk_embeddings = self.embedding_model.encode(chunks)
            
            self.source_embeddings[source.id] = {
                'chunks': chunks,
                'embeddings': chunk_embeddings
            }
    
    def find_sources_for_statement(
        self,
        statement: str,
        top_k: int = 3,
        min_similarity: float = 0.6
    ) -> List[Tuple[str, float, str]]:
        """Find source documents that support a statement"""
        
        statement_embedding = self.embedding_model.encode([statement])
        
        matches = []
        
        for source_id, source_data in self.source_embeddings.items():
            # Calculate similarities with all chunks
            similarities = cosine_similarity(statement_embedding, source_data['embeddings'])[0]
            
            # Get best match
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            if best_similarity >= min_similarity:
                best_chunk = source_data['chunks'][best_idx]
                matches.append((source_id, float(best_similarity), best_chunk))
        
        # Sort by similarity
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches[:top_k]
    
    def _chunk_source(self, content: str, chunk_size: int = 300) -> List[str]:
        """Split source content into chunks for matching"""
        
        # Simple sentence-based chunking
        sentences = re.split(r'[.!?]+', content)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

class FactVerifier:
    """Verify factual claims against source documents"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.contradiction_threshold = 0.3
        self.support_threshold = 0.7
    
    def verify_claim(
        self,
        claim: str,
        sources: Dict[str, SourceDocument],
        source_embeddings: Dict[str, Any]
    ) -> VerificationResult:
        """Verify a factual claim against available sources"""
        
        claim_embedding = self.embedding_model.encode([claim])
        
        supporting_evidence = []
        contradicting_evidence = []
        all_sources = []
        
        for source_id, source_data in source_embeddings.items():
            similarities = cosine_similarity(claim_embedding, source_data['embeddings'])[0]
            
            for i, similarity in enumerate(similarities):
                chunk = source_data['chunks'][i]
                
                if similarity >= self.support_threshold:
                    supporting_evidence.append(f"{chunk} (Source: {source_id})")
                    all_sources.append(source_id)
                elif similarity <= self.contradiction_threshold:
                    # Check for contradiction patterns
                    if self._detect_contradiction(claim, chunk):
                        contradicting_evidence.append(f"{chunk} (Source: {source_id})")
                        all_sources.append(source_id)
        
        # Determine verification status
        status, confidence, reasoning = self._determine_verification_status(
            supporting_evidence, contradicting_evidence
        )
        
        return VerificationResult(
            claim=claim,
            status=status,
            confidence=confidence,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            sources=list(set(all_sources)),
            reasoning=reasoning
        )
    
    def _detect_contradiction(self, claim: str, evidence: str) -> bool:
        """Detect if evidence contradicts the claim"""
        
        # Simple negation detection
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        
        negation_words = ['not', 'no', 'never', 'none', 'neither', 'cannot', 'does not', 'did not']
        
        # Check for explicit negations
        for neg in negation_words:
            if neg in evidence_lower and neg not in claim_lower:
                return True
            if neg in claim_lower and neg not in evidence_lower:
                return True
        
        # Check for contrasting keywords
        contrasting_pairs = [
            ('increase', 'decrease'), ('rise', 'fall'), ('up', 'down'),
            ('positive', 'negative'), ('success', 'failure'), ('yes', 'no')
        ]
        
        for word1, word2 in contrasting_pairs:
            if (word1 in claim_lower and word2 in evidence_lower) or \
               (word2 in claim_lower and word1 in evidence_lower):
                return True
        
        return False
    
    def _determine_verification_status(
        self,
        supporting: List[str],
        contradicting: List[str]
    ) -> Tuple[VerificationStatus, float, str]:
        """Determine overall verification status"""
        
        support_count = len(supporting)
        contradict_count = len(contradicting)
        
        if contradict_count > 0 and contradict_count >= support_count:
            return VerificationStatus.CONTRADICTED, 0.8, "Found contradicting evidence"
        elif support_count >= 2:
            return VerificationStatus.VERIFIED, 0.9, f"Supported by {support_count} sources"
        elif support_count == 1:
            return VerificationStatus.PARTIALLY_VERIFIED, 0.6, "Limited supporting evidence"
        else:
            return VerificationStatus.INSUFFICIENT_EVIDENCE, 0.2, "No supporting evidence found"

class CitationGenerator:
    """Generate proper citations and attributed responses"""
    
    def __init__(self):
        self.citation_extractor = CitationExtractor()
        self.source_matcher = SourceMatcher()
        self.fact_verifier = FactVerifier()
    
    def add_sources(self, sources: List[SourceDocument]):
        """Add source documents for citation"""
        self.source_matcher.add_sources(sources)
    
    async def generate_cited_response(
        self,
        response_text: str,
        require_verification: bool = True
    ) -> CitedResponse:
        """Generate a response with proper citations and verification"""
        
        # Extract citable statements
        statements = self.citation_extractor.extract_citable_statements(response_text)
        
        citations = []
        verification_results = []
        
        for i, statement in enumerate(statements):
            # Find supporting sources
            source_matches = self.source_matcher.find_sources_for_statement(
                statement['text']
            )
            
            if source_matches:
                # Create citation
                best_source_id, confidence, excerpt = source_matches[0]
                source = self.source_matcher.sources[best_source_id]
                
                citation = Citation(
                    id=f"cite_{i:03d}",
                    text=statement['text'],
                    source_id=best_source_id,
                    source_title=source.title,
                    source_excerpt=excerpt,
                    confidence_score=confidence,
                    citation_type=statement['citation_type'],
                    metadata={
                        'importance': statement['importance'],
                        'entities': statement['entities']
                    }
                )
                
                # Verify if required
                if require_verification:
                    verification = self.fact_verifier.verify_claim(
                        statement['text'],
                        self.source_matcher.sources,
                        self.source_matcher.source_embeddings
                    )
                    verification_results.append(verification)
                    citation.verification_status = verification.status
                
                citations.append(citation)
        
        # Calculate overall credibility
        overall_credibility = self._calculate_credibility(citations, verification_results)
        
        # Format response with citations
        formatted_text = self._format_response_with_citations(response_text, citations)
        
        return CitedResponse(
            text=formatted_text,
            citations=citations,
            verification_results=verification_results,
            overall_credibility=overall_credibility,
            metadata={
                'citation_count': len(citations),
                'verification_count': len(verification_results),
                'generated_at': datetime.utcnow().isoformat()
            }
        )
    
    def _calculate_credibility(
        self,
        citations: List[Citation],
        verifications: List[VerificationResult]
    ) -> float:
        """Calculate overall credibility score"""
        
        if not citations:
            return 0.0
        
        citation_scores = [c.confidence_score for c in citations]
        avg_citation_confidence = np.mean(citation_scores)
        
        if verifications:
            verification_weights = {
                VerificationStatus.VERIFIED: 1.0,
                VerificationStatus.PARTIALLY_VERIFIED: 0.6,
                VerificationStatus.UNVERIFIED: 0.3,
                VerificationStatus.CONTRADICTED: 0.0,
                VerificationStatus.INSUFFICIENT_EVIDENCE: 0.2
            }
            
            verification_scores = [
                verification_weights[v.status] * v.confidence
                for v in verifications
            ]
            avg_verification_score = np.mean(verification_scores)
            
            # Weighted combination
            credibility = 0.4 * avg_citation_confidence + 0.6 * avg_verification_score
        else:
            credibility = avg_citation_confidence
        
        return float(credibility)
    
    def _format_response_with_citations(
        self,
        text: str,
        citations: List[Citation]
    ) -> str:
        """Format response text with citation markers"""
        
        # Sort citations by position in text
        citations_by_pos = []
        for citation in citations:
            pos = text.find(citation.text)
            if pos != -1:
                citations_by_pos.append((pos, citation))
        
        citations_by_pos.sort(key=lambda x: x[0])
        
        # Insert citation markers
        formatted_text = text
        offset = 0
        
        for i, (pos, citation) in enumerate(citations_by_pos):
            citation_marker = f"[{i+1}]"
            insert_pos = pos + len(citation.text) + offset
            formatted_text = formatted_text[:insert_pos] + citation_marker + formatted_text[insert_pos:]
            offset += len(citation_marker)
        
        # Add citation list at the end
        if citations:
            formatted_text += "\n\n**Citations:**\n"
            for i, citation in enumerate(citations):
                formatted_text += f"[{i+1}] {citation.source_title} - \"{citation.source_excerpt[:100]}...\"\n"
        
        return formatted_text

# Global instance
citation_generator = CitationGenerator()

def get_citation_generator() -> CitationGenerator:
    """Get the global citation generator instance"""
    return citation_generator