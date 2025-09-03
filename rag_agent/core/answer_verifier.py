"""
Answer verification system for validating and improving RAG responses.
Optimized for Windows CPU operation.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class VerificationStatus(Enum):
    """Status of answer verification"""
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    CONTRADICTORY = "contradictory"
    INSUFFICIENT_CONTEXT = "insufficient_context"

class FactCheckResult(Enum):
    """Result of fact checking"""
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"

@dataclass
class VerificationResult:
    """Container for verification results"""
    status: VerificationStatus
    confidence: float
    fact_checks: List[Dict[str, Any]]
    issues: List[str]
    suggestions: List[str]
    corrected_answer: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class FactClaim:
    """Container for a factual claim to verify"""
    claim: str
    source: str
    confidence: float
    evidence: List[str]
    check_result: Optional[FactCheckResult] = None

class AnswerVerifier:
    """
    Verifies and validates answers from RAG system.
    CPU-optimized for Windows.
    """
    
    def __init__(self,
                 ollama_client: Optional[Any] = None,
                 verification_model: str = "llama3.2",
                 use_llm_verification: bool = False,
                 confidence_threshold: float = 0.7):
        """
        Initialize answer verifier.
        
        Args:
            ollama_client: Optional Ollama client for LLM-based verification
            verification_model: Model to use for verification
            use_llm_verification: Whether to use LLM (CPU-intensive)
            confidence_threshold: Minimum confidence for verification
        """
        self.ollama_client = ollama_client
        self.verification_model = verification_model
        self.use_llm_verification = use_llm_verification
        self.confidence_threshold = confidence_threshold
        
        # Verification patterns for CPU-efficient checking
        self.verification_patterns = {
            "contradictions": [
                (r"(\d+).*but.*(\d+)", "numerical_mismatch"),
                (r"not.*but actually", "direct_contradiction"),
                (r"however.*contrary", "contradiction_marker"),
                (r"(\w+) is (\w+).*\1 is not \2", "property_contradiction")
            ],
            "uncertainty_markers": [
                r"might|may|possibly|probably|perhaps",
                r"I think|I believe|it seems",
                r"approximately|roughly|about",
                r"unclear|uncertain|unsure"
            ],
            "citation_patterns": [
                r"according to|based on|as per",
                r"source:|reference:",
                r"\[[\d,\s]+\]",  # [1, 2, 3] style citations
                r"\(\w+,?\s*\d{4}\)"  # (Author, Year) style
            ],
            "factual_claims": [
                r"(\d+(?:\.\d+)?)\s*(?:percent|%|dollars?|\$|euros?|€)",  # Numbers with units
                r"in (\d{4})",  # Years
                r"(\w+) (?:is|was|are|were) (?:the|a|an)? (\w+)",  # Definitions
                r"(?:first|last|only|largest|smallest|best|worst) (\w+)"  # Superlatives
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for category, patterns in self.verification_patterns.items():
            if isinstance(patterns[0], tuple):
                self.compiled_patterns[category] = [
                    (re.compile(p, re.IGNORECASE), label) 
                    for p, label in patterns
                ]
            else:
                self.compiled_patterns[category] = [
                    re.compile(p, re.IGNORECASE) for p in patterns
                ]
        
        logger.info(f"Answer verifier initialized (LLM: {use_llm_verification})")
    
    def verify_answer(self,
                      answer: str,
                      context: List[str],
                      query: str,
                      metadata: Optional[Dict[str, Any]] = None) -> VerificationResult:
        """
        Verify an answer against context.
        
        Args:
            answer: Generated answer to verify
            context: Retrieved context used for answer
            query: Original query
            metadata: Optional metadata
            
        Returns:
            VerificationResult
        """
        # Extract factual claims from answer
        claims = self._extract_claims(answer)
        
        # Verify using appropriate method
        if self.use_llm_verification and self.ollama_client:
            return self._llm_verification(answer, context, query, claims, metadata)
        else:
            return self._pattern_verification(answer, context, query, claims, metadata)
    
    def _extract_claims(self, answer: str) -> List[FactClaim]:
        """
        Extract factual claims from answer.
        
        Args:
            answer: Answer text
            
        Returns:
            List of factual claims
        """
        claims = []
        
        # Split answer into sentences
        sentences = re.split(r'[.!?]+', answer)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check for factual patterns
            for pattern in self.compiled_patterns["factual_claims"]:
                if pattern.search(sentence):
                    claim = FactClaim(
                        claim=sentence,
                        source="answer",
                        confidence=0.5,  # Initial confidence
                        evidence=[]
                    )
                    claims.append(claim)
                    break
        
        return claims
    
    def _pattern_verification(self,
                            answer: str,
                            context: List[str],
                            query: str,
                            claims: List[FactClaim],
                            metadata: Optional[Dict[str, Any]]) -> VerificationResult:
        """
        Pattern-based verification for CPU efficiency.
        
        Args:
            answer: Answer to verify
            context: Context documents
            query: Original query
            claims: Extracted claims
            metadata: Optional metadata
            
        Returns:
            VerificationResult
        """
        issues = []
        suggestions = []
        fact_checks = []
        
        # Check for contradictions
        contradictions = self._check_contradictions(answer)
        if contradictions:
            issues.extend(contradictions)
            suggestions.append("Review answer for internal contradictions")
        
        # Check for uncertainty
        uncertainty_score = self._check_uncertainty(answer)
        if uncertainty_score > 0.3:
            issues.append(f"High uncertainty detected (score: {uncertainty_score:.2f})")
            suggestions.append("Consider providing more definitive information")
        
        # Check for citations
        has_citations = self._check_citations(answer)
        if not has_citations and len(claims) > 0:
            issues.append("No citations found for factual claims")
            suggestions.append("Add source references for factual statements")
        
        # Verify claims against context
        for claim in claims:
            check_result = self._verify_claim_against_context(claim, context)
            claim.check_result = check_result
            
            fact_checks.append({
                "claim": claim.claim,
                "result": check_result.value,
                "confidence": claim.confidence,
                "evidence": claim.evidence
            })
            
            if check_result == FactCheckResult.CONTRADICTED:
                issues.append(f"Claim contradicted by context: {claim.claim[:50]}...")
            elif check_result == FactCheckResult.UNSUPPORTED:
                issues.append(f"Claim not supported by context: {claim.claim[:50]}...")
        
        # Calculate overall confidence
        if fact_checks:
            supported_count = sum(1 for fc in fact_checks if fc["result"] == "supported")
            confidence = supported_count / len(fact_checks)
        else:
            confidence = 0.5 if not issues else 0.3
        
        # Determine status
        if not issues:
            status = VerificationStatus.VERIFIED
        elif confidence >= self.confidence_threshold:
            status = VerificationStatus.PARTIALLY_VERIFIED
        elif any("contradicted" in issue.lower() for issue in issues):
            status = VerificationStatus.CONTRADICTORY
        elif not context:
            status = VerificationStatus.INSUFFICIENT_CONTEXT
        else:
            status = VerificationStatus.UNVERIFIED
        
        # Generate corrected answer if needed
        corrected_answer = None
        if issues and status in [VerificationStatus.CONTRADICTORY, VerificationStatus.UNVERIFIED]:
            corrected_answer = self._generate_corrected_answer(answer, issues, context)
        
        return VerificationResult(
            status=status,
            confidence=confidence,
            fact_checks=fact_checks,
            issues=issues,
            suggestions=suggestions,
            corrected_answer=corrected_answer,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "verification_method": "pattern",
                "uncertainty_score": uncertainty_score,
                "has_citations": has_citations,
                **(metadata or {})
            }
        )
    
    def _llm_verification(self,
                         answer: str,
                         context: List[str],
                         query: str,
                         claims: List[FactClaim],
                         metadata: Optional[Dict[str, Any]]) -> VerificationResult:
        """
        LLM-based verification for better accuracy (CPU-intensive).
        
        Args:
            answer: Answer to verify
            context: Context documents
            query: Original query
            claims: Extracted claims
            metadata: Optional metadata
            
        Returns:
            VerificationResult
        """
        context_text = "\n\n".join(context[:3])  # Limit context for CPU
        
        prompt = f"""Verify this answer against the provided context.

Query: "{query}"

Answer: "{answer}"

Context:
{context_text}

Tasks:
1. Check if all factual claims in the answer are supported by the context
2. Identify any contradictions or inaccuracies
3. Rate confidence (0.0-1.0)
4. Suggest improvements if needed

Respond in JSON format:
{{
    "status": "verified|partially_verified|unverified|contradictory",
    "confidence": 0.0-1.0,
    "fact_checks": [
        {{"claim": "...", "result": "supported|unsupported|contradicted", "evidence": "..."}}
    ],
    "issues": ["..."],
    "suggestions": ["..."],
    "corrected_answer": "..." or null
}}"""

        try:
            response = self.ollama_client.generate(
                model=self.verification_model,
                prompt=prompt,
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse JSON response
            result = json.loads(response)
            
            return VerificationResult(
                status=VerificationStatus[result["status"].upper()],
                confidence=result["confidence"],
                fact_checks=result["fact_checks"],
                issues=result["issues"],
                suggestions=result["suggestions"],
                corrected_answer=result.get("corrected_answer"),
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "verification_method": "llm",
                    **(metadata or {})
                }
            )
            
        except Exception as e:
            logger.warning(f"LLM verification failed, using pattern-based: {e}")
            return self._pattern_verification(answer, context, query, claims, metadata)
    
    def _check_contradictions(self, text: str) -> List[str]:
        """
        Check for internal contradictions.
        
        Args:
            text: Text to check
            
        Returns:
            List of contradiction issues
        """
        issues = []
        
        for pattern, label in self.compiled_patterns["contradictions"]:
            matches = pattern.findall(text)
            if matches:
                issues.append(f"Potential {label}: {matches[0] if matches else 'detected'}")
        
        return issues
    
    def _check_uncertainty(self, text: str) -> float:
        """
        Calculate uncertainty score.
        
        Args:
            text: Text to check
            
        Returns:
            Uncertainty score (0.0-1.0)
        """
        word_count = len(text.split())
        if word_count == 0:
            return 1.0
        
        uncertainty_count = 0
        for pattern in self.compiled_patterns["uncertainty_markers"]:
            uncertainty_count += len(pattern.findall(text))
        
        return min(uncertainty_count / word_count, 1.0)
    
    def _check_citations(self, text: str) -> bool:
        """
        Check if text has citations.
        
        Args:
            text: Text to check
            
        Returns:
            True if citations found
        """
        for pattern in self.compiled_patterns["citation_patterns"]:
            if pattern.search(text):
                return True
        return False
    
    def _verify_claim_against_context(self, 
                                     claim: FactClaim,
                                     context: List[str]) -> FactCheckResult:
        """
        Verify a claim against context.
        
        Args:
            claim: Claim to verify
            context: Context documents
            
        Returns:
            FactCheckResult
        """
        if not context:
            return FactCheckResult.UNKNOWN
        
        claim_lower = claim.claim.lower()
        
        # Check each context document
        for ctx in context:
            ctx_lower = ctx.lower()
            
            # Simple keyword matching for CPU efficiency
            claim_words = set(claim_lower.split())
            ctx_words = set(ctx_lower.split())
            
            # Calculate overlap
            overlap = len(claim_words & ctx_words) / len(claim_words) if claim_words else 0
            
            if overlap > 0.7:
                # High overlap - likely supported
                claim.evidence.append(ctx[:100] + "...")
                return FactCheckResult.SUPPORTED
            elif overlap > 0.3:
                # Partial overlap - check for negation
                if any(neg in ctx_lower for neg in ["not", "no", "false", "incorrect"]):
                    if overlap > 0.5:
                        return FactCheckResult.CONTRADICTED
            
        # No strong evidence found
        return FactCheckResult.UNSUPPORTED
    
    def _generate_corrected_answer(self,
                                  answer: str,
                                  issues: List[str],
                                  context: List[str]) -> Optional[str]:
        """
        Generate corrected answer addressing issues.
        
        Args:
            answer: Original answer
            issues: List of issues found
            context: Context documents
            
        Returns:
            Corrected answer or None
        """
        if not self.ollama_client:
            return None
        
        issues_text = "\n".join(f"- {issue}" for issue in issues)
        context_text = "\n\n".join(context[:2])  # Limit for CPU
        
        prompt = f"""Correct this answer to address the identified issues.

Original Answer: "{answer}"

Issues Found:
{issues_text}

Context:
{context_text}

Provide a corrected answer that:
1. Addresses all identified issues
2. Maintains accuracy based on the context
3. Includes appropriate citations
4. Avoids uncertainty where possible

Corrected Answer:"""

        try:
            corrected = self.ollama_client.generate(
                model=self.verification_model,
                prompt=prompt,
                temperature=0.3,
                max_tokens=300
            )
            return corrected
        except Exception as e:
            logger.error(f"Failed to generate corrected answer: {e}")
            return None
    
    def batch_verify(self,
                    answers: List[Tuple[str, List[str], str]],
                    batch_size: int = 1) -> List[VerificationResult]:
        """
        Verify multiple answers (CPU-optimized with small batches).
        
        Args:
            answers: List of (answer, context, query) tuples
            batch_size: Batch size (1 for CPU)
            
        Returns:
            List of VerificationResults
        """
        results = []
        
        for answer, context, query in answers:
            result = self.verify_answer(answer, context, query)
            results.append(result)
        
        return results