"""
LangExtract integration for enhanced metadata extraction during chunking.
Optimized for both cloud (Vertex AI) and local (Ollama) RAG systems.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

# Check if LangExtract is available
try:
    import langextract as lx
    from langextract.data import Extraction, ExampleData
    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False
    logger.warning("LangExtract not installed. Install with: pip install langextract")

@dataclass
class EnhancedChunk:
    """Enhanced chunk with LangExtract metadata"""
    text: str
    chunk_id: str
    position: int
    metadata: Dict[str, Any]
    extracted_entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    topics: List[str]
    summary: Optional[str] = None
    confidence_score: float = 0.0
    source_spans: List[Tuple[int, int]] = field(default_factory=list)

class LangExtractEnhancer:
    """
    Enhances document chunks with structured metadata using LangExtract.
    Supports both cloud (Gemini) and local (Ollama) models.
    """
    
    def __init__(self,
                 model_provider: str = "local",  # "gemini", "openai", "local"
                 model_id: Optional[str] = None,
                 extraction_mode: str = "comprehensive",  # "fast", "balanced", "comprehensive"
                 use_cache: bool = True,
                 cache_dir: Optional[str] = None):
        """
        Initialize LangExtract enhancer.
        
        Args:
            model_provider: LLM provider to use
            model_id: Specific model ID
            extraction_mode: Extraction detail level
            use_cache: Whether to cache extractions
            cache_dir: Directory for caching
        """
        if not LANGEXTRACT_AVAILABLE:
            raise ImportError("LangExtract is required. Install with: pip install langextract")
        
        self.model_provider = model_provider
        self.extraction_mode = extraction_mode
        self.use_cache = use_cache
        
        # Set default model based on provider
        if model_id:
            self.model_id = model_id
        else:
            self.model_id = self._get_default_model()
        
        # Setup cache
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".langextract_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure extraction templates based on mode
        self.extraction_configs = self._setup_extraction_configs()
        
        logger.info(f"LangExtract enhancer initialized with {model_provider} provider")
    
    def _get_default_model(self) -> str:
        """Get default model for provider"""
        models = {
            "gemini": "gemini-1.5-flash",
            "openai": "gpt-4o-mini",
            "local": "llama3.2"  # For Ollama
        }
        return models.get(self.model_provider, "gemini-1.5-flash")
    
    def _setup_extraction_configs(self) -> Dict[str, Dict[str, Any]]:
        """Setup extraction configurations for different modes"""
        configs = {
            "fast": {
                "prompt": "Extract key entities and topics from this text.",
                "max_tokens": 500,
                "extraction_passes": 1,
                "max_char_buffer": 1000,
                "extract_entities": True,
                "extract_relationships": False,
                "extract_topics": True,
                "generate_summary": False
            },
            "balanced": {
                "prompt": """Extract the following from this text:
                1. Key entities (people, places, organizations, concepts)
                2. Main topics and themes
                3. Important relationships between entities""",
                "max_tokens": 1000,
                "extraction_passes": 2,
                "max_char_buffer": 2000,
                "extract_entities": True,
                "extract_relationships": True,
                "extract_topics": True,
                "generate_summary": False
            },
            "comprehensive": {
                "prompt": """Extract comprehensive information from this text:
                1. All entities with their types and attributes
                2. Relationships between entities with descriptions
                3. Main topics, themes, and subtopics
                4. Key facts and claims
                5. Temporal information (dates, times, durations)
                6. Quantitative data (numbers, amounts, percentages)
                7. Brief summary of the content""",
                "max_tokens": 2000,
                "extraction_passes": 3,
                "max_char_buffer": 3000,
                "extract_entities": True,
                "extract_relationships": True,
                "extract_topics": True,
                "generate_summary": True
            }
        }
        return configs
    
    def enhance_chunk(self, 
                      chunk_text: str,
                      chunk_id: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None) -> EnhancedChunk:
        """
        Enhance a single chunk with LangExtract metadata.
        
        Args:
            chunk_text: Text to enhance
            chunk_id: Optional chunk identifier
            context: Optional context information
            
        Returns:
            EnhancedChunk with extracted metadata
        """
        # Generate chunk ID if not provided
        if not chunk_id:
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
        
        # Check cache
        if self.use_cache:
            cached = self._get_cached_extraction(chunk_id)
            if cached:
                return cached
        
        # Get extraction config
        config = self.extraction_configs[self.extraction_mode]
        
        # Prepare extraction examples for better results
        examples = self._prepare_examples(context)
        
        try:
            # Perform extraction using LangExtract
            result = lx.extract(
                text_or_documents=chunk_text,
                prompt_description=config["prompt"],
                examples=examples,
                model_id=self.model_id,
                max_tokens=config["max_tokens"],
                extraction_passes=config["extraction_passes"],
                max_char_buffer=config["max_char_buffer"]
            )
            
            # Process extraction results
            entities = []
            relationships = []
            topics = []
            source_spans = []
            
            if result and hasattr(result, 'extractions'):
                for extraction in result.extractions:
                    # Extract entities
                    if config["extract_entities"] and extraction.entity_type:
                        entities.append({
                            "text": extraction.text,
                            "type": extraction.entity_type,
                            "attributes": extraction.attributes or {},
                            "confidence": getattr(extraction, 'confidence', 0.8),
                            "span": (extraction.start_char, extraction.end_char)
                        })
                        source_spans.append((extraction.start_char, extraction.end_char))
                    
                    # Extract relationships (if available)
                    if config["extract_relationships"] and hasattr(extraction, 'relationships'):
                        for rel in extraction.relationships:
                            relationships.append({
                                "source": rel.source,
                                "target": rel.target,
                                "type": rel.relationship_type,
                                "description": rel.description
                            })
                    
                    # Extract topics
                    if config["extract_topics"] and hasattr(extraction, 'topics'):
                        topics.extend(extraction.topics)
            
            # Generate summary if configured
            summary = None
            if config["generate_summary"]:
                summary = self._generate_summary(chunk_text, entities, topics)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(entities, relationships, topics)
            
            # Create enhanced chunk
            enhanced_chunk = EnhancedChunk(
                text=chunk_text,
                chunk_id=chunk_id,
                position=context.get("position", 0) if context else 0,
                metadata={
                    "extraction_mode": self.extraction_mode,
                    "model": self.model_id,
                    "timestamp": datetime.now().isoformat(),
                    **(context or {})
                },
                extracted_entities=entities,
                relationships=relationships,
                topics=list(set(topics)),  # Deduplicate topics
                summary=summary,
                confidence_score=confidence_score,
                source_spans=source_spans
            )
            
            # Cache the result
            if self.use_cache:
                self._cache_extraction(chunk_id, enhanced_chunk)
            
            return enhanced_chunk
            
        except Exception as e:
            logger.error(f"LangExtract enhancement failed: {e}")
            # Return basic chunk without enhancements
            return EnhancedChunk(
                text=chunk_text,
                chunk_id=chunk_id,
                position=context.get("position", 0) if context else 0,
                metadata=context or {},
                extracted_entities=[],
                relationships=[],
                topics=[],
                summary=None,
                confidence_score=0.0,
                source_spans=[]
            )
    
    def enhance_chunks_batch(self,
                            chunks: List[str],
                            context: Optional[Dict[str, Any]] = None) -> List[EnhancedChunk]:
        """
        Enhance multiple chunks in batch.
        
        Args:
            chunks: List of text chunks
            context: Optional context
            
        Returns:
            List of enhanced chunks
        """
        enhanced_chunks = []
        
        for i, chunk_text in enumerate(chunks):
            chunk_context = {
                **(context or {}),
                "position": i,
                "total_chunks": len(chunks)
            }
            
            enhanced_chunk = self.enhance_chunk(
                chunk_text,
                chunk_id=f"chunk_{i}",
                context=chunk_context
            )
            enhanced_chunks.append(enhanced_chunk)
        
        # Post-process to add cross-chunk relationships
        enhanced_chunks = self._add_cross_chunk_relationships(enhanced_chunks)
        
        return enhanced_chunks
    
    def _prepare_examples(self, context: Optional[Dict[str, Any]]) -> List[ExampleData]:
        """
        Prepare few-shot examples for better extraction.
        
        Args:
            context: Optional context with domain info
            
        Returns:
            List of examples for LangExtract
        """
        # Domain-specific examples
        domain = context.get("domain", "general") if context else "general"
        
        examples = []
        
        if domain == "technical":
            examples.append(ExampleData(
                text="The RAG system uses LangChain for orchestration and FAISS for vector search.",
                extractions=[
                    {"entity": "RAG system", "type": "Technology"},
                    {"entity": "LangChain", "type": "Framework"},
                    {"entity": "FAISS", "type": "Library"},
                    {"relationship": "RAG system -> uses -> LangChain"},
                    {"relationship": "RAG system -> uses -> FAISS"}
                ]
            ))
        elif domain == "medical":
            examples.append(ExampleData(
                text="Patient was prescribed 500mg of amoxicillin twice daily for bacterial infection.",
                extractions=[
                    {"entity": "amoxicillin", "type": "Medication", "dosage": "500mg"},
                    {"entity": "bacterial infection", "type": "Condition"},
                    {"frequency": "twice daily"}
                ]
            ))
        else:
            # General example
            examples.append(ExampleData(
                text="Google released LangExtract in 2024 as an open-source library.",
                extractions=[
                    {"entity": "Google", "type": "Organization"},
                    {"entity": "LangExtract", "type": "Product"},
                    {"entity": "2024", "type": "Date"},
                    {"relationship": "Google -> released -> LangExtract"}
                ]
            ))
        
        return examples
    
    def _generate_summary(self, 
                         text: str,
                         entities: List[Dict[str, Any]],
                         topics: List[str]) -> str:
        """
        Generate a brief summary of the chunk.
        
        Args:
            text: Original text
            entities: Extracted entities
            topics: Extracted topics
            
        Returns:
            Summary string
        """
        # Simple summarization based on entities and topics
        if not entities and not topics:
            return "No significant content extracted."
        
        summary_parts = []
        
        if topics:
            summary_parts.append(f"Topics: {', '.join(topics[:3])}")
        
        if entities:
            entity_types = {}
            for entity in entities[:5]:  # Limit to top 5
                etype = entity.get("type", "Unknown")
                if etype not in entity_types:
                    entity_types[etype] = []
                entity_types[etype].append(entity["text"])
            
            for etype, items in entity_types.items():
                summary_parts.append(f"{etype}: {', '.join(items[:3])}")
        
        return " | ".join(summary_parts)
    
    def _calculate_confidence(self,
                            entities: List[Dict[str, Any]],
                            relationships: List[Dict[str, Any]],
                            topics: List[str]) -> float:
        """
        Calculate confidence score for extraction quality.
        
        Args:
            entities: Extracted entities
            relationships: Extracted relationships
            topics: Extracted topics
            
        Returns:
            Confidence score (0.0-1.0)
        """
        score = 0.0
        
        # Base score on extraction counts
        if entities:
            score += min(len(entities) / 10, 0.3)  # Max 0.3 for entities
        if relationships:
            score += min(len(relationships) / 5, 0.3)  # Max 0.3 for relationships
        if topics:
            score += min(len(topics) / 5, 0.2)  # Max 0.2 for topics
        
        # Bonus for entity confidence scores
        if entities:
            avg_confidence = sum(e.get("confidence", 0.5) for e in entities) / len(entities)
            score += avg_confidence * 0.2  # Max 0.2 bonus
        
        return min(score, 1.0)
    
    def _add_cross_chunk_relationships(self,
                                      chunks: List[EnhancedChunk]) -> List[EnhancedChunk]:
        """
        Identify relationships between entities across chunks.
        
        Args:
            chunks: List of enhanced chunks
            
        Returns:
            Chunks with cross-chunk relationships added
        """
        # Build entity index across all chunks
        entity_index = {}
        
        for chunk_idx, chunk in enumerate(chunks):
            for entity in chunk.extracted_entities:
                entity_text = entity["text"].lower()
                if entity_text not in entity_index:
                    entity_index[entity_text] = []
                entity_index[entity_text].append({
                    "chunk_id": chunk.chunk_id,
                    "chunk_idx": chunk_idx,
                    "entity": entity
                })
        
        # Find co-references and add relationships
        for entity_text, occurrences in entity_index.items():
            if len(occurrences) > 1:
                # Entity appears in multiple chunks
                for i in range(len(occurrences) - 1):
                    source = occurrences[i]
                    target = occurrences[i + 1]
                    
                    # Add cross-reference relationship
                    chunks[source["chunk_idx"]].relationships.append({
                        "source": source["entity"]["text"],
                        "target": f"chunk_{target['chunk_idx']}",
                        "type": "cross_reference",
                        "description": f"Continues in chunk {target['chunk_idx']}"
                    })
        
        return chunks
    
    def _get_cached_extraction(self, chunk_id: str) -> Optional[EnhancedChunk]:
        """Get cached extraction if available"""
        cache_file = self.cache_dir / f"{chunk_id}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return EnhancedChunk(**data)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        return None
    
    def _cache_extraction(self, chunk_id: str, chunk: EnhancedChunk):
        """Cache extraction result"""
        cache_file = self.cache_dir / f"{chunk_id}.json"
        
        try:
            # Convert to dict for JSON serialization
            data = {
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "position": chunk.position,
                "metadata": chunk.metadata,
                "extracted_entities": chunk.extracted_entities,
                "relationships": chunk.relationships,
                "topics": chunk.topics,
                "summary": chunk.summary,
                "confidence_score": chunk.confidence_score,
                "source_spans": chunk.source_spans
            }
            
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache extraction: {e}")
    
    def visualize_extractions(self,
                             chunks: List[EnhancedChunk],
                             output_path: Optional[str] = None) -> str:
        """
        Generate interactive HTML visualization of extractions.
        
        Args:
            chunks: List of enhanced chunks
            output_path: Optional path to save HTML
            
        Returns:
            Path to generated HTML file
        """
        if not output_path:
            output_path = self.cache_dir / f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Prepare data for visualization
        documents = []
        for chunk in chunks:
            # Convert to LangExtract format
            extractions = []
            for entity in chunk.extracted_entities:
                extraction = Extraction(
                    text=entity["text"],
                    entity_type=entity["type"],
                    attributes=entity.get("attributes", {}),
                    start_char=entity["span"][0] if "span" in entity else 0,
                    end_char=entity["span"][1] if "span" in entity else len(entity["text"])
                )
                extractions.append(extraction)
            
            documents.append({
                "text": chunk.text,
                "extractions": extractions,
                "metadata": chunk.metadata
            })
        
        # Generate visualization
        try:
            lx.visualize(
                documents=documents,
                output_path=str(output_path)
            )
            logger.info(f"Visualization saved to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to generate visualization: {e}")
            return ""


class LangExtractRAGIntegration:
    """
    Integration layer for using LangExtract with RAG systems.
    """
    
    def __init__(self,
                 enhancer: Optional[LangExtractEnhancer] = None,
                 enable_for_indexing: bool = True,
                 enable_for_retrieval: bool = True):
        """
        Initialize RAG integration.
        
        Args:
            enhancer: LangExtractEnhancer instance
            enable_for_indexing: Use during document indexing
            enable_for_retrieval: Use during query processing
        """
        self.enhancer = enhancer or LangExtractEnhancer()
        self.enable_for_indexing = enable_for_indexing
        self.enable_for_retrieval = enable_for_retrieval
    
    def process_documents_for_indexing(self,
                                      documents: List[Dict[str, Any]],
                                      chunking_function: Any) -> List[Dict[str, Any]]:
        """
        Process documents with LangExtract before indexing.
        
        Args:
            documents: List of documents to process
            chunking_function: Original chunking function
            
        Returns:
            Enhanced documents with metadata
        """
        enhanced_documents = []
        
        for doc in documents:
            # Apply original chunking
            chunks = chunking_function(doc["text"])
            
            # Enhance chunks with LangExtract
            if self.enable_for_indexing:
                enhanced_chunks = self.enhancer.enhance_chunks_batch(
                    chunks,
                    context={"document_id": doc.get("id"), "source": doc.get("source")}
                )
                
                # Convert to document format
                for chunk in enhanced_chunks:
                    enhanced_doc = {
                        "text": chunk.text,
                        "metadata": {
                            **doc.get("metadata", {}),
                            **chunk.metadata,
                            "entities": chunk.extracted_entities,
                            "topics": chunk.topics,
                            "relationships": chunk.relationships,
                            "summary": chunk.summary,
                            "confidence_score": chunk.confidence_score
                        },
                        "embedding": None  # Will be filled by embedding model
                    }
                    enhanced_documents.append(enhanced_doc)
            else:
                # Skip enhancement
                for chunk_text in chunks:
                    enhanced_documents.append({
                        "text": chunk_text,
                        "metadata": doc.get("metadata", {}),
                        "embedding": None
                    })
        
        return enhanced_documents
    
    def enhance_retrieval_context(self,
                                 query: str,
                                 retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhance retrieval context with query-specific extraction.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved document chunks
            
        Returns:
            Enhanced context for generation
        """
        if not self.enable_for_retrieval:
            return {"query": query, "context": retrieved_chunks}
        
        # Extract entities from query
        query_enhancement = self.enhancer.enhance_chunk(
            query,
            context={"type": "query"}
        )
        
        # Find relevant entities in retrieved chunks
        relevant_entities = set()
        relevant_topics = set()
        
        for chunk in retrieved_chunks:
            if "entities" in chunk.get("metadata", {}):
                for entity in chunk["metadata"]["entities"]:
                    # Check if entity is relevant to query
                    if any(qe["text"].lower() in entity["text"].lower() 
                          for qe in query_enhancement.extracted_entities):
                        relevant_entities.add(entity["text"])
            
            if "topics" in chunk.get("metadata", {}):
                relevant_topics.update(chunk["metadata"]["topics"])
        
        # Build enhanced context
        enhanced_context = {
            "query": query,
            "query_entities": query_enhancement.extracted_entities,
            "query_topics": query_enhancement.topics,
            "context": retrieved_chunks,
            "relevant_entities": list(relevant_entities),
            "relevant_topics": list(relevant_topics),
            "extraction_confidence": query_enhancement.confidence_score
        }
        
        return enhanced_context