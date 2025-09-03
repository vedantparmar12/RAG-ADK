"""
Hierarchical Search Module
Performs multi-level retrieval: document-level, chunk-level, and sentence-level
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class SearchLevel(Enum):
    """Hierarchical search levels"""
    DOCUMENT = "document"
    CHUNK = "chunk"
    SENTENCE = "sentence"
    PHRASE = "phrase"

@dataclass
class HierarchicalDocument:
    """Document with hierarchical structure"""
    doc_id: str
    title: str
    content: str
    chunks: List[Dict[str, Any]]
    sentences: List[Dict[str, Any]]
    phrases: List[Dict[str, Any]]
    embeddings: Dict[str, np.ndarray]  # embeddings for each level
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SearchResult:
    """Result from hierarchical search"""
    doc_id: str
    level: SearchLevel
    content: str
    score: float
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    chunk_id: Optional[str] = None
    sentence_id: Optional[str] = None
    phrase_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HierarchicalSearchResult:
    """Complete hierarchical search result"""
    query: str
    results_by_level: Dict[SearchLevel, List[SearchResult]]
    aggregated_results: List[SearchResult]
    best_level: SearchLevel
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class DocumentHierarchyBuilder:
    """Build hierarchical document structures"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        
        # Import spaCy for sentence and phrase extraction
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except (ImportError, IOError):
            logger.warning("spaCy not available. Using simple text splitting.")
            self.nlp = None
    
    def build_hierarchy(
        self,
        doc_id: str,
        title: str,
        content: str,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        metadata: Optional[Dict[str, Any]] = None
    ) -> HierarchicalDocument:
        """Build hierarchical structure for a document"""
        
        logger.debug(f"Building hierarchy for document: {doc_id}")
        
        # Level 1: Document level
        doc_embedding = self.embedding_model.encode([content])[0]
        
        # Level 2: Chunk level
        chunks = self._create_chunks(content, chunk_size, chunk_overlap)
        chunk_texts = [chunk['content'] for chunk in chunks]
        chunk_embeddings = self.embedding_model.encode(chunk_texts) if chunk_texts else np.array([])
        
        # Level 3: Sentence level
        sentences = self._extract_sentences(content)
        sentence_texts = [sent['content'] for sent in sentences]
        sentence_embeddings = self.embedding_model.encode(sentence_texts) if sentence_texts else np.array([])
        
        # Level 4: Phrase level
        phrases = self._extract_phrases(content)
        phrase_texts = [phrase['content'] for phrase in phrases]
        phrase_embeddings = self.embedding_model.encode(phrase_texts) if phrase_texts else np.array([])
        
        # Store embeddings by level
        embeddings = {
            'document': doc_embedding,
            'chunks': chunk_embeddings,
            'sentences': sentence_embeddings,
            'phrases': phrase_embeddings
        }
        
        return HierarchicalDocument(
            doc_id=doc_id,
            title=title,
            content=content,
            chunks=chunks,
            sentences=sentences,
            phrases=phrases,
            embeddings=embeddings,
            metadata=metadata or {}
        )
    
    def _create_chunks(
        self,
        content: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Dict[str, Any]]:
        """Create overlapping chunks"""
        
        chunks = []
        words = content.split()
        
        if not words:
            return chunks
        
        # Approximate words per chunk (assuming ~5 chars per word)
        words_per_chunk = chunk_size // 5
        words_overlap = chunk_overlap // 5
        
        start_idx = 0
        chunk_id = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + words_per_chunk, len(words))
            
            chunk_words = words[start_idx:end_idx]
            chunk_content = ' '.join(chunk_words)
            
            # Find character positions
            start_char = content.find(chunk_words[0])
            if start_char == -1:
                start_char = 0
            
            end_char = content.rfind(chunk_words[-1])
            if end_char != -1:
                end_char += len(chunk_words[-1])
            else:
                end_char = len(content)
            
            chunks.append({
                'chunk_id': f"chunk_{chunk_id}",
                'content': chunk_content,
                'start_char': start_char,
                'end_char': end_char,
                'word_count': len(chunk_words)
            })
            
            # Move to next chunk with overlap
            start_idx += max(1, words_per_chunk - words_overlap)
            chunk_id += 1
        
        return chunks
    
    def _extract_sentences(self, content: str) -> List[Dict[str, Any]]:
        """Extract sentences with positions"""
        
        sentences = []
        
        if self.nlp:
            # Use spaCy for better sentence segmentation
            doc = self.nlp(content)
            for i, sent in enumerate(doc.sents):
                sentences.append({
                    'sentence_id': f"sent_{i}",
                    'content': sent.text.strip(),
                    'start_char': sent.start_char,
                    'end_char': sent.end_char,
                    'entities': [(ent.text, ent.label_) for ent in sent.ents]
                })
        else:
            # Simple sentence splitting
            import re
            sent_patterns = r'[.!?]+\s+'
            sent_texts = re.split(sent_patterns, content)
            
            char_pos = 0
            for i, sent_text in enumerate(sent_texts):
                sent_text = sent_text.strip()
                if not sent_text:
                    continue
                
                start_pos = content.find(sent_text, char_pos)
                if start_pos != -1:
                    sentences.append({
                        'sentence_id': f"sent_{i}",
                        'content': sent_text,
                        'start_char': start_pos,
                        'end_char': start_pos + len(sent_text),
                        'entities': []
                    })
                    char_pos = start_pos + len(sent_text)
        
        return sentences
    
    def _extract_phrases(self, content: str) -> List[Dict[str, Any]]:
        """Extract key phrases (noun phrases, named entities)"""
        
        phrases = []
        
        if self.nlp:
            doc = self.nlp(content)
            
            # Extract noun phrases
            for i, chunk in enumerate(doc.noun_chunks):
                if len(chunk.text.strip()) > 3:  # Minimum phrase length
                    phrases.append({
                        'phrase_id': f"np_{i}",
                        'content': chunk.text.strip(),
                        'start_char': chunk.start_char,
                        'end_char': chunk.end_char,
                        'type': 'noun_phrase',
                        'root': chunk.root.text
                    })
            
            # Extract named entities
            for i, ent in enumerate(doc.ents):
                if len(ent.text.strip()) > 2:
                    phrases.append({
                        'phrase_id': f"ne_{i}",
                        'content': ent.text.strip(),
                        'start_char': ent.start_char,
                        'end_char': ent.end_char,
                        'type': 'named_entity',
                        'label': ent.label_
                    })
        else:
            # Simple phrase extraction (basic patterns)
            import re
            
            # Extract capitalized phrases (potential named entities)
            cap_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
            
            char_pos = 0
            for i, phrase in enumerate(cap_phrases):
                start_pos = content.find(phrase, char_pos)
                if start_pos != -1:
                    phrases.append({
                        'phrase_id': f"cap_{i}",
                        'content': phrase,
                        'start_char': start_pos,
                        'end_char': start_pos + len(phrase),
                        'type': 'capitalized',
                        'label': 'UNKNOWN'
                    })
                    char_pos = start_pos + len(phrase)
        
        return phrases

class HierarchicalSearcher:
    """Perform hierarchical search across multiple levels"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.documents: Dict[str, HierarchicalDocument] = {}
        
        # Level weights for scoring
        self.level_weights = {
            SearchLevel.DOCUMENT: 1.0,
            SearchLevel.CHUNK: 0.8,
            SearchLevel.SENTENCE: 0.6,
            SearchLevel.PHRASE: 0.4
        }
    
    def add_document(self, hierarchical_doc: HierarchicalDocument):
        """Add a hierarchical document to the search index"""
        self.documents[hierarchical_doc.doc_id] = hierarchical_doc
        logger.debug(f"Added document {hierarchical_doc.doc_id} to hierarchical index")
    
    def search(
        self,
        query: str,
        top_k_per_level: int = 10,
        levels: Optional[List[SearchLevel]] = None,
        min_score: float = 0.3
    ) -> HierarchicalSearchResult:
        """Perform hierarchical search across all levels"""
        
        if not self.documents:
            return HierarchicalSearchResult(
                query=query,
                results_by_level={},
                aggregated_results=[],
                best_level=SearchLevel.DOCUMENT,
                confidence=0.0
            )
        
        # Default to all levels if not specified
        if levels is None:
            levels = list(SearchLevel)
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search at each level
        results_by_level = {}
        all_results = []
        
        for level in levels:
            level_results = self._search_at_level(
                query, query_embedding, level, top_k_per_level, min_score
            )
            results_by_level[level] = level_results
            all_results.extend(level_results)
        
        # Aggregate and rank results
        aggregated_results = self._aggregate_results(all_results)
        
        # Determine best level
        best_level = self._determine_best_level(results_by_level)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(aggregated_results)
        
        return HierarchicalSearchResult(
            query=query,
            results_by_level=results_by_level,
            aggregated_results=aggregated_results,
            best_level=best_level,
            confidence=confidence,
            metadata={
                'total_documents': len(self.documents),
                'levels_searched': len(levels),
                'total_results': len(all_results)
            }
        )
    
    def _search_at_level(
        self,
        query: str,
        query_embedding: np.ndarray,
        level: SearchLevel,
        top_k: int,
        min_score: float
    ) -> List[SearchResult]:
        """Search at a specific hierarchical level"""
        
        results = []
        
        for doc_id, doc in self.documents.items():
            if level == SearchLevel.DOCUMENT:
                # Document-level search
                doc_embedding = doc.embeddings['document']
                similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                
                if similarity >= min_score:
                    results.append(SearchResult(
                        doc_id=doc_id,
                        level=level,
                        content=doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                        score=float(similarity) * self.level_weights[level],
                        metadata={'title': doc.title, 'full_content_length': len(doc.content)}
                    ))
            
            elif level == SearchLevel.CHUNK:
                # Chunk-level search
                if len(doc.embeddings['chunks']) > 0:
                    similarities = cosine_similarity([query_embedding], doc.embeddings['chunks'])[0]
                    
                    for i, similarity in enumerate(similarities):
                        if similarity >= min_score:
                            chunk = doc.chunks[i]
                            results.append(SearchResult(
                                doc_id=doc_id,
                                level=level,
                                content=chunk['content'],
                                score=float(similarity) * self.level_weights[level],
                                start_char=chunk['start_char'],
                                end_char=chunk['end_char'],
                                chunk_id=chunk['chunk_id'],
                                metadata={'word_count': chunk['word_count']}
                            ))
            
            elif level == SearchLevel.SENTENCE:
                # Sentence-level search
                if len(doc.embeddings['sentences']) > 0:
                    similarities = cosine_similarity([query_embedding], doc.embeddings['sentences'])[0]
                    
                    for i, similarity in enumerate(similarities):
                        if similarity >= min_score:
                            sentence = doc.sentences[i]
                            results.append(SearchResult(
                                doc_id=doc_id,
                                level=level,
                                content=sentence['content'],
                                score=float(similarity) * self.level_weights[level],
                                start_char=sentence['start_char'],
                                end_char=sentence['end_char'],
                                sentence_id=sentence['sentence_id'],
                                metadata={'entities': sentence.get('entities', [])}
                            ))
            
            elif level == SearchLevel.PHRASE:
                # Phrase-level search
                if len(doc.embeddings['phrases']) > 0:
                    similarities = cosine_similarity([query_embedding], doc.embeddings['phrases'])[0]
                    
                    for i, similarity in enumerate(similarities):
                        if similarity >= min_score:
                            phrase = doc.phrases[i]
                            results.append(SearchResult(
                                doc_id=doc_id,
                                level=level,
                                content=phrase['content'],
                                score=float(similarity) * self.level_weights[level],
                                start_char=phrase['start_char'],
                                end_char=phrase['end_char'],
                                phrase_id=phrase['phrase_id'],
                                metadata={'type': phrase.get('type', 'unknown')}
                            ))
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _aggregate_results(self, all_results: List[SearchResult]) -> List[SearchResult]:
        """Aggregate results from different levels"""
        
        # Group results by document and position
        doc_results = {}
        
        for result in all_results:
            doc_id = result.doc_id
            
            if doc_id not in doc_results:
                doc_results[doc_id] = []
            
            doc_results[doc_id].append(result)
        
        # For each document, merge overlapping results and boost scores
        aggregated = []
        
        for doc_id, doc_results_list in doc_results.items():
            # Sort by position
            doc_results_list.sort(key=lambda x: x.start_char or 0)
            
            # Merge nearby results
            merged = []
            current_group = [doc_results_list[0]] if doc_results_list else []
            
            for result in doc_results_list[1:]:
                # If results are close together, group them
                if (current_group and 
                    result.start_char is not None and 
                    current_group[-1].end_char is not None and
                    result.start_char - current_group[-1].end_char < 50):
                    current_group.append(result)
                else:
                    # Process current group
                    if current_group:
                        merged.append(self._merge_result_group(current_group))
                    current_group = [result]
            
            # Process final group
            if current_group:
                merged.append(self._merge_result_group(current_group))
            
            aggregated.extend(merged)
        
        # Sort by final score
        aggregated.sort(key=lambda x: x.score, reverse=True)
        
        return aggregated
    
    def _merge_result_group(self, group: List[SearchResult]) -> SearchResult:
        """Merge a group of overlapping search results"""
        
        if len(group) == 1:
            return group[0]
        
        # Use the highest-level result as base
        level_priority = {
            SearchLevel.DOCUMENT: 4,
            SearchLevel.CHUNK: 3,
            SearchLevel.SENTENCE: 2,
            SearchLevel.PHRASE: 1
        }
        
        base_result = max(group, key=lambda x: level_priority[x.level])
        
        # Boost score based on multiple level matches
        score_boost = 1.0 + (len(group) - 1) * 0.1
        boosted_score = min(base_result.score * score_boost, 1.0)
        
        # Combine content if necessary
        if base_result.level != SearchLevel.DOCUMENT:
            # Use longest content
            longest_content = max(group, key=lambda x: len(x.content))
            content = longest_content.content
        else:
            content = base_result.content
        
        # Combine metadata
        combined_metadata = base_result.metadata.copy()
        combined_metadata['merged_from_levels'] = [r.level.value for r in group]
        combined_metadata['level_count'] = len(set(r.level for r in group))
        
        return SearchResult(
            doc_id=base_result.doc_id,
            level=base_result.level,
            content=content,
            score=boosted_score,
            start_char=base_result.start_char,
            end_char=base_result.end_char,
            chunk_id=base_result.chunk_id,
            sentence_id=base_result.sentence_id,
            phrase_id=base_result.phrase_id,
            metadata=combined_metadata
        )
    
    def _determine_best_level(
        self,
        results_by_level: Dict[SearchLevel, List[SearchResult]]
    ) -> SearchLevel:
        """Determine which level provided the best results"""
        
        level_scores = {}
        
        for level, results in results_by_level.items():
            if results:
                # Average score of top 3 results
                top_results = results[:3]
                avg_score = np.mean([r.score for r in top_results])
                level_scores[level] = avg_score
            else:
                level_scores[level] = 0.0
        
        if not level_scores:
            return SearchLevel.DOCUMENT
        
        return max(level_scores.keys(), key=lambda x: level_scores[x])
    
    def _calculate_confidence(self, results: List[SearchResult]) -> float:
        """Calculate overall confidence in search results"""
        
        if not results:
            return 0.0
        
        # Average score of top results
        top_results = results[:5]
        scores = [r.score for r in top_results]
        
        base_confidence = np.mean(scores)
        
        # Boost confidence if we have good results across multiple levels
        unique_levels = len(set(r.level for r in top_results))
        level_boost = min(unique_levels * 0.1, 0.3)
        
        return min(base_confidence + level_boost, 1.0)

# Global instance
hierarchical_searcher = HierarchicalSearcher()
hierarchy_builder = DocumentHierarchyBuilder()

def get_hierarchical_searcher() -> HierarchicalSearcher:
    """Get the global hierarchical searcher instance"""
    return hierarchical_searcher

def get_hierarchy_builder() -> DocumentHierarchyBuilder:
    """Get the global hierarchy builder instance"""
    return hierarchy_builder