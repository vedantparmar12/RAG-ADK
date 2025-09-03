"""
Advanced Semantic Chunking Module
Replaces fixed-size chunking with semantic boundary-aware chunking
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import nltk
import spacy
from sentence_transformers import SentenceTransformer
from sentence_splitter import SentenceSplitter
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"Could not download NLTK data: {e}")

class ChunkingStrategy(Enum):
    """Different semantic chunking strategies"""
    SENTENCE_SIMILARITY = "sentence_similarity"
    TOPIC_SEGMENTATION = "topic_segmentation"
    HIERARCHICAL_CLUSTERING = "hierarchical_clustering"
    HYBRID = "hybrid"
    PARAGRAPH_AWARE = "paragraph_aware"

@dataclass
class SemanticChunk:
    """Container for semantic chunks with metadata"""
    id: str
    content: str
    start_char: int
    end_char: int
    sentence_count: int
    paragraph_count: int
    coherence_score: float
    topic_keywords: List[str]
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class SemanticSentenceSplitter:
    """Advanced sentence splitting with semantic awareness"""
    
    def __init__(self, language: str = "en"):
        self.language = language
        self.splitter = SentenceSplitter(language=language)
        
        # Initialize spaCy if available
        try:
            if language == "en":
                self.nlp = spacy.load("en_core_web_sm")
            else:
                self.nlp = None
        except IOError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def split_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into sentences with metadata"""
        sentences = []
        
        if self.nlp:
            # Use spaCy for better sentence segmentation
            doc = self.nlp(text)
            for sent in doc.sents:
                sentences.append({
                    'text': sent.text.strip(),
                    'start_char': sent.start_char,
                    'end_char': sent.end_char,
                    'entities': [(ent.text, ent.label_) for ent in sent.ents],
                    'pos_tags': [(token.text, token.pos_) for token in sent if not token.is_space]
                })
        else:
            # Fallback to sentence_splitter
            sent_texts = self.splitter.split(text)
            char_pos = 0
            for sent_text in sent_texts:
                start_pos = text.find(sent_text, char_pos)
                if start_pos != -1:
                    sentences.append({
                        'text': sent_text.strip(),
                        'start_char': start_pos,
                        'end_char': start_pos + len(sent_text),
                        'entities': [],
                        'pos_tags': []
                    })
                    char_pos = start_pos + len(sent_text)
        
        return sentences

class TopicSegmenter:
    """Topic-based text segmentation"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.similarity_threshold = 0.7
        
    def segment_by_topic(
        self,
        sentences: List[Dict[str, Any]],
        window_size: int = 3,
        similarity_threshold: float = None
    ) -> List[List[int]]:
        """Segment sentences into topic-coherent groups"""
        
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
        
        if len(sentences) < 2:
            return [[0]] if sentences else []
        
        # Get sentence embeddings
        sentence_texts = [sent['text'] for sent in sentences]
        embeddings = self.embedding_model.encode(sentence_texts)
        
        # Calculate sliding window similarities
        segments = []
        current_segment = [0]
        
        for i in range(1, len(sentences)):
            # Calculate similarity with previous window
            start_idx = max(0, i - window_size)
            prev_embeddings = embeddings[start_idx:i]
            curr_embedding = embeddings[i:i+1]
            
            # Average similarity with previous window
            similarities = cosine_similarity(curr_embedding, prev_embeddings)[0]
            avg_similarity = np.mean(similarities)
            
            if avg_similarity >= self.similarity_threshold:
                current_segment.append(i)
            else:
                # Start new segment
                segments.append(current_segment)
                current_segment = [i]
        
        if current_segment:
            segments.append(current_segment)
        
        return segments

class HierarchicalChunker:
    """Hierarchical clustering-based chunking"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        
    def chunk_hierarchically(
        self,
        sentences: List[Dict[str, Any]],
        max_chunk_size: int = 512,
        distance_threshold: float = 0.5
    ) -> List[List[int]]:
        """Use hierarchical clustering to group sentences"""
        
        if len(sentences) < 2:
            return [[0]] if sentences else []
        
        # Get embeddings
        sentence_texts = [sent['text'] for sent in sentences]
        embeddings = self.embedding_model.encode(sentence_texts)
        
        # Hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage='ward'
        )
        
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Group sentences by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        # Sort clusters by first sentence index
        sorted_clusters = sorted(clusters.values(), key=lambda x: x[0])
        
        # Split large clusters
        final_chunks = []
        for cluster in sorted_clusters:
            if len(cluster) == 1:
                final_chunks.append(cluster)
            else:
                # Split large clusters based on text length
                current_chunk = []
                current_length = 0
                
                for sent_idx in cluster:
                    sent_length = len(sentences[sent_idx]['text'])
                    
                    if current_length + sent_length > max_chunk_size and current_chunk:
                        final_chunks.append(current_chunk)
                        current_chunk = [sent_idx]
                        current_length = sent_length
                    else:
                        current_chunk.append(sent_idx)
                        current_length += sent_length
                
                if current_chunk:
                    final_chunks.append(current_chunk)
        
        return final_chunks

class AdvancedSemanticChunker:
    """Main semantic chunking orchestrator"""
    
    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.HYBRID,
        model_name: str = "all-MiniLM-L6-v2",
        language: str = "en"
    ):
        self.strategy = strategy
        self.model_name = model_name
        self.language = language
        
        # Initialize components
        self.sentence_splitter = SemanticSentenceSplitter(language)
        self.topic_segmenter = TopicSegmenter(model_name)
        self.hierarchical_chunker = HierarchicalChunker(model_name)
        self.embedding_model = SentenceTransformer(model_name)
        
    def chunk_text(
        self,
        text: str,
        max_chunk_size: int = 512,
        min_chunk_size: int = 50,
        overlap_size: int = 50,
        preserve_paragraphs: bool = True
    ) -> List[SemanticChunk]:
        """Main chunking method"""
        
        logger.info(f"Chunking text with strategy: {self.strategy.value}")
        
        # Pre-process text
        text = self._preprocess_text(text)
        
        # Split into sentences
        sentences = self.sentence_splitter.split_text(text)
        
        if not sentences:
            return []
        
        # Apply chunking strategy
        if self.strategy == ChunkingStrategy.SENTENCE_SIMILARITY:
            sentence_groups = self._similarity_based_chunking(sentences, max_chunk_size)
        elif self.strategy == ChunkingStrategy.TOPIC_SEGMENTATION:
            sentence_groups = self.topic_segmenter.segment_by_topic(sentences)
        elif self.strategy == ChunkingStrategy.HIERARCHICAL_CLUSTERING:
            sentence_groups = self.hierarchical_chunker.chunk_hierarchically(sentences, max_chunk_size)
        elif self.strategy == ChunkingStrategy.PARAGRAPH_AWARE:
            sentence_groups = self._paragraph_aware_chunking(text, sentences, max_chunk_size)
        else:  # HYBRID
            sentence_groups = self._hybrid_chunking(text, sentences, max_chunk_size)
        
        # Create chunks with overlap
        chunks = self._create_chunks_with_overlap(
            text, sentences, sentence_groups, overlap_size, min_chunk_size, max_chunk_size
        )
        
        # Add embeddings and metadata
        chunks = self._enrich_chunks(chunks)
        
        logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'[''']', "'", text)
        
        # Ensure proper sentence endings
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        return text.strip()
    
    def _similarity_based_chunking(
        self,
        sentences: List[Dict[str, Any]],
        max_chunk_size: int
    ) -> List[List[int]]:
        """Group sentences by semantic similarity"""
        
        if len(sentences) <= 1:
            return [[0]] if sentences else []
        
        sentence_texts = [sent['text'] for sent in sentences]
        embeddings = self.embedding_model.encode(sentence_texts)
        
        chunks = []
        current_chunk = [0]
        current_embedding = embeddings[0:1]
        current_size = len(sentences[0]['text'])
        
        for i in range(1, len(sentences)):
            sent_text = sentences[i]['text']
            sent_embedding = embeddings[i:i+1]
            
            # Calculate similarity with current chunk centroid
            similarity = cosine_similarity(sent_embedding, np.mean(current_embedding, axis=0, keepdims=True))[0][0]
            
            # Check if sentence fits in current chunk
            new_size = current_size + len(sent_text)
            
            if similarity >= 0.6 and new_size <= max_chunk_size:
                current_chunk.append(i)
                current_embedding = np.vstack([current_embedding, sent_embedding])
                current_size = new_size
            else:
                # Start new chunk
                chunks.append(current_chunk)
                current_chunk = [i]
                current_embedding = sent_embedding
                current_size = len(sent_text)
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _paragraph_aware_chunking(
        self,
        text: str,
        sentences: List[Dict[str, Any]],
        max_chunk_size: int
    ) -> List[List[int]]:
        """Chunk while preserving paragraph boundaries"""
        
        # Find paragraph boundaries
        paragraphs = text.split('\n\n')
        paragraph_boundaries = []
        char_pos = 0
        
        for para in paragraphs:
            para_start = char_pos
            para_end = char_pos + len(para)
            paragraph_boundaries.append((para_start, para_end))
            char_pos = para_end + 2  # Account for \n\n
        
        # Group sentences by paragraphs
        sentence_to_para = {}
        for i, sent in enumerate(sentences):
            sent_start = sent['start_char']
            for j, (para_start, para_end) in enumerate(paragraph_boundaries):
                if para_start <= sent_start <= para_end:
                    sentence_to_para[i] = j
                    break
        
        # Create paragraph-aware chunks
        chunks = []
        current_chunk = []
        current_size = 0
        current_para = -1
        
        for i, sent in enumerate(sentences):
            sent_para = sentence_to_para.get(i, -1)
            sent_size = len(sent['text'])
            
            # Check if we should start a new chunk
            if (current_para != -1 and sent_para != current_para) or \
               (current_size + sent_size > max_chunk_size and current_chunk):
                chunks.append(current_chunk)
                current_chunk = [i]
                current_size = sent_size
                current_para = sent_para
            else:
                current_chunk.append(i)
                current_size += sent_size
                if current_para == -1:
                    current_para = sent_para
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _hybrid_chunking(
        self,
        text: str,
        sentences: List[Dict[str, Any]],
        max_chunk_size: int
    ) -> List[List[int]]:
        """Combine multiple chunking strategies"""
        
        # First, try paragraph-aware chunking
        para_chunks = self._paragraph_aware_chunking(text, sentences, max_chunk_size)
        
        # Then apply semantic similarity within each paragraph chunk
        final_chunks = []
        
        for para_chunk in para_chunks:
            if len(para_chunk) <= 3:  # Small chunks, keep as is
                final_chunks.append(para_chunk)
            else:
                # Apply semantic chunking within paragraph
                para_sentences = [sentences[i] for i in para_chunk]
                semantic_groups = self._similarity_based_chunking(para_sentences, max_chunk_size)
                
                # Convert back to global indices
                for group in semantic_groups:
                    final_chunks.append([para_chunk[i] for i in group])
        
        return final_chunks
    
    def _create_chunks_with_overlap(
        self,
        text: str,
        sentences: List[Dict[str, Any]],
        sentence_groups: List[List[int]],
        overlap_size: int,
        min_chunk_size: int,
        max_chunk_size: int
    ) -> List[SemanticChunk]:
        """Create final chunks with overlap"""
        
        chunks = []
        
        for i, group in enumerate(sentence_groups):
            if not group:
                continue
            
            # Get sentences for this group
            group_sentences = [sentences[j] for j in group]
            
            # Calculate chunk boundaries
            start_char = group_sentences[0]['start_char']
            end_char = group_sentences[-1]['end_char']
            chunk_content = text[start_char:end_char]
            
            # Skip chunks that are too small
            if len(chunk_content.strip()) < min_chunk_size:
                continue
            
            # Add overlap from previous chunk
            if i > 0 and overlap_size > 0:
                prev_group = sentence_groups[i-1]
                if prev_group:
                    overlap_start = max(0, len(prev_group) - 2)  # Last 2 sentences
                    overlap_sentences = [sentences[j] for j in prev_group[overlap_start:]]
                    overlap_text = text[overlap_sentences[0]['start_char']:overlap_sentences[-1]['end_char']]
                    
                    if len(overlap_text) <= overlap_size:
                        chunk_content = overlap_text + " " + chunk_content
                        start_char = overlap_sentences[0]['start_char']
            
            # Calculate metadata
            paragraph_count = chunk_content.count('\n\n') + 1
            coherence_score = self._calculate_coherence_score(group_sentences)
            topic_keywords = self._extract_topic_keywords(chunk_content)
            
            chunk = SemanticChunk(
                id=f"chunk_{i:04d}",
                content=chunk_content,
                start_char=start_char,
                end_char=end_char,
                sentence_count=len(group_sentences),
                paragraph_count=paragraph_count,
                coherence_score=coherence_score,
                topic_keywords=topic_keywords,
                metadata={
                    'sentence_indices': group,
                    'strategy': self.strategy.value,
                    'word_count': len(chunk_content.split())
                }
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _enrich_chunks(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Add embeddings and additional metadata to chunks"""
        
        if not chunks:
            return chunks
        
        # Generate embeddings for all chunks
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(chunk_texts)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            
            # Add similarity scores to metadata
            if len(chunks) > 1:
                similarities = cosine_similarity([embedding], embeddings)[0]
                chunk.metadata['avg_similarity'] = float(np.mean(similarities))
                chunk.metadata['max_similarity'] = float(np.max(similarities[similarities < 1.0])) if len(similarities) > 1 else 0.0
        
        return chunks
    
    def _calculate_coherence_score(self, sentences: List[Dict[str, Any]]) -> float:
        """Calculate semantic coherence score for a group of sentences"""
        
        if len(sentences) <= 1:
            return 1.0
        
        sentence_texts = [sent['text'] for sent in sentences]
        embeddings = self.embedding_model.encode(sentence_texts)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Average similarity excluding self-similarities
        mask = np.ones_like(similarities, dtype=bool)
        np.fill_diagonal(mask, False)
        
        if mask.sum() == 0:
            return 1.0
        
        return float(np.mean(similarities[mask]))
    
    def _extract_topic_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """Extract key topic words from text"""
        
        # Simple keyword extraction based on frequency and length
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {'the', 'and', 'are', 'for', 'with', 'this', 'that', 'have', 'will', 'can', 'but', 'not'}
        words = [word for word in words if word not in stop_words]
        
        # Count frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top k
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in sorted_words[:top_k]]

# Global instance
semantic_chunker = AdvancedSemanticChunker()

def get_semantic_chunker() -> AdvancedSemanticChunker:
    """Get the global semantic chunker instance"""
    return semantic_chunker