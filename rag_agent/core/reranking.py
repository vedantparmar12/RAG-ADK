"""
Advanced re-ranking module with coherence scoring, clustering, and document transformers.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import torch

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class RankedDocument:
    """Represents a ranked document with metadata"""
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    coherence_score: Optional[float] = None
    diversity_score: Optional[float] = None
    cluster_id: Optional[int] = None


class DocumentTransformer(ABC):
    """Abstract base class for document transformers"""
    
    @abstractmethod
    def transform(
        self, 
        documents: List[RankedDocument], 
        query: str,
        **kwargs
    ) -> List[RankedDocument]:
        """Transform the document list"""
        pass


class CoherenceReranker(DocumentTransformer):
    """Re-ranks documents based on coherence and relevance"""
    
    def __init__(self):
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            # Use cross-encoder for pairwise relevance scoring
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        else:
            logger.warning("Sentence transformers not available, using mock reranker")
            self.cross_encoder = None
            self.encoder = None
    
    def transform(
        self, 
        documents: List[RankedDocument], 
        query: str,
        coherence_weight: float = 0.3,
        diversity_weight: float = 0.2,
        **kwargs
    ) -> List[RankedDocument]:
        """Rerank documents considering coherence, relevance, and diversity"""
        
        if not documents or not self.cross_encoder:
            return documents
        
        # Step 1: Calculate cross-encoder relevance scores
        relevance_scores = self._calculate_relevance_scores(documents, query)
        
        # Step 2: Calculate coherence scores
        coherence_scores = self._calculate_coherence_scores(documents)
        
        # Step 3: Calculate diversity scores
        diversity_scores = self._calculate_diversity_scores(documents)
        
        # Step 4: Combine scores
        final_scores = []
        for i, doc in enumerate(documents):
            relevance = relevance_scores[i]
            coherence = coherence_scores[i]
            diversity = diversity_scores[i]
            
            # Weighted combination
            final_score = (
                (1 - coherence_weight - diversity_weight) * relevance +
                coherence_weight * coherence +
                diversity_weight * diversity
            )
            
            doc.coherence_score = coherence
            doc.diversity_score = diversity
            final_scores.append(final_score)
        
        # Sort by final score
        ranked_indices = np.argsort(final_scores)[::-1]
        return [documents[i] for i in ranked_indices]
    
    def _calculate_relevance_scores(
        self, 
        documents: List[RankedDocument], 
        query: str
    ) -> List[float]:
        """Calculate relevance scores using cross-encoder"""
        
        # Prepare query-document pairs
        pairs = [[query, doc.text] for doc in documents]
        
        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)
        
        # Normalize scores
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores.tolist()
    
    def _calculate_coherence_scores(
        self, 
        documents: List[RankedDocument]
    ) -> List[float]:
        """Calculate coherence scores based on document flow"""
        
        if len(documents) < 2:
            return [1.0] * len(documents)
        
        # Get embeddings for all documents
        embeddings = []
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = self.encoder.encode(doc.text)
            embeddings.append(doc.embedding)
        
        embeddings = np.array(embeddings)
        
        # Calculate pairwise similarities
        coherence_scores = []
        
        for i in range(len(documents)):
            if i == 0:
                # First document - check similarity with next
                sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                coherence_scores.append(sim)
            elif i == len(documents) - 1:
                # Last document - check similarity with previous
                sim = cosine_similarity([embeddings[i]], [embeddings[i-1]])[0][0]
                coherence_scores.append(sim)
            else:
                # Middle documents - average similarity with neighbors
                sim_prev = cosine_similarity([embeddings[i]], [embeddings[i-1]])[0][0]
                sim_next = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                coherence_scores.append((sim_prev + sim_next) / 2)
        
        return coherence_scores
    
    def _calculate_diversity_scores(
        self, 
        documents: List[RankedDocument]
    ) -> List[float]:
        """Calculate diversity scores to avoid redundancy"""
        
        if len(documents) < 2:
            return [1.0] * len(documents)
        
        # Get embeddings
        embeddings = []
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = self.encoder.encode(doc.text)
            embeddings.append(doc.embedding)
        
        embeddings = np.array(embeddings)
        
        # Calculate similarity matrix
        sim_matrix = cosine_similarity(embeddings)
        
        # Diversity score = 1 - average similarity to other documents
        diversity_scores = []
        for i in range(len(documents)):
            # Exclude self-similarity
            similarities = np.concatenate([sim_matrix[i, :i], sim_matrix[i, i+1:]])
            avg_similarity = np.mean(similarities) if len(similarities) > 0 else 0
            diversity_scores.append(1 - avg_similarity)
        
        return diversity_scores


class DocumentClusterer(DocumentTransformer):
    """Clusters documents and reorders based on cluster coherence"""
    
    def __init__(self, clustering_method: Literal["kmeans", "dbscan"] = "kmeans"):
        self.clustering_method = clustering_method
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        else:
            self.encoder = None
    
    def transform(
        self, 
        documents: List[RankedDocument], 
        query: str,
        n_clusters: Optional[int] = None,
        **kwargs
    ) -> List[RankedDocument]:
        """Cluster documents and reorder by cluster relevance"""
        
        if not documents or len(documents) < 3 or not self.encoder:
            return documents
        
        # Get embeddings
        embeddings = []
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = self.encoder.encode(doc.text)
            embeddings.append(doc.embedding)
        
        embeddings = np.array(embeddings)
        
        # Perform clustering
        if self.clustering_method == "kmeans":
            clusters = self._kmeans_clustering(embeddings, n_clusters)
        else:
            clusters = self._dbscan_clustering(embeddings)
        
        # Assign cluster IDs to documents
        for i, doc in enumerate(documents):
            doc.cluster_id = int(clusters[i])
        
        # Get query embedding
        query_embedding = self.encoder.encode(query)
        
        # Calculate cluster relevance scores
        cluster_scores = self._calculate_cluster_scores(
            documents, embeddings, clusters, query_embedding
        )
        
        # Reorder documents by cluster relevance, then by individual relevance
        reordered = []
        for cluster_id in sorted(cluster_scores.keys(), 
                               key=lambda x: cluster_scores[x], 
                               reverse=True):
            cluster_docs = [doc for doc in documents if doc.cluster_id == cluster_id]
            # Sort within cluster by original score
            cluster_docs.sort(key=lambda x: x.score, reverse=True)
            reordered.extend(cluster_docs)
        
        return reordered
    
    def _kmeans_clustering(
        self, 
        embeddings: np.ndarray, 
        n_clusters: Optional[int] = None
    ) -> np.ndarray:
        """Perform K-means clustering"""
        
        if n_clusters is None:
            # Automatically determine number of clusters
            n_clusters = min(5, max(2, len(embeddings) // 5))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(embeddings)
    
    def _dbscan_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform DBSCAN clustering"""
        
        # Calculate similarity matrix
        sim_matrix = cosine_similarity(embeddings)
        
        # Convert to distance matrix
        distance_matrix = 1 - sim_matrix
        
        # DBSCAN with adaptive epsilon
        dbscan = DBSCAN(eps=0.3, min_samples=2, metric='precomputed')
        return dbscan.fit_predict(distance_matrix)
    
    def _calculate_cluster_scores(
        self,
        documents: List[RankedDocument],
        embeddings: np.ndarray,
        clusters: np.ndarray,
        query_embedding: np.ndarray
    ) -> Dict[int, float]:
        """Calculate relevance score for each cluster"""
        
        cluster_scores = {}
        unique_clusters = np.unique(clusters)
        
        for cluster_id in unique_clusters:
            # Get indices of documents in this cluster
            cluster_indices = np.where(clusters == cluster_id)[0]
            
            # Calculate average embedding for cluster
            cluster_embedding = np.mean(embeddings[cluster_indices], axis=0)
            
            # Calculate similarity to query
            similarity = cosine_similarity([query_embedding], [cluster_embedding])[0][0]
            
            # Weight by cluster size (prefer larger clusters slightly)
            size_weight = np.log(len(cluster_indices) + 1) / np.log(len(documents) + 1)
            
            # Combined score
            cluster_scores[cluster_id] = similarity * (0.8 + 0.2 * size_weight)
        
        return cluster_scores


class DiversityTransformer(DocumentTransformer):
    """Ensures diversity in the result set using MMR (Maximal Marginal Relevance)"""
    
    def __init__(self):
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        else:
            self.encoder = None
    
    def transform(
        self, 
        documents: List[RankedDocument], 
        query: str,
        lambda_param: float = 0.5,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[RankedDocument]:
        """Apply MMR to balance relevance and diversity"""
        
        if not documents or not self.encoder:
            return documents
        
        if top_k is None:
            top_k = len(documents)
        
        # Get embeddings
        embeddings = []
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = self.encoder.encode(doc.text)
            embeddings.append(doc.embedding)
        
        embeddings = np.array(embeddings)
        query_embedding = self.encoder.encode(query)
        
        # Calculate relevance scores
        relevance_scores = cosine_similarity([query_embedding], embeddings)[0]
        
        # MMR selection
        selected_indices = []
        remaining_indices = list(range(len(documents)))
        
        # Select first document (highest relevance)
        first_idx = np.argmax(relevance_scores)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Iteratively select documents
        while len(selected_indices) < min(top_k, len(documents)) and remaining_indices:
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance to query
                relevance = relevance_scores[idx]
                
                # Maximum similarity to already selected documents
                max_sim = 0
                for selected_idx in selected_indices:
                    sim = cosine_similarity(
                        [embeddings[idx]], 
                        [embeddings[selected_idx]]
                    )[0][0]
                    max_sim = max(max_sim, sim)
                
                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append(mmr)
            
            # Select document with highest MMR score
            best_idx = remaining_indices[np.argmax(mmr_scores)]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Return reordered documents
        return [documents[i] for i in selected_indices]


class ReorderTransformer(DocumentTransformer):
    """Reorders documents based on multiple criteria"""
    
    def __init__(self, strategy: Literal["relevance", "recency", "length", "source"] = "relevance"):
        self.strategy = strategy
    
    def transform(
        self, 
        documents: List[RankedDocument], 
        query: str,
        **kwargs
    ) -> List[RankedDocument]:
        """Reorder documents based on strategy"""
        
        if self.strategy == "relevance":
            # Already sorted by relevance (default)
            return sorted(documents, key=lambda x: x.score, reverse=True)
        
        elif self.strategy == "recency":
            # Sort by timestamp if available
            def get_timestamp(doc):
                return doc.metadata.get("timestamp", doc.metadata.get("created_at", 0))
            return sorted(documents, key=get_timestamp, reverse=True)
        
        elif self.strategy == "length":
            # Prefer more comprehensive documents
            return sorted(documents, key=lambda x: len(x.text), reverse=True)
        
        elif self.strategy == "source":
            # Group by source, then by relevance
            source_groups = defaultdict(list)
            for doc in documents:
                source = doc.metadata.get("source", "unknown")
                source_groups[source].append(doc)
            
            # Sort within each group
            reordered = []
            for source in sorted(source_groups.keys()):
                group = sorted(source_groups[source], key=lambda x: x.score, reverse=True)
                reordered.extend(group)
            
            return reordered
        
        else:
            return documents


class HybridReranker:
    """Combines multiple reranking strategies"""
    
    def __init__(self):
        self.coherence_reranker = CoherenceReranker()
        self.clusterer = DocumentClusterer()
        self.diversity_transformer = DiversityTransformer()
        self.reorder_transformer = ReorderTransformer()
        
        # Load cross-encoder if available
        if TRANSFORMERS_AVAILABLE:
            try:
                self.cross_encoder_model = AutoModelForSequenceClassification.from_pretrained(
                    'cross-encoder/ms-marco-MiniLM-L-6-v2'
                )
                self.cross_encoder_tokenizer = AutoTokenizer.from_pretrained(
                    'cross-encoder/ms-marco-MiniLM-L-6-v2'
                )
                logger.info("Cross-encoder model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}")
                self.cross_encoder_model = None
        else:
            self.cross_encoder_model = None
    
    def rerank(
        self,
        documents: List[Dict[str, Any]],
        query: str,
        strategies: List[str] = ["coherence", "diversity"],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Apply multiple reranking strategies"""
        
        # Convert to RankedDocument objects
        ranked_docs = []
        for i, doc in enumerate(documents):
            ranked_doc = RankedDocument(
                doc_id=doc.get("doc_id", str(i)),
                text=doc.get("text", ""),
                score=doc.get("score", 0.0),
                metadata=doc.get("metadata", {})
            )
            ranked_docs.append(ranked_doc)
        
        # Apply strategies in sequence
        for strategy in strategies:
            if strategy == "coherence":
                ranked_docs = self.coherence_reranker.transform(ranked_docs, query, **kwargs)
            elif strategy == "clustering":
                ranked_docs = self.clusterer.transform(ranked_docs, query, **kwargs)
            elif strategy == "diversity":
                ranked_docs = self.diversity_transformer.transform(
                    ranked_docs, query, top_k=top_k, **kwargs
                )
            elif strategy == "reorder":
                reorder_strategy = kwargs.get("reorder_strategy", "relevance")
                self.reorder_transformer.strategy = reorder_strategy
                ranked_docs = self.reorder_transformer.transform(ranked_docs, query, **kwargs)
        
        # Convert back to dictionaries with additional metadata
        results = []
        for doc in ranked_docs[:top_k] if top_k else ranked_docs:
            result = {
                "doc_id": doc.doc_id,
                "text": doc.text,
                "score": doc.score,
                "metadata": doc.metadata,
                "reranking_metadata": {
                    "coherence_score": doc.coherence_score,
                    "diversity_score": doc.diversity_score,
                    "cluster_id": doc.cluster_id
                }
            }
            results.append(result)
        
        return results
    
    def evaluate_ranking(
        self,
        documents: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, float]:
        """Evaluate the quality of the ranking"""
        
        # Convert to RankedDocument objects
        ranked_docs = []
        for doc in documents:
            ranked_doc = RankedDocument(
                doc_id=doc.get("doc_id", ""),
                text=doc.get("text", ""),
                score=doc.get("score", 0.0),
                metadata=doc.get("metadata", {})
            )
            ranked_docs.append(ranked_doc)
        
        # Calculate metrics
        metrics = {}
        
        # 1. Average coherence score
        coherence_scores = self.coherence_reranker._calculate_coherence_scores(ranked_docs)
        metrics["avg_coherence"] = np.mean(coherence_scores)
        
        # 2. Diversity (average pairwise dissimilarity)
        if len(ranked_docs) > 1 and self.coherence_reranker.encoder:
            embeddings = []
            for doc in ranked_docs:
                if doc.embedding is None:
                    doc.embedding = self.coherence_reranker.encoder.encode(doc.text)
                embeddings.append(doc.embedding)
            
            sim_matrix = cosine_similarity(embeddings)
            # Exclude diagonal
            mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
            avg_similarity = sim_matrix[mask].mean()
            metrics["diversity"] = 1 - avg_similarity
        else:
            metrics["diversity"] = 0.0
        
        # 3. Coverage (unique topics/clusters)
        if len(ranked_docs) > 2:
            clusterer = DocumentClusterer()
            clustered = clusterer.transform(ranked_docs, query)
            unique_clusters = len(set(doc.cluster_id for doc in clustered if doc.cluster_id is not None))
            metrics["topic_coverage"] = unique_clusters / len(ranked_docs)
        else:
            metrics["topic_coverage"] = 1.0
        
        return metrics


# Global reranker instance
reranker = HybridReranker()