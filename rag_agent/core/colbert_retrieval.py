"""
ColBERT-style Multi-Vector Embeddings and Late Interaction Retrieval
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import re
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import faiss

logger = logging.getLogger(__name__)

@dataclass
class ColBERTDocument:
    """Document with ColBERT-style multi-vector embeddings"""
    doc_id: str
    content: str
    token_embeddings: np.ndarray  # Shape: (num_tokens, embedding_dim)
    token_ids: List[int]
    attention_mask: List[int]
    metadata: Dict[str, Any]

@dataclass
class ColBERTQuery:
    """Query with ColBERT-style embeddings"""
    query_text: str
    token_embeddings: np.ndarray  # Shape: (num_tokens, embedding_dim)
    token_ids: List[int]
    attention_mask: List[int]

@dataclass
class LateInteractionResult:
    """Result from late interaction scoring"""
    doc_id: str
    score: float
    token_scores: List[float]
    matched_tokens: List[Tuple[int, int, float]]  # (query_token, doc_token, score)
    content: str
    metadata: Dict[str, Any]

class ColBERTEncoder(nn.Module):
    """ColBERT-style encoder with linear projection"""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embedding_dim: int = 128,
        similarity_metric: str = "cosine"
    ):
        super().__init__()
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.similarity_metric = similarity_metric
        
        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Linear projection layer (similar to ColBERT)
        self.projection = nn.Linear(
            self.encoder.config.hidden_size,
            embedding_dim
        )
        
        # Query and document specific [CLS] tokens
        self.query_token = "[Q]"
        self.doc_token = "[D]"
        
        # Add special tokens to tokenizer
        special_tokens = {"additional_special_tokens": [self.query_token, self.doc_token]}
        self.tokenizer.add_special_tokens(special_tokens)
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def encode_query(self, query_text: str, max_length: int = 64) -> ColBERTQuery:
        """Encode query with ColBERT-style representation"""
        
        # Add query marker
        marked_query = f"{self.query_token} {query_text}"
        
        # Tokenize
        encoded = self.tokenizer(
            marked_query,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        with torch.no_grad():
            # Get embeddings
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            
            # Apply projection
            projected = self.projection(last_hidden_state)
            
            # L2 normalize for cosine similarity
            if self.similarity_metric == "cosine":
                projected = F.normalize(projected, dim=-1)
            
            # Filter out padding tokens
            mask = attention_mask.squeeze().bool()
            token_embeddings = projected.squeeze()[mask]
        
        return ColBERTQuery(
            query_text=query_text,
            token_embeddings=token_embeddings.cpu().numpy(),
            token_ids=input_ids.squeeze().cpu().numpy().tolist(),
            attention_mask=attention_mask.squeeze().cpu().numpy().tolist()
        )
    
    def encode_document(self, doc_text: str, doc_id: str, max_length: int = 512) -> ColBERTDocument:
        """Encode document with ColBERT-style representation"""
        
        # Add document marker
        marked_doc = f"{self.doc_token} {doc_text}"
        
        # Tokenize
        encoded = self.tokenizer(
            marked_doc,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        with torch.no_grad():
            # Get embeddings
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            
            # Apply projection
            projected = self.projection(last_hidden_state)
            
            # L2 normalize for cosine similarity
            if self.similarity_metric == "cosine":
                projected = F.normalize(projected, dim=-1)
            
            # Filter out padding tokens
            mask = attention_mask.squeeze().bool()
            token_embeddings = projected.squeeze()[mask]
        
        return ColBERTDocument(
            doc_id=doc_id,
            content=doc_text,
            token_embeddings=token_embeddings.cpu().numpy(),
            token_ids=input_ids.squeeze().cpu().numpy().tolist(),
            attention_mask=attention_mask.squeeze().cpu().numpy().tolist(),
            metadata={}
        )

class LateInteractionScorer:
    """Late interaction scoring mechanism"""
    
    def __init__(self, similarity_metric: str = "cosine"):
        self.similarity_metric = similarity_metric
        
    def compute_interaction_score(
        self,
        query: ColBERTQuery,
        document: ColBERTDocument,
        k: int = 1
    ) -> LateInteractionResult:
        """Compute late interaction score between query and document"""
        
        query_embeddings = query.token_embeddings
        doc_embeddings = document.token_embeddings
        
        # Compute pairwise similarities
        if self.similarity_metric == "cosine":
            similarities = np.dot(query_embeddings, doc_embeddings.T)
        else:
            # L2 distance
            similarities = -np.linalg.norm(
                query_embeddings[:, None, :] - doc_embeddings[None, :, :], 
                axis=2
            )
        
        # Max pooling over document tokens for each query token
        token_scores = np.max(similarities, axis=1)
        
        # Overall score (mean of query token max scores)
        overall_score = np.mean(token_scores)
        
        # Find best matching tokens
        matched_tokens = []
        for q_idx, q_scores in enumerate(similarities):
            best_doc_idx = np.argmax(q_scores)
            best_score = q_scores[best_doc_idx]
            matched_tokens.append((q_idx, best_doc_idx, float(best_score)))
        
        return LateInteractionResult(
            doc_id=document.doc_id,
            score=float(overall_score),
            token_scores=token_scores.tolist(),
            matched_tokens=matched_tokens,
            content=document.content,
            metadata=document.metadata
        )
    
    def batch_score(
        self,
        query: ColBERTQuery,
        documents: List[ColBERTDocument]
    ) -> List[LateInteractionResult]:
        """Batch scoring for efficiency"""
        
        results = []
        for doc in documents:
            result = self.compute_interaction_score(query, doc)
            results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results

class ApproximateLateInteraction:
    """Approximate late interaction using clustering for efficiency"""
    
    def __init__(
        self,
        n_clusters: int = 1000,
        similarity_metric: str = "cosine"
    ):
        self.n_clusters = n_clusters
        self.similarity_metric = similarity_metric
        self.document_index = None
        self.cluster_centers = None
        self.doc_to_clusters = {}
        
    def build_index(self, documents: List[ColBERTDocument]):
        """Build approximate index using clustering"""
        
        logger.info("Building approximate ColBERT index...")
        
        # Collect all document token embeddings
        all_embeddings = []
        doc_embedding_ranges = []
        
        start_idx = 0
        for doc in documents:
            embeddings = doc.token_embeddings
            all_embeddings.append(embeddings)
            end_idx = start_idx + len(embeddings)
            doc_embedding_ranges.append((doc.doc_id, start_idx, end_idx))
            start_idx = end_idx
        
        # Stack all embeddings
        all_embeddings_matrix = np.vstack(all_embeddings)
        
        # Perform clustering
        n_clusters = min(self.n_clusters, len(all_embeddings_matrix))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_assignments = kmeans.fit_predict(all_embeddings_matrix)
        
        self.cluster_centers = kmeans.cluster_centers_
        
        # Map each document to its clusters
        for doc_id, start_idx, end_idx in doc_embedding_ranges:
            doc_clusters = set(cluster_assignments[start_idx:end_idx])
            self.doc_to_clusters[doc_id] = doc_clusters
        
        # Build FAISS index for fast cluster search
        if self.similarity_metric == "cosine":
            # Normalize cluster centers for cosine similarity
            self.cluster_centers = self.cluster_centers / np.linalg.norm(
                self.cluster_centers, axis=1, keepdims=True
            )
            self.document_index = faiss.IndexFlatIP(self.cluster_centers.shape[1])
        else:
            self.document_index = faiss.IndexFlatL2(self.cluster_centers.shape[1])
        
        self.document_index.add(self.cluster_centers.astype(np.float32))
        
        logger.info(f"Index built with {n_clusters} clusters")
    
    def approximate_search(
        self,
        query: ColBERTQuery,
        documents: Dict[str, ColBERTDocument],
        top_k: int = 100,
        cluster_search_k: int = 50
    ) -> List[str]:
        """Approximate search using cluster-based filtering"""
        
        if self.document_index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Find relevant clusters for query tokens
        query_embeddings = query.token_embeddings.astype(np.float32)
        if self.similarity_metric == "cosine":
            query_embeddings = query_embeddings / np.linalg.norm(
                query_embeddings, axis=1, keepdims=True
            )
        
        # Search for top clusters for each query token
        _, cluster_indices = self.document_index.search(query_embeddings, cluster_search_k)
        
        # Collect candidate documents
        candidate_docs = set()
        for token_clusters in cluster_indices:
            for cluster_id in token_clusters:
                # Find documents that have embeddings in this cluster
                for doc_id, doc_clusters in self.doc_to_clusters.items():
                    if cluster_id in doc_clusters:
                        candidate_docs.add(doc_id)
        
        # Limit candidates
        candidate_docs = list(candidate_docs)[:top_k * 2]  # Get more candidates than needed
        
        return candidate_docs

class ColBERTRetriever:
    """Main ColBERT retrieval system"""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embedding_dim: int = 128,
        use_approximate: bool = True
    ):
        self.encoder = ColBERTEncoder(model_name, embedding_dim)
        self.scorer = LateInteractionScorer()
        
        self.use_approximate = use_approximate
        if use_approximate:
            self.approximate_index = ApproximateLateInteraction()
        
        self.documents: Dict[str, ColBERTDocument] = {}
        self.index_built = False
    
    def add_documents(self, texts: List[str], doc_ids: List[str]):
        """Add documents to the retriever"""
        
        logger.info(f"Encoding {len(texts)} documents with ColBERT...")
        
        for text, doc_id in zip(texts, doc_ids):
            doc = self.encoder.encode_document(text, doc_id)
            self.documents[doc_id] = doc
        
        self.index_built = False
    
    def build_index(self):
        """Build the retrieval index"""
        
        if not self.documents:
            raise ValueError("No documents added")
        
        if self.use_approximate:
            self.approximate_index.build_index(list(self.documents.values()))
        
        self.index_built = True
        logger.info("ColBERT index built successfully")
    
    def search(
        self,
        query_text: str,
        top_k: int = 10,
        rerank_top_k: int = 100
    ) -> List[LateInteractionResult]:
        """Search for relevant documents"""
        
        if not self.index_built:
            self.build_index()
        
        # Encode query
        query = self.encoder.encode_query(query_text)
        
        if self.use_approximate and len(self.documents) > 1000:
            # Use approximate search for large collections
            candidate_doc_ids = self.approximate_index.approximate_search(
                query, self.documents, rerank_top_k
            )
            candidate_docs = [self.documents[doc_id] for doc_id in candidate_doc_ids]
        else:
            # Use exact search for smaller collections
            candidate_docs = list(self.documents.values())
        
        # Score candidates
        results = self.scorer.batch_score(query, candidate_docs)
        
        return results[:top_k]
    
    def explain_result(self, query_text: str, doc_id: str) -> Dict[str, Any]:
        """Explain why a document was retrieved"""
        
        if doc_id not in self.documents:
            raise ValueError(f"Document {doc_id} not found")
        
        query = self.encoder.encode_query(query_text)
        document = self.documents[doc_id]
        
        result = self.scorer.compute_interaction_score(query, document)
        
        # Decode tokens for explanation
        query_tokens = self.encoder.tokenizer.convert_ids_to_tokens(query.token_ids)
        doc_tokens = self.encoder.tokenizer.convert_ids_to_tokens(document.token_ids)
        
        # Build explanation
        explanation = {
            "overall_score": result.score,
            "query_text": query_text,
            "document_content": document.content[:500] + "..." if len(document.content) > 500 else document.content,
            "token_interactions": []
        }
        
        for q_idx, d_idx, score in result.matched_tokens:
            if q_idx < len(query_tokens) and d_idx < len(doc_tokens):
                explanation["token_interactions"].append({
                    "query_token": query_tokens[q_idx],
                    "doc_token": doc_tokens[d_idx],
                    "similarity": score,
                    "query_token_idx": q_idx,
                    "doc_token_idx": d_idx
                })
        
        # Sort by similarity
        explanation["token_interactions"].sort(key=lambda x: x["similarity"], reverse=True)
        
        return explanation

# Global instance
colbert_retriever = ColBERTRetriever()

def get_colbert_retriever() -> ColBERTRetriever:
    """Get the global ColBERT retriever instance"""
    return colbert_retriever