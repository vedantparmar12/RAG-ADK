"""
Simple Meilisearch client wrapper for FTS indexing/search in RAG.
"""

from typing import List, Dict, Any, Optional
import requests
import logging

logger = logging.getLogger(__name__)


class MeiliClient:
    def __init__(self, url: str = "http://localhost:7700", api_key: Optional[str] = None):
        self.base = url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def ensure_index(self, index: str, primary_key: str = "id"):
        try:
            # Try to create; if exists, it's ok
            r = requests.post(
                f"{self.base}/indexes",
                json={"uid": index, "primaryKey": primary_key},
                headers=self.headers,
                timeout=10,
            )
            # 409 if exists; ignore
        except Exception as e:
            logger.warning(f"Meilisearch ensure_index error for {index}: {e}")

    def upsert_documents(self, index: str, docs: List[Dict[str, Any]]):
        try:
            r = requests.post(
                f"{self.base}/indexes/{index}/documents",
                json=docs,
                headers={**self.headers, "Content-Type": "application/json"},
                timeout=30,
            )
            r.raise_for_status()
        except Exception as e:
            logger.error(f"Meilisearch upsert failed: {e}")

    def search(self, index: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            r = requests.post(
                f"{self.base}/indexes/{index}/search",
                json={"q": query, "limit": limit},
                headers=self.headers,
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            hits = data.get("hits", [])
            # Normalize
            results = []
            for h in hits:
                results.append(
                    {
                        "id": h.get("id"),
                        "text": h.get("text"),
                        "document_id": h.get("document_id"),
                        "chunk_index": h.get("chunk_index"),
                        "metadata": h.get("metadata", {}),
                        "score": float(h.get("_rankingScore", 0.0)),
                    }
                )
            return results
        except Exception as e:
            logger.error(f"Meilisearch query failed: {e}")
            return []
    
    def delete_documents_by_filter(self, index: str, filter_expr: str) -> bool:
        """Delete documents matching a filter expression."""
        try:
            r = requests.post(
                f"{self.base}/indexes/{index}/documents/delete",
                json={"filter": filter_expr},
                headers={**self.headers, "Content-Type": "application/json"},
                timeout=30,
            )
            r.raise_for_status()
            logger.info(f"Deleted documents from index '{index}' with filter: {filter_expr}")
            return True
        except Exception as e:
            logger.error(f"Meilisearch delete by filter failed: {e}")
            return False
    
    def delete_document_by_id(self, index: str, doc_id: str) -> bool:
        """Delete a single document by ID."""
        try:
            r = requests.delete(
                f"{self.base}/indexes/{index}/documents/{doc_id}",
                headers=self.headers,
                timeout=10,
            )
            r.raise_for_status()
            logger.info(f"Deleted document '{doc_id}' from index '{index}'")
            return True
        except Exception as e:
            logger.error(f"Meilisearch delete by ID failed: {e}")
            return False
    
    def delete_documents_by_document_id(self, index: str, document_id: str) -> bool:
        """Delete all chunks belonging to a document_id."""
        return self.delete_documents_by_filter(index, f"document_id = '{document_id}'")

