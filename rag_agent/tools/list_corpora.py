"""
Tool for listing all available RAG corpora.
Supports both Vertex AI RAG and local Gemini API-based implementation.
"""

from typing import Dict, List, Union
from ..config import settings


def list_corpora() -> dict:
    """
    List all available RAG corpora.

    Returns:
        dict: A list of available corpora and status, with each corpus containing:
            - resource_name: The full resource name to use with other tools
            - display_name: The human-readable name of the corpus
            - create_time: When the corpus was created
            - update_time: When the corpus was last updated
    """
    try:
        # Use local implementation if not using Vertex AI
        if not settings.use_vertex_ai:
            from ..core.local_rag_store import get_local_rag_store
            
            store = get_local_rag_store()
            corpora = store.list_corpora()
            
            # Process corpus information into expected format
            corpus_info: List[Dict[str, Union[str, int]]] = []
            for corpus in corpora:
                corpus_data: Dict[str, Union[str, int]] = {
                    "resource_name": corpus.name,  # Use corpus name as resource name
                    "display_name": corpus.display_name,
                    "create_time": corpus.created_at,
                    "update_time": corpus.updated_at,
                    "document_count": corpus.document_count,
                    "description": corpus.description,
                }
                corpus_info.append(corpus_data)
            
            return {
                "status": "success",
                "message": f"Found {len(corpus_info)} available corpora",
                "corpora": corpus_info,
                "mode": "local",  # Indicate we're using local mode
            }
        
        # Original Vertex AI implementation
        else:
            from vertexai import rag
            
            # Get the list of corpora
            corpora = rag.list_corpora()

            # Process corpus information into a more usable format
            corpus_info: List[Dict[str, Union[str, int]]] = []
            for corpus in corpora:
                corpus_data: Dict[str, Union[str, int]] = {
                    "resource_name": corpus.name,  # Full resource name for use with other tools
                    "display_name": corpus.display_name,
                    "create_time": (
                        str(corpus.create_time) if hasattr(corpus, "create_time") else ""
                    ),
                    "update_time": (
                        str(corpus.update_time) if hasattr(corpus, "update_time") else ""
                    ),
                }

                corpus_info.append(corpus_data)

            return {
                "status": "success",
                "message": f"Found {len(corpus_info)} available corpora",
                "corpora": corpus_info,
                "mode": "vertex_ai",  # Indicate we're using Vertex AI mode
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error listing corpora: {str(e)}",
            "corpora": [],
        }