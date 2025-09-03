"""
Tool for retrieving detailed information about a specific RAG corpus.
Supports both Vertex AI RAG and local Gemini API-based implementation.
"""

from google.adk.tools.tool_context import ToolContext

from ..config import settings
from .utils import check_corpus_exists, get_corpus_resource_name


def get_corpus_info(
    corpus_name: str,
    tool_context: ToolContext,
) -> dict:
    """
    Get detailed information about a specific RAG corpus, including its files.

    Args:
        corpus_name (str): The full resource name of the corpus to get information about.
                           Preferably use the resource_name from list_corpora results.
        tool_context (ToolContext): The tool context

    Returns:
        dict: Information about the corpus and its files
    """
    try:
        # Use local implementation if not using Vertex AI
        if not settings.use_vertex_ai:
            from ..core.local_rag_store import get_local_rag_store
            
            store = get_local_rag_store()
            
            # Get corpus info
            corpus = store.get_corpus(corpus_name)
            if not corpus:
                return {
                    "status": "error",
                    "message": f"Corpus '{corpus_name}' does not exist",
                    "corpus_name": corpus_name,
                }
            
            # Get documents in corpus
            documents = store.list_documents(corpus_name)
            
            # Format document details
            file_details = []
            for doc in documents:
                file_details.append({
                    "file_id": doc.id,
                    "display_name": doc.display_name,
                    "source_uri": doc.source_uri,
                    "create_time": doc.created_at,
                    "update_time": doc.updated_at,
                    "chunk_count": len(doc.chunks),
                    "metadata": doc.metadata,
                })
            
            return {
                "status": "success",
                "message": f"Successfully retrieved information for corpus '{corpus.display_name}'",
                "corpus_name": corpus_name,
                "corpus_display_name": corpus.display_name,
                "description": corpus.description,
                "created_at": corpus.created_at,
                "updated_at": corpus.updated_at,
                "file_count": len(file_details),
                "files": file_details,
                "mode": "local",
            }
        
        # Original Vertex AI implementation
        else:
            from vertexai import rag
            
            # Check if corpus exists
            if not check_corpus_exists(corpus_name, tool_context):
                return {
                    "status": "error",
                    "message": f"Corpus '{corpus_name}' does not exist",
                    "corpus_name": corpus_name,
                }

            # Get the corpus resource name
            corpus_resource_name = get_corpus_resource_name(corpus_name)

            # Try to get corpus details first
            corpus_display_name = corpus_name  # Default if we can't get actual display name

            # Process file information
            file_details = []
            try:
                # Get the list of files
                files = rag.list_files(corpus_resource_name)
                for rag_file in files:
                    # Get document specific details
                    try:
                        # Extract the file ID from the name
                        file_id = rag_file.name.split("/")[-1]

                        file_info = {
                            "file_id": file_id,
                            "display_name": (
                                rag_file.display_name
                                if hasattr(rag_file, "display_name")
                                else ""
                            ),
                            "source_uri": (
                                rag_file.source_uri
                                if hasattr(rag_file, "source_uri")
                                else ""
                            ),
                            "create_time": (
                                str(rag_file.create_time)
                                if hasattr(rag_file, "create_time")
                                else ""
                            ),
                            "update_time": (
                                str(rag_file.update_time)
                                if hasattr(rag_file, "update_time")
                                else ""
                            ),
                        }

                        file_details.append(file_info)
                    except Exception:
                        # Continue to the next file
                        continue
            except Exception:
                # Continue without file details
                pass

            # Basic corpus info
            return {
                "status": "success",
                "message": f"Successfully retrieved information for corpus '{corpus_display_name}'",
                "corpus_name": corpus_name,
                "corpus_display_name": corpus_display_name,
                "file_count": len(file_details),
                "files": file_details,
                "mode": "vertex_ai",
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting corpus information: {str(e)}",
            "corpus_name": corpus_name,
        }