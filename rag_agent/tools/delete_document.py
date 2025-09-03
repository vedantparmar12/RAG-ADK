"""
Tool for deleting a specific document from a RAG corpus.
Supports both Vertex AI RAG and local Gemini API-based implementation.
"""

from google.adk.tools.tool_context import ToolContext

from ..config import settings
from .utils import check_corpus_exists, get_corpus_resource_name


def delete_document(
    corpus_name: str,
    document_id: str,
    tool_context: ToolContext,
) -> dict:
    """
    Delete a specific document from a RAG corpus.

    Args:
        corpus_name (str): The full resource name of the corpus containing the document.
                          Preferably use the resource_name from list_corpora results.
        document_id (str): The ID of the specific document/file to delete. This can be
                          obtained from get_corpus_info results.
        tool_context (ToolContext): The tool context

    Returns:
        dict: Status information about the deletion operation
    """
    # Use local implementation if not using Vertex AI
    if not settings.use_vertex_ai:
        from ..core.local_rag_store import get_local_rag_store
        
        store = get_local_rag_store()
        
        # Check if corpus exists
        corpus = store.get_corpus(corpus_name)
        if not corpus:
            return {
                "status": "error",
                "message": f"Corpus '{corpus_name}' does not exist",
                "corpus_name": corpus_name,
                "document_id": document_id,
            }
        
        try:
            # Delete the document
            success = store.delete_document(corpus_name, document_id)
            
            if success:
                return {
                    "status": "success",
                    "message": f"Successfully deleted document '{document_id}' from corpus '{corpus_name}'",
                    "corpus_name": corpus_name,
                    "document_id": document_id,
                    "mode": "local",
                }
            else:
                return {
                    "status": "error",
                    "message": f"Document '{document_id}' not found in corpus '{corpus_name}'",
                    "corpus_name": corpus_name,
                    "document_id": document_id,
                    "mode": "local",
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error deleting document: {str(e)}",
                "corpus_name": corpus_name,
                "document_id": document_id,
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
                "document_id": document_id,
            }

        try:
            # Get the corpus resource name
            corpus_resource_name = get_corpus_resource_name(corpus_name)

            # Delete the document
            rag_file_path = f"{corpus_resource_name}/ragFiles/{document_id}"
            rag.delete_file(rag_file_path)

            return {
                "status": "success",
                "message": f"Successfully deleted document '{document_id}' from corpus '{corpus_name}'",
                "corpus_name": corpus_name,
                "document_id": document_id,
                "mode": "vertex_ai",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error deleting document: {str(e)}",
                "corpus_name": corpus_name,
                "document_id": document_id,
                "mode": "vertex_ai",
            }