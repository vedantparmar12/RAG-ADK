"""
Tool for querying RAG corpora and retrieving relevant information.
Supports both Vertex AI RAG and local Gemini API-based implementation.
"""

import logging

from google.adk.tools.tool_context import ToolContext

from ..config import (
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_TOP_K,
    settings,
)
from .utils import check_corpus_exists, get_corpus_resource_name


def rag_query(
    corpus_name: str,
    query: str,
    tool_context: ToolContext,
) -> dict:
    """
    Query a RAG corpus with a user question and return relevant information.

    Args:
        corpus_name (str): The name of the corpus to query. If empty, the current corpus will be used.
                          Preferably use the resource_name from list_corpora results.
        query (str): The text query to search for in the corpus
        tool_context (ToolContext): The tool context

    Returns:
        dict: The query results and status
    """
    try:
        # Use local implementation if not using Vertex AI
        if not settings.use_vertex_ai:
            from ..core.local_rag_store import get_local_rag_store
            
            store = get_local_rag_store()
            
            # Check if corpus exists
            corpus = store.get_corpus(corpus_name)
            if not corpus:
                return {
                    "status": "error",
                    "message": f"Corpus '{corpus_name}' does not exist. Please create it first using the create_corpus tool.",
                    "query": query,
                    "corpus_name": corpus_name,
                }
            
            # Perform query using local store
            try:
                result = store.query(
                    corpus_name=corpus_name,
                    query=query,
                    top_k=DEFAULT_TOP_K
                )
                
                # Format results to match expected output
                formatted_results = []
                for source in result.get("sources", []):
                    formatted_results.append({
                        "source_uri": source.get("source_uri", ""),
                        "source_name": source.get("source_name", ""),
                        "text": source.get("text", ""),
                        "score": source.get("score", 0.0),
                    })
                
                return {
                    "status": "success",
                    "message": f"Successfully queried corpus '{corpus_name}'",
                    "query": query,
                    "corpus_name": corpus_name,
                    "results": formatted_results,
                    "results_count": len(formatted_results),
                    "answer": result.get("answer", ""),  # Include generated answer
                }
                
            except Exception as e:
                error_msg = f"Error querying local corpus: {str(e)}"
                logging.error(error_msg)
                return {
                    "status": "error",
                    "message": error_msg,
                    "query": query,
                    "corpus_name": corpus_name,
                }
        
        # Original Vertex AI implementation
        else:
            from vertexai import rag
            
            # Check if the corpus exists
            if not check_corpus_exists(corpus_name, tool_context):
                return {
                    "status": "error",
                    "message": f"Corpus '{corpus_name}' does not exist. Please create it first using the create_corpus tool.",
                    "query": query,
                    "corpus_name": corpus_name,
                }

            # Get the corpus resource name
            corpus_resource_name = get_corpus_resource_name(corpus_name)

            # Configure retrieval parameters
            rag_retrieval_config = rag.RagRetrievalConfig(
                top_k=DEFAULT_TOP_K,
                filter=rag.Filter(vector_distance_threshold=DEFAULT_DISTANCE_THRESHOLD),
            )

            # Perform the query
            print("Performing retrieval query...")
            response = rag.retrieval_query(
                rag_resources=[
                    rag.RagResource(
                        rag_corpus=corpus_resource_name,
                    )
                ],
                text=query,
                rag_retrieval_config=rag_retrieval_config,
            )

            # Process the response into a more usable format
            results = []
            if hasattr(response, "contexts") and response.contexts:
                for ctx_group in response.contexts.contexts:
                    result = {
                        "source_uri": (
                            ctx_group.source_uri if hasattr(ctx_group, "source_uri") else ""
                        ),
                        "source_name": (
                            ctx_group.source_display_name
                            if hasattr(ctx_group, "source_display_name")
                            else ""
                        ),
                        "text": ctx_group.text if hasattr(ctx_group, "text") else "",
                        "score": ctx_group.score if hasattr(ctx_group, "score") else 0.0,
                    }
                    results.append(result)

            # If we didn't find any results
            if not results:
                return {
                    "status": "warning",
                    "message": f"No results found in corpus '{corpus_name}' for query: '{query}'",
                    "query": query,
                    "corpus_name": corpus_name,
                    "results": [],
                    "results_count": 0,
                }

            return {
                "status": "success",
                "message": f"Successfully queried corpus '{corpus_name}'",
                "query": query,
                "corpus_name": corpus_name,
                "results": results,
                "results_count": len(results),
            }

    except Exception as e:
        error_msg = f"Error querying corpus: {str(e)}"
        logging.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "query": query,
            "corpus_name": corpus_name,
        }