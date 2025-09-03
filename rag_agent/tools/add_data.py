"""
Tool for adding new data sources to a RAG corpus.
Supports both Vertex AI RAG and local Gemini API-based implementation.
"""

import re
from typing import List

from google.adk.tools.tool_context import ToolContext

from ..config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_REQUESTS_PER_MIN,
    settings,
)
from .utils import check_corpus_exists, get_corpus_resource_name


def add_data(
    corpus_name: str,
    paths: List[str],
    tool_context: ToolContext,
) -> dict:
    """
    Add new data sources to a RAG corpus.

    Args:
        corpus_name (str): The name of the corpus to add data to. If empty, the current corpus will be used.
        paths (List[str]): List of URLs or GCS paths to add to the corpus.
                          Supported formats:
                          - Google Drive: "https://drive.google.com/file/d/{FILE_ID}/view"
                          - Google Docs/Sheets/Slides: "https://docs.google.com/{type}/d/{FILE_ID}/..."
                          - Google Cloud Storage: "gs://{BUCKET}/{PATH}"
                          - Regular URLs: "https://example.com/document.pdf"
                          Example: ["https://drive.google.com/file/d/123", "gs://my_bucket/my_files_dir"]
        tool_context (ToolContext): The tool context

    Returns:
        dict: Information about the added data and status
    """
    # Validate inputs
    if not paths or not all(isinstance(path, str) for path in paths):
        return {
            "status": "error",
            "message": "Invalid paths: Please provide a list of URLs or GCS paths",
            "corpus_name": corpus_name,
            "paths": paths,
        }
    
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
                "corpus_name": corpus_name,
                "paths": paths,
            }
        
        # Process each path
        successful_imports = []
        failed_imports = []
        
        for path in paths:
            try:
                # For local mode, we'll fetch content from URLs
                if path.startswith("http://") or path.startswith("https://"):
                    document = store.import_from_url(corpus_name, path)
                    successful_imports.append({
                        "path": path,
                        "document_id": document.id,
                        "display_name": document.display_name,
                    })
                elif path.startswith("gs://"):
                    # For GCS paths, we would need additional implementation
                    failed_imports.append({
                        "path": path,
                        "error": "GCS paths not yet supported in local mode"
                    })
                else:
                    # For local files
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        document = store.add_document(
                            corpus_name=corpus_name,
                            source_uri=path,
                            content=content,
                            display_name=path.split('/')[-1],
                            metadata={"source_type": "local_file"}
                        )
                        
                        successful_imports.append({
                            "path": path,
                            "document_id": document.id,
                            "display_name": document.display_name,
                        })
                    except Exception as e:
                        failed_imports.append({
                            "path": path,
                            "error": str(e)
                        })
                        
            except Exception as e:
                failed_imports.append({
                    "path": path,
                    "error": str(e)
                })
        
        # Set this as the current corpus if not already set
        if not tool_context.state.get("current_corpus"):
            tool_context.state["current_corpus"] = corpus_name
        
        return {
            "status": "success" if successful_imports else "error",
            "message": f"Imported {len(successful_imports)} document(s) to corpus '{corpus_name}'",
            "corpus_name": corpus_name,
            "successful_imports": successful_imports,
            "failed_imports": failed_imports,
            "mode": "local",
        }
    
    # Original Vertex AI implementation
    else:
        from vertexai import rag
        
        # Check if the corpus exists
        if not check_corpus_exists(corpus_name, tool_context):
            return {
                "status": "error",
                "message": f"Corpus '{corpus_name}' does not exist. Please create it first using the create_corpus tool.",
                "corpus_name": corpus_name,
                "paths": paths,
            }

        # Pre-process paths to validate and convert Google Docs URLs to Drive format if needed
        validated_paths = []
        invalid_paths = []
        conversions = []

        for path in paths:
            if not path or not isinstance(path, str):
                invalid_paths.append(f"{path} (Not a valid string)")
                continue

            # Check for Google Docs/Sheets/Slides URLs and convert them to Drive format
            docs_match = re.match(
                r"https:\/\/docs\.google\.com\/(?:document|spreadsheets|presentation)\/d\/([a-zA-Z0-9_-]+)(?:\/|$)",
                path,
            )
            if docs_match:
                file_id = docs_match.group(1)
                drive_url = f"https://drive.google.com/file/d/{file_id}/view"
                validated_paths.append(drive_url)
                conversions.append(f"{path} → {drive_url}")
                continue

            # Check for valid Drive URL format
            drive_match = re.match(
                r"https:\/\/drive\.google\.com\/(?:file\/d\/|open\?id=)([a-zA-Z0-9_-]+)(?:\/|$)",
                path,
            )
            if drive_match:
                # Normalize to the standard Drive URL format
                file_id = drive_match.group(1)
                drive_url = f"https://drive.google.com/file/d/{file_id}/view"
                validated_paths.append(drive_url)
                if drive_url != path:
                    conversions.append(f"{path} → {drive_url}")
                continue

            # Check for GCS paths
            if path.startswith("gs://"):
                validated_paths.append(path)
                continue

            # If we're here, the path wasn't in a recognized format
            invalid_paths.append(f"{path} (Invalid format)")

        # Check if we have any valid paths after validation
        if not validated_paths:
            return {
                "status": "error",
                "message": "No valid paths provided. Please provide Google Drive URLs or GCS paths.",
                "corpus_name": corpus_name,
                "invalid_paths": invalid_paths,
            }

        try:
            # Get the corpus resource name
            corpus_resource_name = get_corpus_resource_name(corpus_name)

            # Set up chunking configuration
            transformation_config = rag.TransformationConfig(
                chunking_config=rag.ChunkingConfig(
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                ),
            )

            # Import files to the corpus
            import_result = rag.import_files(
                corpus_resource_name,
                validated_paths,
                transformation_config=transformation_config,
                max_embedding_requests_per_min=DEFAULT_EMBEDDING_REQUESTS_PER_MIN,
            )

            # Set this as the current corpus if not already set
            if not tool_context.state.get("current_corpus"):
                tool_context.state["current_corpus"] = corpus_name

            # Build the success message
            conversion_msg = ""
            if conversions:
                conversion_msg = " (Converted Google Docs URLs to Drive format)"

            return {
                "status": "success",
                "message": f"Successfully added {import_result.imported_rag_files_count} file(s) to corpus '{corpus_name}'{conversion_msg}",
                "corpus_name": corpus_name,
                "files_added": import_result.imported_rag_files_count,
                "paths": validated_paths,
                "invalid_paths": invalid_paths,
                "conversions": conversions,
                "mode": "vertex_ai",
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Error adding data to corpus: {str(e)}",
                "corpus_name": corpus_name,
                "paths": paths,
                "mode": "vertex_ai",
            }