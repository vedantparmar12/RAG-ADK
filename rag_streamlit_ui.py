import streamlit as st
import requests
import json
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
import time
import os

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Enhanced Vertex AI RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-message {
        padding: 10px;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-message {
        padding: 10px;
        border-radius: 5px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.query_history = []
    st.session_state.current_corpus = None
    st.session_state.corpus_list = []
    st.session_state.api_status = "unknown"

# Helper functions
def check_api_health():
    """Check if API is running and get system status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if response.status_code == 200:
            st.session_state.api_status = "online"
            return True, response.json()
        else:
            st.session_state.api_status = "error"
            return False, {"error": f"Status code: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        st.session_state.api_status = "offline"
        return False, {"error": str(e)}

def make_api_request(endpoint: str, method: str = "GET", data: Optional[Dict] = None, timeout: int = 30):
    """Make API request with comprehensive error handling"""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        headers = {"Content-Type": "application/json"}
        
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=timeout)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=timeout)
        else:
            return False, {"error": f"Unsupported method: {method}"}
        
        # Handle successful responses
        if response.status_code == 200:
            return True, response.json()
        else:
            # Try to parse error response
            try:
                error_data = response.json()
                return False, {
                    "status_code": response.status_code,
                    "detail": error_data.get("detail", str(error_data))
                }
            except:
                return False, {
                    "status_code": response.status_code,
                    "detail": response.text or "Unknown error"
                }
                
    except requests.exceptions.Timeout:
        return False, {"error": "Request timed out. The operation might still be processing."}
    except requests.exceptions.ConnectionError:
        return False, {"error": f"Cannot connect to API at {API_BASE_URL}. Please ensure the API is running."}
    except Exception as e:
        return False, {"error": f"Unexpected error: {str(e)}"}

def format_timestamp(timestamp):
    """Format timestamp for display"""
    if isinstance(timestamp, str):
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    else:
        dt = timestamp
    return dt.strftime("%Y-%m-%d %H:%M:%S")

# Sidebar
with st.sidebar:
    st.title("🤖 RAG Agent Control Panel")
    
    # API Status
    api_healthy, health_data = check_api_health()
    
    if api_healthy:
        st.success("✅ API Connected")
        with st.expander("System Details", expanded=False):
            components = health_data.get("components", {})
            settings = health_data.get("settings", {})
            
            st.markdown("**Components Status:**")
            st.text(f"• RAG System: {components.get('rag_system', 'unknown')}")
            st.text(f"• Caching: {components.get('caching', 'unknown')}")
            st.text(f"• ColBERT: {components.get('colbert', 'unknown')}")
            
            if settings:
                st.markdown("**Settings:**")
                for key, value in settings.items():
                    st.text(f"• {key}: {value}")
    else:
        st.error("❌ API Disconnected")
        error_info = health_data.get("error", "Unknown error")
        st.warning(f"Error: {error_info}")
        
        with st.expander("Troubleshooting Steps"):
            st.markdown("""
            1. Ensure the API is running:
               ```bash
               python main.py
               # or
               uvicorn main:app --reload
               ```
            2. Check if the API URL is correct:
               - Current: `{}`
            3. Verify no firewall is blocking port 8000
            """.format(API_BASE_URL))
    
    st.markdown("---")
    
    # Navigation Menu
    page = st.selectbox(
        "Select Feature",
        [
            "🏠 Home",
            "🔍 Query Interface",
            "📚 Corpus Management",
            "📄 Document Indexing",
            "📊 Evaluation & Testing",
            "💾 Cache Management",
            "📈 Analytics Dashboard"
        ]
    )

# Main content area
if page == "🏠 Home":
    st.title("🚀 Enhanced Vertex AI RAG Agent")
    st.markdown("### Welcome to the Advanced Retrieval-Augmented Generation System")
    
    # Quick stats
    if api_healthy:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("API Status", "Online" if api_healthy else "Offline")
        with col2:
            st.metric("Active Queries", len(st.session_state.query_history))
        with col3:
            st.metric("Current Corpus", st.session_state.current_corpus or "None")
        with col4:
            st.metric("Cache Status", health_data.get("components", {}).get("caching", "unknown"))
    
    # Feature overview
    st.markdown("---")
    st.markdown("### 🎯 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔍 Advanced Retrieval**
        - Hybrid search (dense + sparse)
        - ColBERT neural retrieval
        - Semantic reranking
        - Query expansion
        
        **📊 Performance Optimization**
        - Redis-based caching
        - Context pruning
        - Parallel processing
        - Token optimization
        """)
    
    with col2:
        st.markdown("""
        **📄 Document Processing**
        - Layout-aware parsing
        - Intelligent chunking
        - Late chunking support
        - Multi-format support
        
        **🛠️ Developer Tools**
        - A/B testing framework
        - Performance analytics
        - API monitoring
        - Batch processing
        """)
    
    # Quick start guide
    with st.expander("⚡ Quick Start Guide", expanded=True):
        st.markdown("""
        1. **Create a Corpus** - Go to 📚 Corpus Management
        2. **Add Documents** - Use 📄 Document Indexing
        3. **Query Your Data** - Visit 🔍 Query Interface
        
        **Example Query Patterns:**
        - "What is the main purpose of this document?"
        - "Summarize the key findings about X"
        - "Compare the approaches mentioned for Y"
        """)

elif page == "🔍 Query Interface":
    st.title("🔍 Advanced Query Interface")
    
    # Corpus selection
    col1, col2 = st.columns([3, 1])
    with col1:
        corpus_id = st.text_input(
            "Corpus ID",
            value=st.session_state.current_corpus or "",
            placeholder="Enter corpus ID (e.g., corpus-123)",
            help="The ID of the corpus to search in"
        )
    with col2:
        if st.button("📋 Set as Default", use_container_width=True):
            st.session_state.current_corpus = corpus_id
            st.success("Default corpus updated!")
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="What would you like to know about your documents?",
        help="Enter a natural language question"
    )
    
    # Configuration tabs
    tab1, tab2, tab3 = st.tabs(["⚙️ Retrieval Settings", "🚀 Performance", "📊 Output Options"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            retrieval_strategy = st.selectbox(
                "Retrieval Strategy",
                ["hybrid", "dense", "sparse", "colbert"],
                help="Hybrid combines dense and sparse retrieval for best results"
            )
            
            top_k = st.slider(
                "Number of Results (Top-K)",
                min_value=1,
                max_value=50,
                value=10,
                help="Number of document chunks to retrieve"
            )
        
        with col2:
            if retrieval_strategy == "hybrid":
                alpha = st.slider(
                    "Hybrid Balance (α)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="0 = pure sparse, 1 = pure dense"
                )
            else:
                alpha = 0.5
            
            enable_reranking = st.checkbox(
                "Enable Semantic Reranking",
                value=True,
                help="Use advanced reranking model for better results"
            )
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            use_cache = st.checkbox(
                "Use Cache",
                value=True,
                help="Enable Redis caching for faster repeated queries"
            )
            
            enable_pruning = st.checkbox(
                "Enable Context Pruning",
                value=True,
                help="Remove less relevant context to improve generation"
            )
        
        with col2:
            max_context_tokens = st.number_input(
                "Max Context Tokens",
                min_value=1024,
                max_value=32768,
                value=8192,
                step=1024,
                help="Maximum tokens to include in generation context"
            )
            
            enable_query_expansion = st.checkbox(
                "Enable Query Expansion",
                value=False,
                help="Expand query with synonyms and related terms"
            )
    
    with tab3:
        return_chunks = st.checkbox(
            "Show Retrieved Chunks",
            value=False,
            help="Display the document chunks used to generate the answer"
        )
        
        show_metadata = st.checkbox(
            "Show Detailed Metadata",
            value=True,
            help="Display performance metrics and system information"
        )
    
    # Submit query
    if st.button("🚀 Submit Query", type="primary", use_container_width=True):
        if not query:
            st.warning("⚠️ Please enter a query")
        elif not corpus_id:
            st.warning("⚠️ Please specify a corpus ID")
        else:
            # Prepare request
            query_data = {
                "query": query,
                "corpus_id": corpus_id,
                "top_k": top_k,
                "retrieval_strategy": retrieval_strategy,
                "alpha": alpha,
                "use_cache": use_cache,
                "enable_pruning": enable_pruning,
                "max_context_tokens": max_context_tokens,
                "enable_reranking": enable_reranking,
                "enable_query_expansion": enable_query_expansion,
                "return_chunks": return_chunks
            }
            
            # Show progress
            with st.spinner("🔄 Processing your query..."):
                start_time = time.time()
                success, response = make_api_request("/query", "POST", query_data, timeout=60)
                end_time = time.time()
            
            if success:
                # Display answer
                st.success("✅ Query completed successfully!")
                
                answer_col, metric_col = st.columns([3, 1])
                
                with answer_col:
                    st.markdown("### 💡 Answer")
                    answer = response.get("answer", "No answer generated")
                    st.markdown(answer)
                
                with metric_col:
                    confidence = response.get("confidence", 0.0)
                    st.metric("Confidence Score", f"{confidence:.1%}")
                
                # Performance metrics
                if show_metadata:
                    metadata = response.get("metadata", {})
                    
                    st.markdown("---")
                    st.markdown("### 📊 Performance Metrics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Total Time",
                            f"{end_time - start_time:.2f}s"
                        )
                    
                    with col2:
                        retrieval_time = metadata.get("retrieval_time", 0)
                        st.metric(
                            "Retrieval Time",
                            f"{retrieval_time:.2f}s"
                        )
                    
                    with col3:
                        generation_time = metadata.get("generation_time", 0)
                        st.metric(
                            "Generation Time",
                            f"{generation_time:.2f}s"
                        )
                    
                    with col4:
                        cache_hit = metadata.get("cache_hit", False)
                        st.metric(
                            "Cache Status",
                            "HIT ✅" if cache_hit else "MISS ❌"
                        )
                    
                    # Pruning statistics
                    if enable_pruning and "pruning_stats" in metadata:
                        with st.expander("🔍 Context Pruning Details"):
                            pruning = metadata["pruning_stats"]
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Original Tokens",
                                    pruning.get("original_tokens", "N/A")
                                )
                            with col2:
                                st.metric(
                                    "Pruned Tokens",
                                    pruning.get("pruned_tokens", "N/A")
                                )
                            with col3:
                                reduction = pruning.get("reduction_percentage", 0)
                                st.metric(
                                    "Reduction",
                                    f"{reduction:.1f}%"
                                )
                
                # Retrieved chunks
                if return_chunks and "chunks" in response:
                    st.markdown("---")
                    st.markdown("### 📄 Retrieved Chunks")
                    
                    chunks = response["chunks"]
                    st.info(f"Retrieved {len(chunks)} relevant chunks")
                    
                    for i, chunk in enumerate(chunks, 1):
                        with st.expander(f"Chunk {i} - Score: {chunk.get('score', 0):.3f}"):
                            st.text(chunk.get("text", ""))
                            if "metadata" in chunk:
                                st.json(chunk["metadata"])
                
                # Add to history
                st.session_state.query_history.append({
                    "timestamp": datetime.now(),
                    "query": query,
                    "corpus_id": corpus_id,
                    "answer": answer[:200] + "..." if len(answer) > 200 else answer,
                    "confidence": confidence,
                    "retrieval_strategy": retrieval_strategy,
                    "response_time": end_time - start_time,
                    "cache_hit": metadata.get("cache_hit", False)
                })
                
            else:
                st.error("❌ Query failed")
                error_msg = response.get("detail", response.get("error", "Unknown error"))
                st.error(f"Error: {error_msg}")
                
                if "status_code" in response:
                    st.info(f"HTTP Status Code: {response['status_code']}")

elif page == "📚 Corpus Management":
    st.title("📚 Corpus Management")
    
    tab1, tab2, tab3 = st.tabs(["➕ Create Corpus", "📊 View Stats", "📋 List Corpora"])
    
    with tab1:
        st.markdown("### Create New Corpus")
        
        with st.form("create_corpus_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                corpus_name = st.text_input(
                    "Corpus Name",
                    placeholder="my-knowledge-base",
                    help="A unique name for your corpus"
                )
                
                corpus_description = st.text_area(
                    "Description",
                    placeholder="A corpus containing technical documentation...",
                    help="Describe the purpose and content of this corpus"
                )
                
                indexing_strategy = st.selectbox(
                    "Indexing Strategy",
                    ["hybrid", "dense", "sparse"],
                    help="Hybrid is recommended for most use cases"
                )
            
            with col2:
                chunk_size = st.slider(
                    "Chunk Size (tokens)",
                    min_value=128,
                    max_value=2048,
                    value=512,
                    step=64,
                    help="Size of text chunks for indexing"
                )
                
                chunk_overlap = st.slider(
                    "Chunk Overlap (tokens)",
                    min_value=0,
                    max_value=256,
                    value=64,
                    step=16,
                    help="Overlap between consecutive chunks"
                )
                
                st.markdown("**Advanced Features**")
                enable_layout_parser = st.checkbox(
                    "Enable Layout Parser",
                    value=True,
                    help="Use AI to understand document structure"
                )
                
                enable_late_chunking = st.checkbox(
                    "Enable Late Chunking",
                    value=True,
                    help="Preserve context during chunking"
                )
                
                enable_colbert = st.checkbox(
                    "Enable ColBERT",
                    value=True,
                    help="Use neural retrieval for better accuracy"
                )
            
            submitted = st.form_submit_button("Create Corpus", type="primary", use_container_width=True)
            
            if submitted:
                if not corpus_name:
                    st.error("❌ Please provide a corpus name")
                elif not corpus_description:
                    st.error("❌ Please provide a description")
                else:
                    corpus_data = {
                        "name": corpus_name,
                        "description": corpus_description,
                        "indexing_strategy": indexing_strategy,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "enable_layout_parser": enable_layout_parser,
                        "enable_late_chunking": enable_late_chunking,
                        "enable_colbert": enable_colbert
                    }
                    
                    with st.spinner("Creating corpus..."):
                        success, response = make_api_request("/corpus", "POST", corpus_data)
                    
                    if success:
                        st.success(f"✅ Corpus '{corpus_name}' created successfully!")
                        if "corpus_id" in response:
                            st.info(f"Corpus ID: {response['corpus_id']}")
                            st.session_state.current_corpus = response['corpus_id']
                        st.json(response)
                    else:
                        st.error("❌ Failed to create corpus")
                        st.error(f"Error: {response.get('detail', response)}")
    
    with tab2:
        st.markdown("### Corpus Statistics")
        
        stats_corpus_id = st.text_input(
            "Corpus ID",
            value=st.session_state.current_corpus or "",
            placeholder="Enter corpus ID"
        )
        
        if st.button("📊 Get Statistics", use_container_width=True):
            if stats_corpus_id:
                with st.spinner("Fetching corpus statistics..."):
                    success, response = make_api_request(f"/corpus/{stats_corpus_id}/stats")
                
                if success:
                    st.success("✅ Statistics retrieved successfully")
                    
                    # Display key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Documents",
                            response.get("document_count", 0)
                        )
                    
                    with col2:
                        st.metric(
                            "Total Chunks",
                            response.get("chunk_count", 0)
                        )
                    
                    with col3:
                        st.metric(
                            "Index Size",
                            response.get("index_size", "N/A")
                        )
                    
                    with col4:
                        st.metric(
                            "Avg Chunk Size",
                            response.get("avg_chunk_size", "N/A")
                        )
                    
                    # Additional details
                    with st.expander("📋 Full Statistics"):
                        st.json(response)
                else:
                    st.error("❌ Failed to retrieve statistics")
                    st.error(f"Error: {response.get('detail', response)}")
            else:
                st.warning("⚠️ Please enter a corpus ID")
    
    with tab3:
        st.markdown("### List All Corpora")
        st.info("This feature would list all available corpora. Implementation depends on API endpoint availability.")

elif page == "📄 Document Indexing":
    st.title("📄 Document Indexing")
    
    corpus_id = st.text_input(
        "Target Corpus ID",
        value=st.session_state.current_corpus or "",
        placeholder="Enter the corpus ID where documents will be added"
    )
    
    st.markdown("### Document Sources")
    st.info("Enter document URIs below. Supported formats: GCS paths (gs://), Google Drive links, HTTP URLs")
    
    # Document input methods
    tab1, tab2 = st.tabs(["📝 Manual Entry", "📁 Batch Upload"])
    
    with tab1:
        doc_uris = st.text_area(
            "Document URIs (one per line)",
            height=200,
            placeholder="gs://my-bucket/documents/doc1.pdf\ngs://my-bucket/documents/doc2.pdf\nhttps://example.com/document.pdf",
            help="Enter each document URI on a new line"
        )
    
    with tab2:
        st.markdown("### Batch Upload from CSV")
        uploaded_file = st.file_uploader(
            "Upload CSV file with URIs",
            type=['csv'],
            help="CSV should have a column named 'uri' containing document paths"
        )
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            if 'uri' in df.columns:
                doc_uris = '\n'.join(df['uri'].tolist())
                st.success(f"✅ Loaded {len(df)} document URIs from CSV")
            else:
                st.error("❌ CSV must contain a 'uri' column")
                doc_uris = ""
    
    # Processing options
    st.markdown("### Processing Options")
    col1, col2 = st.columns(2)
    
    with col1:
        use_layout_parser = st.checkbox(
            "Use Layout Parser",
            value=True,
            help="AI-powered document structure understanding"
        )
    
    with col2:
        enable_late_chunking = st.checkbox(
            "Enable Late Chunking",
            value=True,
            help="Better context preservation during chunking"
        )
    
    # Submit documents
    if st.button("📤 Index Documents", type="primary", use_container_width=True):
        if not corpus_id:
            st.warning("⚠️ Please specify a corpus ID")
        elif not doc_uris or not doc_uris.strip():
            st.warning("⚠️ Please enter at least one document URI")
        else:
            # Parse URIs
            uris = [uri.strip() for uri in doc_uris.strip().split('\n') if uri.strip()]
            
            if uris:
                st.info(f"📋 Processing {len(uris)} documents...")
                
                document_data = {
                    "uris": uris,
                    "use_layout_parser": use_layout_parser,
                    "enable_late_chunking": enable_late_chunking
                }
                
                with st.spinner("Submitting documents for indexing..."):
                    success, response = make_api_request(
                        f"/corpus/{corpus_id}/documents",
                        "POST",
                        document_data,
                        timeout=60
                    )
                
                if success:
                    st.success("✅ Documents submitted for processing!")
                    st.json(response)
                    st.info("📌 Documents are being processed in the background. Check corpus stats to monitor progress.")
                else:
                    st.error("❌ Failed to submit documents")
                    st.error(f"Error: {response.get('detail', response)}")

elif page == "📊 Evaluation & Testing":
    st.title("📊 A/B Configuration Testing")
    
    st.markdown("""
    Compare different RAG configurations to find the optimal settings for your use case.
    """)
    
    # Test configuration
    corpus_id = st.text_input(
        "Corpus ID for Testing",
        value=st.session_state.current_corpus or "",
        placeholder="Enter corpus ID"
    )
    
    # Test queries
    st.markdown("### Test Queries")
    test_queries_text = st.text_area(
        "Enter test queries (one per line)",
        height=150,
        placeholder="What is machine learning?\nExplain neural networks\nHow does RAG work?",
        help="These queries will be run against both configurations"
    )
    
    # Configuration comparison
    st.markdown("### Configuration Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Configuration A")
        config_a = {}
        config_a["retrieval_strategy"] = st.selectbox(
            "Retrieval Strategy",
            ["hybrid", "dense", "sparse", "colbert"],
            key="config_a_strategy"
        )
        config_a["top_k"] = st.slider(
            "Top K",
            1, 50, 10,
            key="config_a_topk"
        )
        config_a["enable_reranking"] = st.checkbox(
            "Enable Reranking",
            value=True,
            key="config_a_rerank"
        )
        if config_a["retrieval_strategy"] == "hybrid":
            config_a["alpha"] = st.slider(
                "Alpha",
                0.0, 1.0, 0.5,
                key="config_a_alpha"
            )
    
    with col2:
        st.markdown("#### Configuration B")
        config_b = {}
        config_b["retrieval_strategy"] = st.selectbox(
            "Retrieval Strategy",
            ["hybrid", "dense", "sparse", "colbert"],
            key="config_b_strategy"
        )
        config_b["top_k"] = st.slider(
            "Top K",
            1, 50, 10,
            key="config_b_topk"
        )
        config_b["enable_reranking"] = st.checkbox(
            "Enable Reranking",
            value=True,
            key="config_b_rerank"
        )
        if config_b["retrieval_strategy"] == "hybrid":
            config_b["alpha"] = st.slider(
                "Alpha",
                0.0, 1.0, 0.5,
                key="config_b_alpha"
            )
    
    # Run evaluation
    if st.button("🔬 Run A/B Test", type="primary", use_container_width=True):
        if not corpus_id:
            st.warning("⚠️ Please specify a corpus ID")
        elif not test_queries_text.strip():
            st.warning("⚠️ Please enter test queries")
        else:
            test_queries = [q.strip() for q in test_queries_text.strip().split('\n') if q.strip()]
            
            eval_data = {
                "test_queries": test_queries,
                "corpus_id": corpus_id,
                "config_a": config_a,
                "config_b": config_b
            }
            
            with st.spinner(f"Running A/B test with {len(test_queries)} queries..."):
                success, response = make_api_request("/evaluate", "POST", eval_data, timeout=120)
            
            if success:
                st.success("✅ Evaluation completed!")
                
                # Display comparison results
                if "comparison" in response:
                    st.markdown("### 📊 Results Summary")
                    
                    comp = response["comparison"]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Configuration A**")
                        st.metric(
                            "Avg Retrieval Time",
                            f"{comp.get('avg_retrieval_time_a', 0):.3f}s"
                        )
                        st.metric(
                            "Avg Quality Score",
                            f"{comp.get('avg_quality_a', 0):.2%}"
                        )
                    
                    with col2:
                        st.markdown("**Configuration B**")
                        st.metric(
                            "Avg Retrieval Time",
                            f"{comp.get('avg_retrieval_time_b', 0):.3f}s"
                        )
                        st.metric(
                            "Avg Quality Score",
                            f"{comp.get('avg_quality_b', 0):.2%}"
                        )
                    
                    # Winner analysis
                    st.markdown("### 🏆 Winner Analysis")
                    
                    time_winner = "A" if comp.get('avg_retrieval_time_a', float('inf')) < comp.get('avg_retrieval_time_b', float('inf')) else "B"
                    quality_winner = "A" if comp.get('avg_quality_a', 0) > comp.get('avg_quality_b', 0) else "B"
                    
                    st.info(f"⏱️ Faster: Configuration {time_winner}")
                    st.info(f"🎯 Better Quality: Configuration {quality_winner}")
                
                # Detailed results
                with st.expander("📋 Detailed Results"):
                    st.json(response)
            else:
                st.error("❌ Evaluation failed")
                st.error(f"Error: {response.get('detail', response)}")

elif page == "💾 Cache Management":
    st.title("💾 Cache Management")
    
    tab1, tab2 = st.tabs(["📊 Cache Statistics", "🔧 Cache Operations"])
    
    with tab1:
        st.markdown("### Cache Performance Metrics")
        
        if st.button("🔄 Refresh Statistics", use_container_width=True):
            with st.spinner("Fetching cache statistics..."):
                success, response = make_api_request("/cache/stats")
            
            if success:
                if response.get("cache_enabled", True):
                    st.success("✅ Cache is enabled and running")
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        hit_rate = response.get("hit_rate", 0)
                        st.metric(
                            "Hit Rate",
                            f"{hit_rate:.1%}"
                        )
                    
                    with col2:
                        st.metric(
                            "Total Hits",
                            response.get("hits", 0)
                        )
                    
                    with col3:
                        st.metric(
                            "Total Misses",
                            response.get("misses", 0)
                        )
                    
                    with col4:
                        total_requests = response.get("hits", 0) + response.get("misses", 0)
                        st.metric(
                            "Total Requests",
                            total_requests
                        )
                    
                    # Additional stats
                    with st.expander("📋 Detailed Cache Statistics"):
                        st.json(response)
                else:
                    st.warning("⚠️ Cache is disabled")
            else:
                st.error("❌ Failed to fetch cache statistics")
                st.error(f"Error: {response.get('detail', response)}")
    
    with tab2:
        st.markdown("### Cache Operations")
        
        # Cache invalidation
        st.markdown("#### 🗑️ Invalidate Cache")
        
        invalidate_corpus_id = st.text_input(
            "Corpus ID to invalidate",
            placeholder="Enter corpus ID"
        )
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("🗑️ Invalidate", type="secondary", use_container_width=True):
                if invalidate_corpus_id:
                    with st.spinner("Invalidating cache..."):
                        success, response = make_api_request(
                            f"/cache/invalidate/{invalidate_corpus_id}",
                            "POST"
                        )
                    
                    if success:
                        st.success(f"✅ Cache invalidated for corpus: {invalidate_corpus_id}")
                        st.json(response)
                    else:
                        st.error("❌ Failed to invalidate cache")
                        st.error(f"Error: {response.get('detail', response)}")
                else:
                    st.warning("⚠️ Please enter a corpus ID")
        
        with col2:
            st.info("💡 Invalidating cache will clear all cached queries for the specified corpus")

elif page == "📈 Analytics Dashboard":
    st.title("📈 Analytics Dashboard")
    
    # Query history analytics
    if st.session_state.query_history:
        st.markdown("### 📊 Query Analytics")
        
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.query_history)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", len(df))
        
        with col2:
            avg_confidence = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        with col3:
            avg_response_time = df['response_time'].mean()
            st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
        
        with col4:
            cache_hits = df['cache_hit'].sum()
            cache_rate = cache_hits / len(df) if len(df) > 0 else 0
            st.metric("Cache Hit Rate", f"{cache_rate:.1%}")
        
        # Charts
        st.markdown("### 📈 Trends")
        
        # Response time over time
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_sorted = df.sort_values('timestamp')
        
        st.line_chart(
            df_sorted.set_index('timestamp')['response_time'],
            use_container_width=True
        )
        st.caption("Response Time Trend")
        
        # Retrieval strategy distribution
        strategy_counts = df['retrieval_strategy'].value_counts()
        st.bar_chart(strategy_counts)
        st.caption("Retrieval Strategy Usage")
        
        # Recent queries table
        st.markdown("### 📋 Recent Queries")
        recent_df = df.nlargest(10, 'timestamp')[['timestamp', 'query', 'confidence', 'response_time', 'retrieval_strategy']]
        recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(recent_df, use_container_width=True)
        
    else:
        st.info("📊 No query data available yet. Start making queries to see analytics!")
        
        # Placeholder charts
        st.markdown("### 🎯 What you'll see here:")
        st.markdown("""
        - Query volume trends
        - Response time analytics  
        - Confidence score distribution
        - Cache performance metrics
        - Retrieval strategy effectiveness
        """)

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])

with footer_col1:
    st.markdown("🤖 **Enhanced Vertex AI RAG Agent v2.0**")

with footer_col2:
    st.markdown(f"API: {'🟢 Online' if api_healthy else '🔴 Offline'}")

with footer_col3:
    st.markdown(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")