"""Streamlit web interface for the RAG system."""

import streamlit as st
import requests
import json
from typing import List, Dict, Any
from datetime import datetime
import time
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Page config
st.set_page_config(
    page_title="Smart Document Q&A",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4ECDC4;
    }
    
    .source-card {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        border-left: 3px solid #007bff;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"


def call_api(endpoint: str, method: str = "GET", data: Dict = None, files: List = None) -> Dict:
    """Make API calls with error handling."""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, data=data)
            else:
                response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to API server. Please ensure the FastAPI server is running."}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"API request failed: {str(e)}"}
    except json.JSONDecodeError:
        return {"success": False, "error": "Invalid response from API server"}


def display_system_status():
    """Display system status in sidebar."""
    with st.sidebar:
        st.header("🔧 System Status")
        
        status_response = call_api("/api/status")
        
        if status_response.get("success", False):
            status_data = status_response.get("system_info", {})
            
            # Status indicators
            if status_data.get("rag_chain_ready", False):
                st.success("✅ System Ready")
            else:
                st.warning("⚠️ System Initializing")
            
            st.metric("Vector Store", "Loaded" if status_data.get("vector_store_loaded") else "Empty")
            st.metric("LLM Provider", status_data.get("llm_provider", "Unknown"))
            st.metric("Embedding Model", status_data.get("embedding_model", "Unknown").split("/")[-1])
            
            # Vector store stats
            if "vector_store_stats" in status_data:
                vs_stats = status_data["vector_store_stats"]
                st.metric("Documents", vs_stats.get("total_vectors", "N/A"))
            
            # Conversation length
            conv_len = status_data.get("conversation_length", 0)
            st.metric("Conversation", f"{conv_len} exchanges")
            
        else:
            st.error(f"❌ {status_response.get('error', 'System not available')}")


def main_interface():
    """Main application interface."""
    # Header
    st.markdown('<h1 class="main-header">🚀 Smart Document Q&A System</h1>', unsafe_allow_html=True)
    st.markdown("**Enterprise RAG System powered by LangChain**")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Chat", 
        "📄 Upload Documents", 
        "📊 Analytics", 
        "🔍 System Info", 
        "🧪 Testing"
    ])
    
    with tab1:
        chat_interface()
    
    with tab2:
        document_upload_interface()
    
    with tab3:
        analytics_interface()
    
    with tab4:
        system_info_interface()
    
    with tab5:
        testing_interface()


def chat_interface():
    """Chat interface for asking questions."""
    st.header("💬 Ask Questions About Your Documents")
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_question = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., What are the main findings in the research papers?",
            key="question_input"
        )
    
    with col2:
        ask_button = st.button("Ask", type="primary", use_container_width=True)
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        include_sources = st.checkbox("Include source citations", value=True)
        store_name = st.text_input("Vector store name", value="default")
    
    # Process question
    if ask_button and user_question:
        with st.spinner("Thinking..."):
            query_data = {
                "question": user_question,
                "include_sources": include_sources,
                "store_name": store_name
            }
            
            response = call_api("/api/query", "POST", query_data)
            
            if response.get("success"):
                answer_data = response["data"]
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": answer_data["answer"],
                    "sources": answer_data.get("sources", []),
                    "metadata": answer_data.get("metadata", {}),
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                # Clear input
                st.rerun()
            else:
                st.error(f"Error: {response.get('error', 'Unknown error')}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💭 Conversation History")
        
        for i, exchange in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.markdown(f"**[{exchange['timestamp']}] Question:**")
                st.markdown(f"*{exchange['question']}*")
                
                st.markdown("**Answer:**")
                st.markdown(exchange['answer'])
                
                # Display sources if available
                if exchange.get('sources'):
                    st.markdown("**📚 Sources:**")
                    for j, source in enumerate(exchange['sources'][:3], 1):
                        st.markdown(
                            f'<div class="source-card">'
                            f'<strong>Source {j}:</strong> {source.get("file_name", "Unknown")}<br>'
                            f'<small>{source.get("content_preview", "")}</small>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                
                # Performance metrics
                if exchange.get('metadata'):
                    metadata = exchange['metadata']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Response Time", f"{metadata.get('total_time', 0):.2f}s")
                    with col2:
                        st.metric("Documents Found", metadata.get('documents_retrieved', 0))
                    with col3:
                        st.metric("LLM Time", f"{metadata.get('llm_time', 0):.2f}s")
                
                st.divider()
        
        # Clear history button
        if st.button("🗑️ Clear Chat History"):
            response = call_api("/api/conversation-history", "DELETE")
            if response.get("success"):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
                st.rerun()
    else:
        st.info("👋 Upload some documents and start asking questions!")


def document_upload_interface():
    """Document upload and processing interface."""
    st.header("📄 Upload & Process Documents")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose documents to upload",
        type=['pdf', 'docx', 'doc', 'txt', 'md'],
        accept_multiple_files=True,
        help="Supported formats: PDF, Word docs, text files, and Markdown"
    )
    
    # Processing options
    with st.expander("⚙️ Processing Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            store_name = st.text_input("Vector store name", value="default")
            chunk_size = st.number_input("Chunk size", min_value=100, max_value=2000, value=1000)
        
        with col2:
            chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=200)
    
    # Process documents
    if uploaded_files and st.button("🚀 Process Documents", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Prepare files for upload
            files = []
            for uploaded_file in uploaded_files:
                files.append(('files', (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)))
            
            status_text.text("Uploading files...")
            progress_bar.progress(25)
            
            # Prepare form data
            form_data = {
                'store_name': store_name,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap
            }
            
            status_text.text("Processing documents...")
            progress_bar.progress(50)
            
            # Make API call
            response = call_api("/api/process-documents", "POST", data=form_data, files=files)
            
            progress_bar.progress(100)
            status_text.text("Processing complete!")
            
            if response.get("success"):
                st.markdown(
                    '<div class="success-message">'
                    f'✅ Successfully processed {len(uploaded_files)} documents!'
                    '</div>',
                    unsafe_allow_html=True
                )
                
                # Display statistics
                stats = response.get("statistics", {})
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Files Processed", stats.get("files_processed", 0))
                
                with col2:
                    st.metric("Text Chunks", stats.get("total_documents", 0))
                
                with col3:
                    st.metric("Total Characters", f"{stats.get('total_characters', 0):,}")
                
                with col4:
                    st.metric("Avg Chunk Size", f"{stats.get('average_chunk_size', 0):.0f}")
                
                # File type breakdown
                if stats.get("file_types"):
                    st.subheader("📊 File Types Processed")
                    file_types_df = pd.DataFrame(
                        list(stats["file_types"].items()),
                        columns=["File Type", "Count"]
                    )
                    
                    fig = px.pie(
                        file_types_df, 
                        values="Count", 
                        names="File Type",
                        title="Document Types Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.markdown(
                    f'<div class="error-message">'
                    f'❌ Error: {response.get("error", "Unknown error")}'
                    f'</div>',
                    unsafe_allow_html=True
                )
        
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
        
        finally:
            progress_bar.empty()
            status_text.empty()
    
    # List existing vector stores
    st.subheader("📚 Existing Document Collections")
    
    stores_response = call_api("/api/vector-stores")
    
    if stores_response.get("success"):
        stores = stores_response.get("data", [])
        
        if stores:
            stores_df = pd.DataFrame(stores)
            stores_df['size_mb'] = stores_df['size'] / (1024 * 1024)
            stores_df['created'] = pd.to_datetime(stores_df['created']).dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(
                stores_df[['name', 'created', 'size_mb']].rename(columns={
                    'name': 'Collection Name',
                    'created': 'Created',
                    'size_mb': 'Size (MB)'
                }),
                use_container_width=True
            )
        else:
            st.info("No document collections found. Upload some documents to get started!")


def analytics_interface():
    """Analytics and performance monitoring."""
    st.header("📊 System Analytics")
    
    # Run benchmark
    if st.button("🧪 Run Performance Benchmark", type="primary"):
        with st.spinner("Running benchmark..."):
            benchmark_response = call_api("/api/benchmark", "POST")
            
            if benchmark_response.get("success"):
                benchmark_data = benchmark_response["data"]
                
                # Display benchmark results
                st.subheader("🔬 Benchmark Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("⚡ Performance Metrics")
                    
                    if "e2e_benchmark" in benchmark_data:
                        e2e = benchmark_data["e2e_benchmark"]
                        
                        st.metric("Avg Response Time", f"{e2e.get('average_response_time', 0):.2f}s")
                        st.metric("Queries per Second", f"{e2e.get('queries_per_second', 0):.1f}")
                        st.metric("Successful Queries", e2e.get('successful_queries', 0))
                
                with col2:
                    st.subheader("🔍 Retrieval Quality")
                    
                    if "retrieval_benchmark" in benchmark_data:
                        retrieval = benchmark_data["retrieval_benchmark"]
                        
                        st.metric("Avg Retrieval Time", f"{retrieval.get('avg_retrieval_time', 0):.3f}s")
                        st.metric("Avg Documents Retrieved", f"{retrieval.get('avg_documents_retrieved', 0):.1f}")
                        st.metric("Retrieval Consistency", f"{retrieval.get('retrieval_consistency', 0):.2%}")
                
                # Embedding benchmark
                if "embedding_benchmark" in benchmark_data:
                    embed = benchmark_data["embedding_benchmark"]
                    
                    st.subheader("🧠 Embedding Performance")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Documents/Second", f"{embed.get('documents_per_second', 0):.1f}")
                    
                    with col2:
                        st.metric("Document Embedding Time", f"{embed.get('document_embedding_time', 0):.3f}s")
                    
                    with col3:
                        st.metric("Query Embedding Time", f"{embed.get('query_embedding_time', 0):.3f}s")
    
    # System metrics over time (mock data for demo)
    st.subheader("📈 Performance Over Time")
    
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    metrics_df = pd.DataFrame({
        'Date': dates,
        'Response Time (s)': 1.5 + 0.5 * pd.Series(range(len(dates))).apply(lambda x: (x % 30) / 30),
        'Documents Retrieved': 4 + pd.Series(range(len(dates))).apply(lambda x: (x % 7)),
        'Success Rate (%)': 95 + 5 * pd.Series(range(len(dates))).apply(lambda x: (x % 20) / 20)
    })
    
    # Response time chart
    fig_response = px.line(
        metrics_df, 
        x='Date', 
        y='Response Time (s)',
        title='Average Response Time Over Time',
        color_discrete_sequence=['#FF6B6B']
    )
    st.plotly_chart(fig_response, use_container_width=True)
    
    # Success rate chart
    fig_success = px.area(
        metrics_df,
        x='Date',
        y='Success Rate (%)',
        title='Query Success Rate Over Time',
        color_discrete_sequence=['#4ECDC4']
    )
    st.plotly_chart(fig_success, use_container_width=True)


def system_info_interface():
    """System information and configuration."""
    st.header("🔍 System Information")
    
    # Get system status
    status_response = call_api("/api/status")
    
    if status_response.get("success"):
        system_info = status_response.get("system_info", {})
        
        # System overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏗️ System Configuration")
            
            config_data = {
                "LLM Provider": system_info.get("llm_provider", "Unknown"),
                "Embedding Model": system_info.get("embedding_model", "Unknown"),
                "Vector Store Type": system_info.get("vector_store_type", "Unknown"),
                "System Status": "Ready" if system_info.get("rag_chain_ready") else "Initializing"
            }
            
            for key, value in config_data.items():
                st.text(f"{key}: {value}")
        
        with col2:
            st.subheader("📊 System Metrics")
            
            if "vector_store_stats" in system_info:
                vs_stats = system_info["vector_store_stats"]
                
                st.metric("Total Vectors", vs_stats.get("total_vectors", "N/A"))
                st.metric("Vector Dimensions", vs_stats.get("vector_dimension", "N/A"))
            
            if "embedding_info" in system_info:
                embed_info = system_info["embedding_info"]
                
                st.metric("Model Dimensions", embed_info.get("dimensions", "Unknown"))
                st.text(f"Description: {embed_info.get('description', 'N/A')}")
        
        # Detailed system info
        with st.expander("🔧 Detailed System Information"):
            st.json(system_info)
    
    else:
        st.error(f"Could not retrieve system information: {status_response.get('error')}")


def testing_interface():
    """Testing and validation interface."""
    st.header("🧪 System Testing")
    
    # Batch query testing
    st.subheader("📋 Batch Query Testing")
    
    # Predefined test questions
    default_questions = [
        "What is this document about?",
        "Can you summarize the main points?",
        "What are the key findings mentioned?",
        "Who are the main authors or contributors?",
        "What methodology was used?"
    ]
    
    # Custom questions input
    test_questions = st.text_area(
        "Test Questions (one per line):",
        value="\n".join(default_questions),
        height=150
    )
    
    if st.button("🚀 Run Batch Test", type="primary"):
        questions_list = [q.strip() for q in test_questions.split("\n") if q.strip()]
        
        if questions_list:
            with st.spinner(f"Testing {len(questions_list)} questions..."):
                batch_data = {
                    "questions": questions_list,
                    "include_sources": True
                }
                
                response = call_api("/api/batch-query", "POST", batch_data)
                
                if response.get("success"):
                    results = response["data"]
                    
                    st.success(f"✅ Completed {len(results)} test queries!")
                    
                    # Display results
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Question {i}: {result['question'][:60]}..."):
                            st.markdown("**Answer:**")
                            st.write(result["answer"])
                            
                            if result.get("sources"):
                                st.markdown("**Sources:**")
                                for j, source in enumerate(result["sources"][:2], 1):
                                    st.text(f"{j}. {source.get('file_name', 'Unknown')}")
                            
                            # Performance metrics
                            if result.get("metadata"):
                                metadata = result["metadata"]
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Response Time", f"{metadata.get('total_time', 0):.2f}s")
                                with col2:
                                    st.metric("Documents Retrieved", metadata.get('documents_retrieved', 0))
                else:
                    st.error(f"Batch test failed: {response.get('error')}")
        else:
            st.warning("Please provide at least one test question.")


if __name__ == "__main__":
    # Display system status in sidebar
    display_system_status()
    
    # Main interface
    main_interface()