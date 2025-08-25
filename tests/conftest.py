"""Pytest configuration and fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.rag_system import RAGSystem
from src.config.settings import settings


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_documents(temp_dir):
    """Create sample documents for testing."""
    docs = []
    
    # Create a sample text file
    txt_file = temp_dir / "sample1.txt"
    txt_content = """
    This is a sample document about artificial intelligence.
    AI has revolutionized many industries including healthcare, finance, and transportation.
    Machine learning algorithms can process vast amounts of data to find patterns.
    Natural language processing enables computers to understand human language.
    The future of AI holds great promise for solving complex problems.
    """
    txt_file.write_text(txt_content.strip())
    docs.append(txt_file)
    
    # Create another sample document
    txt_file2 = temp_dir / "sample2.txt"
    txt_content2 = """
    LangChain is a framework for developing applications powered by language models.
    It provides components for document loading, text splitting, and vector storage.
    RAG (Retrieval Augmented Generation) combines retrieval with language generation.
    Vector databases enable semantic search over large document collections.
    Embeddings convert text into numerical representations for similarity comparison.
    """
    txt_file2.write_text(txt_content2.strip())
    docs.append(txt_file2)
    
    # Create a markdown file
    md_file = temp_dir / "sample3.md"
    md_content = """
    # Document Processing Pipeline
    
    ## Overview
    This document describes the automated document processing pipeline.
    
    ## Key Components
    - Document ingestion
    - Text extraction
    - Semantic chunking
    - Embedding generation
    - Vector storage
    
    ## Benefits
    - Faster information retrieval
    - Improved accuracy
    - Scalable processing
    """
    md_file.write_text(md_content.strip())
    docs.append(md_file)
    
    return docs


@pytest.fixture
def mock_embedding_manager():
    """Mock embedding manager for testing."""
    mock_manager = Mock()
    mock_manager.model_name = "test-model"
    mock_manager.embed_documents.return_value = [[0.1, 0.2, 0.3]] * 5
    mock_manager.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_manager.get_model_info.return_value = {
        "model_name": "test-model",
        "dimensions": 3,
        "description": "Test model"
    }
    return mock_manager


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock_llm = Mock()
    mock_llm.invoke.return_value = "This is a test response from the mock LLM."
    return mock_llm


@pytest.fixture
def test_rag_system(temp_dir, mock_embedding_manager):
    """Create a test RAG system with mocked components."""
    with patch('src.core.rag_system.EmbeddingManager', return_value=mock_embedding_manager):
        system = RAGSystem(
            embedding_model="test-model",
            vector_store_type="faiss",
            llm_provider="openai"
        )
        
        # Override paths for testing
        system.vector_store_manager.store_path = temp_dir / "vector_stores"
        system.vector_store_manager.store_path.mkdir(exist_ok=True)
        
        yield system


@pytest.fixture
def test_questions():
    """Sample questions for testing."""
    return [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What is LangChain used for?",
        "Explain RAG architecture",
        "What are the benefits of document processing?"
    ]