"""Tests for RAG system integration."""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

from src.core.rag_system import RAGSystem


class TestRAGSystem:
    """Test cases for RAGSystem integration."""
    
    def test_init_default_configuration(self):
        """Test initialization with default configuration."""
        with patch('src.core.rag_system.EmbeddingManager'), \
             patch('src.core.rag_system.VectorStoreManager'), \
             patch('src.core.rag_system.DocumentProcessor'):
            
            system = RAGSystem()
            
            assert system.embedding_model is not None
            assert system.vector_store_type is not None
            assert system.llm_provider is not None
            assert system.document_processor is not None
            assert system.embedding_manager is not None
            assert system.vector_store_manager is not None
            assert system.rag_chain is None  # Not initialized until documents are processed
    
    def test_init_custom_configuration(self):
        """Test initialization with custom configuration."""
        with patch('src.core.rag_system.EmbeddingManager'), \
             patch('src.core.rag_system.VectorStoreManager'), \
             patch('src.core.rag_system.DocumentProcessor'):
            
            system = RAGSystem(
                embedding_model="custom/model",
                vector_store_type="chroma",
                llm_provider="anthropic"
            )
            
            assert system.embedding_model == "custom/model"
            assert system.vector_store_type == "chroma"
            assert system.llm_provider == "anthropic"
    
    @patch('src.core.rag_system.RAGChain')
    def test_process_documents_success(self, mock_rag_chain, test_rag_system, sample_documents):
        """Test successful document processing."""
        # Mock the vector store manager
        test_rag_system.vector_store_manager.load_vector_store = Mock(return_value=None)
        test_rag_system.vector_store_manager.create_vector_store = Mock()
        test_rag_system.vector_store_manager.save_vector_store = Mock()
        
        # Mock document processor
        mock_chunks = [Mock() for _ in range(5)]
        test_rag_system.document_processor.process_documents = Mock(return_value=mock_chunks)
        test_rag_system.document_processor.get_document_stats = Mock(return_value={
            "total_documents": 5,
            "files_processed": 2,
            "processing_status": "success"
        })
        
        result = test_rag_system.process_documents(sample_documents)
        
        assert result["processing_status"] == "success"
        assert result["files_processed"] == 2
        assert test_rag_system.rag_chain is not None
        
        # Verify method calls
        test_rag_system.document_processor.process_documents.assert_called_once()
        test_rag_system.vector_store_manager.create_vector_store.assert_called_once()
        test_rag_system.vector_store_manager.save_vector_store.assert_called_once()
    
    def test_process_documents_no_files(self, test_rag_system):
        """Test processing with no valid files."""
        with pytest.raises(ValueError, match="No valid documents found"):
            test_rag_system.process_documents([])
    
    def test_process_documents_directory_expansion(self, test_rag_system, temp_dir):
        """Test directory expansion during processing."""
        # Create a directory with documents
        doc_dir = temp_dir / "documents"
        doc_dir.mkdir()
        
        (doc_dir / "test1.txt").write_text("Test document 1")
        (doc_dir / "test2.md").write_text("# Test document 2")
        (doc_dir / "ignored.xyz").write_text("This should be ignored")
        
        # Mock the processing
        test_rag_system.document_processor.process_documents = Mock(return_value=[Mock()])
        test_rag_system.document_processor.get_document_stats = Mock(return_value={
            "processing_status": "success"
        })
        test_rag_system.vector_store_manager.load_vector_store = Mock(return_value=None)
        test_rag_system.vector_store_manager.create_vector_store = Mock()
        test_rag_system.vector_store_manager.save_vector_store = Mock()
        
        with patch('src.core.rag_system.RAGChain'):
            result = test_rag_system.process_documents([doc_dir])
        
        # Should have processed only the supported file types
        processed_files = test_rag_system.document_processor.process_documents.call_args[0][0]
        assert len(processed_files) == 2  # Only .txt and .md files
        assert all(f.suffix in [".txt", ".md"] for f in processed_files)
    
    def test_load_existing_store_success(self, test_rag_system):
        """Test successfully loading an existing vector store."""
        mock_store = Mock()
        test_rag_system.vector_store_manager.load_vector_store = Mock(return_value=mock_store)
        
        with patch('src.core.rag_system.RAGChain'):
            result = test_rag_system.load_existing_store("test_store")
        
        assert result is True
        assert test_rag_system.rag_chain is not None
        test_rag_system.vector_store_manager.load_vector_store.assert_called_once_with("test_store")
    
    def test_load_existing_store_not_found(self, test_rag_system):
        """Test loading a non-existent vector store."""
        test_rag_system.vector_store_manager.load_vector_store = Mock(return_value=None)
        
        result = test_rag_system.load_existing_store("nonexistent_store")
        
        assert result is False
        assert test_rag_system.rag_chain is None
    
    def test_query_without_rag_chain(self, test_rag_system):
        """Test querying without initialized RAG chain."""
        with pytest.raises(ValueError, match="RAG chain not initialized"):
            test_rag_system.query("What is this about?")
    
    def test_query_with_rag_chain(self, test_rag_system):
        """Test successful querying with RAG chain."""
        # Mock RAG chain
        mock_rag_chain = Mock()
        mock_response = {
            "answer": "This is a test answer",
            "sources": [],
            "metadata": {"total_time": 1.5}
        }
        mock_rag_chain.query = Mock(return_value=mock_response)
        test_rag_system.rag_chain = mock_rag_chain
        
        result = test_rag_system.query("What is this about?")
        
        assert result == mock_response
        mock_rag_chain.query.assert_called_once_with("What is this about?")
    
    def test_batch_query(self, test_rag_system, test_questions):
        """Test batch querying functionality."""
        # Mock RAG chain
        mock_rag_chain = Mock()
        mock_responses = [
            {"answer": f"Answer {i}", "question": q} 
            for i, q in enumerate(test_questions, 1)
        ]
        mock_rag_chain.batch_query = Mock(return_value=mock_responses)
        test_rag_system.rag_chain = mock_rag_chain
        
        results = test_rag_system.batch_query(test_questions)
        
        assert len(results) == len(test_questions)
        assert all("answer" in result for result in results)
        mock_rag_chain.batch_query.assert_called_once_with(test_questions)
    
    def test_add_documents(self, test_rag_system, sample_documents):
        """Test adding new documents to existing vector store."""
        # Mock existing vector store
        test_rag_system.vector_store_manager.vector_store = Mock()
        
        # Mock document processing
        mock_chunks = [Mock() for _ in range(3)]
        test_rag_system.document_processor.process_documents = Mock(return_value=mock_chunks)
        test_rag_system.vector_store_manager.add_documents = Mock()
        
        result = test_rag_system.add_documents(sample_documents)
        
        assert result["chunks_added"] == 3
        assert result["status"] == "success"
        test_rag_system.vector_store_manager.add_documents.assert_called_once_with(mock_chunks)
    
    def test_add_documents_no_vector_store(self, test_rag_system, sample_documents):
        """Test adding documents without existing vector store."""
        test_rag_system.vector_store_manager.vector_store = None
        
        with pytest.raises(ValueError, match="No vector store loaded"):
            test_rag_system.add_documents(sample_documents)
    
    def test_get_system_status(self, test_rag_system):
        """Test system status reporting."""
        # Mock components
        test_rag_system.embedding_manager.get_model_info = Mock(return_value={
            "model_name": "test-model",
            "dimensions": 384
        })
        test_rag_system.vector_store_manager.vector_store = Mock()
        test_rag_system.vector_store_manager.get_vector_store_stats = Mock(return_value={
            "total_vectors": 100
        })
        
        status = test_rag_system.get_system_status()
        
        assert isinstance(status, dict)
        assert "initialized" in status
        assert "embedding_model" in status
        assert "vector_store_type" in status
        assert "llm_provider" in status
        assert "vector_store_loaded" in status
        assert "rag_chain_ready" in status
        assert "embedding_info" in status
        assert "vector_store_stats" in status
    
    def test_benchmark_system(self, test_rag_system, test_questions):
        """Test system benchmarking."""
        # Mock RAG chain for benchmarking
        mock_rag_chain = Mock()
        mock_rag_chain.evaluate_retrieval_quality = Mock(return_value={
            "avg_retrieval_time": 0.1,
            "avg_documents_retrieved": 5.0
        })
        test_rag_system.rag_chain = mock_rag_chain
        
        # Mock embedding manager benchmarking
        test_rag_system.embedding_manager.benchmark_embedding_speed = Mock(return_value={
            "documents_per_second": 100.0,
            "query_embedding_time": 0.01
        })
        
        # Mock batch query for e2e benchmarking
        test_rag_system.batch_query = Mock(return_value=[
            {"answer": "test", "error": False} for _ in range(3)
        ])
        
        benchmark = test_rag_system.benchmark_system(test_questions)
        
        assert "test_questions_count" in benchmark
        assert "embedding_benchmark" in benchmark
        assert "retrieval_benchmark" in benchmark
        assert "e2e_benchmark" in benchmark
        
        assert benchmark["test_questions_count"] == len(test_questions)
    
    def test_conversation_management(self, test_rag_system):
        """Test conversation history management."""
        # Mock RAG chain
        mock_rag_chain = Mock()
        mock_rag_chain.get_conversation_history = Mock(return_value=[
            {"question": "Test?", "answer": "Test answer"}
        ])
        mock_rag_chain.clear_conversation_history = Mock()
        test_rag_system.rag_chain = mock_rag_chain
        
        # Test export
        history = test_rag_system.export_conversation_history()
        assert len(history) == 1
        mock_rag_chain.get_conversation_history.assert_called_once()
        
        # Test clear
        test_rag_system.clear_conversation()
        mock_rag_chain.clear_conversation_history.assert_called_once()
    
    def test_create_demo_system(self, temp_dir):
        """Test demo system creation."""
        # Create demo documents
        demo_dir = temp_dir / "demo"
        demo_dir.mkdir()
        (demo_dir / "demo.txt").write_text("Demo document content")
        
        with patch('src.core.rag_system.EmbeddingManager'), \
             patch('src.core.rag_system.VectorStoreManager'), \
             patch('src.core.rag_system.DocumentProcessor'), \
             patch.object(RAGSystem, 'process_documents') as mock_process:
            
            mock_process.return_value = {"processing_status": "success"}
            
            demo_system = RAGSystem.create_demo_system(demo_dir)
            
            assert isinstance(demo_system, RAGSystem)
            assert demo_system.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
            assert demo_system.vector_store_type == "faiss"
            assert demo_system.llm_provider == "openai"
            
            mock_process.assert_called_once()