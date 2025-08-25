"""Tests for embedding manager."""

import pytest
import numpy as np
from unittest.mock import patch, Mock

from src.core.embeddings import EmbeddingManager


class TestEmbeddingManager:
    """Test cases for EmbeddingManager class."""
    
    @patch('src.core.embeddings.HuggingFaceEmbeddings')
    @patch('src.core.embeddings.SentenceTransformer')
    def test_init_default_model(self, mock_st, mock_hf):
        """Test initialization with default model."""
        manager = EmbeddingManager()
        
        assert manager.model_name is not None
        assert manager.device is not None
        mock_hf.assert_called_once()
        mock_st.assert_called_once()
    
    @patch('src.core.embeddings.HuggingFaceEmbeddings')
    @patch('src.core.embeddings.SentenceTransformer')
    def test_init_custom_model(self, mock_st, mock_hf):
        """Test initialization with custom model."""
        custom_model = "custom/model"
        custom_device = "cuda"
        
        manager = EmbeddingManager(
            model_name=custom_model,
            device=custom_device
        )
        
        assert manager.model_name == custom_model
        assert manager.device == custom_device
    
    def test_embed_documents(self, mock_embedding_manager):
        """Test document embedding generation."""
        texts = ["This is document 1", "This is document 2"]
        
        embeddings = mock_embedding_manager.embed_documents(texts)
        
        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
    
    def test_embed_query(self, mock_embedding_manager):
        """Test query embedding generation."""
        query = "What is machine learning?"
        
        embedding = mock_embedding_manager.embed_query(query)
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
    
    def test_compute_similarity(self):
        """Test similarity computation."""
        manager = EmbeddingManager.__new__(EmbeddingManager)  # Skip __init__
        
        # Create mock embeddings
        query_embedding = [1.0, 0.0, 0.0]
        doc_embeddings = [
            [1.0, 0.0, 0.0],  # Identical - similarity = 1.0
            [0.0, 1.0, 0.0],  # Orthogonal - similarity = 0.0
            [0.5, 0.5, 0.0],  # Similar - similarity > 0
        ]
        
        similarities = manager.compute_similarity(query_embedding, doc_embeddings)
        
        assert len(similarities) == len(doc_embeddings)
        assert similarities[0] == pytest.approx(1.0, abs=1e-6)  # Identical vectors
        assert similarities[1] == pytest.approx(0.0, abs=1e-6)  # Orthogonal vectors
        assert 0 < similarities[2] < 1  # Partially similar vectors
    
    def test_get_model_info(self, mock_embedding_manager):
        """Test model information retrieval."""
        info = mock_embedding_manager.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "device" in info
        assert "dimensions" in info
        assert "description" in info
    
    def test_benchmark_embedding_speed(self, mock_embedding_manager):
        """Test embedding speed benchmarking."""
        # Mock the actual embedding methods
        mock_embedding_manager.embed_documents = Mock(return_value=[[0.1, 0.2]] * 5)
        mock_embedding_manager.embed_query = Mock(return_value=[0.1, 0.2])
        
        test_texts = ["test text"] * 5
        
        benchmark = mock_embedding_manager.benchmark_embedding_speed(test_texts)
        
        assert isinstance(benchmark, dict)
        assert "documents_per_second" in benchmark
        assert "document_embedding_time" in benchmark
        assert "query_embedding_time" in benchmark
        assert "total_test_texts" in benchmark
        
        assert benchmark["total_test_texts"] == len(test_texts)
    
    def test_list_available_models(self):
        """Test listing available models."""
        models = EmbeddingManager.list_available_models()
        
        assert isinstance(models, dict)
        assert len(models) > 0
        
        # Check that each model has required information
        for model_info in models.values():
            assert "model_name" in model_info
            assert "dimensions" in model_info
            assert "description" in model_info
    
    def test_recommend_model(self):
        """Test model recommendation."""
        # Test different use cases
        general_model = EmbeddingManager.recommend_model("general")
        qa_model = EmbeddingManager.recommend_model("qa")
        fast_model = EmbeddingManager.recommend_model("fast")
        unknown_model = EmbeddingManager.recommend_model("unknown_use_case")
        
        assert isinstance(general_model, str)
        assert isinstance(qa_model, str)
        assert isinstance(fast_model, str)
        assert isinstance(unknown_model, str)
        
        # Should return valid model names
        assert general_model.startswith("sentence-transformers/")
        assert qa_model.startswith("sentence-transformers/")
        assert fast_model.startswith("sentence-transformers/")
    
    @patch('src.core.embeddings.HuggingFaceEmbeddings')
    def test_initialization_error_handling(self, mock_hf):
        """Test error handling during initialization."""
        mock_hf.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception, match="Model loading failed"):
            EmbeddingManager()
    
    def test_empty_text_list(self, mock_embedding_manager):
        """Test handling of empty text lists."""
        mock_embedding_manager.embed_documents = Mock(return_value=[])
        
        embeddings = mock_embedding_manager.embed_documents([])
        
        assert embeddings == []
        mock_embedding_manager.embed_documents.assert_called_once_with([])