"""Embedding generation and management."""

import logging
from typing import List, Optional, Dict, Any
import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

from ..config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages document embeddings with multiple model support."""
    
    # Popular embedding models with their characteristics
    EMBEDDING_MODELS = {
        "all-MiniLM-L6-v2": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": 384,
            "description": "Lightweight, fast, good for general use",
            "max_seq_length": 256
        },
        "all-mpnet-base-v2": {
            "model_name": "sentence-transformers/all-mpnet-base-v2",
            "dimensions": 768,
            "description": "High quality, slower, best performance",
            "max_seq_length": 384
        },
        "multi-qa-MiniLM-L6-cos-v1": {
            "model_name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            "dimensions": 384,
            "description": "Optimized for question-answering tasks",
            "max_seq_length": 512
        },
        "paraphrase-multilingual-MiniLM-L12-v2": {
            "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "dimensions": 384,
            "description": "Multilingual support",
            "max_seq_length": 128
        }
    }
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ):
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        
        # Initialize the embedding model
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True},
                **kwargs
            )
            
            # Also keep direct access to sentence-transformers model
            self._st_model = SentenceTransformer(self.model_name, device=self.device)
            
            logger.info(f"Initialized embedding model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.debug(f"Generated embeddings for {len(texts)} documents")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating document embeddings: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        try:
            embedding = self.embeddings.embed_query(text)
            logger.debug(f"Generated query embedding for text of length {len(text)}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise
    
    def compute_similarity(
        self, 
        query_embedding: List[float], 
        doc_embeddings: List[List[float]]
    ) -> List[float]:
        """Compute cosine similarity between query and document embeddings."""
        query_vec = np.array(query_embedding).reshape(1, -1)
        doc_vecs = np.array(doc_embeddings)
        
        # Compute cosine similarity
        similarities = np.dot(doc_vecs, query_vec.T).flatten()
        
        # Normalize if vectors aren't already normalized
        query_norm = np.linalg.norm(query_vec)
        doc_norms = np.linalg.norm(doc_vecs, axis=1)
        
        if query_norm > 0 and np.all(doc_norms > 0):
            similarities = similarities / (query_norm * doc_norms)
        
        return similarities.tolist()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model."""
        model_key = self.model_name.split('/')[-1]
        model_info = self.EMBEDDING_MODELS.get(model_key, {})
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dimensions": model_info.get("dimensions", "unknown"),
            "description": model_info.get("description", "Custom model"),
            "max_seq_length": model_info.get("max_seq_length", "unknown")
        }
    
    def benchmark_embedding_speed(self, test_texts: List[str]) -> Dict[str, float]:
        """Benchmark embedding generation speed."""
        import time
        
        # Test document embedding speed
        start_time = time.time()
        _ = self.embed_documents(test_texts)
        doc_time = time.time() - start_time
        
        # Test query embedding speed
        start_time = time.time()
        _ = self.embed_query(test_texts[0] if test_texts else "test query")
        query_time = time.time() - start_time
        
        return {
            "documents_per_second": len(test_texts) / doc_time if doc_time > 0 else 0,
            "document_embedding_time": doc_time,
            "query_embedding_time": query_time,
            "total_test_texts": len(test_texts)
        }
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """List all available embedding models with their characteristics."""
        return cls.EMBEDDING_MODELS
    
    @classmethod
    def recommend_model(cls, use_case: str = "general") -> str:
        """Recommend an embedding model based on use case."""
        recommendations = {
            "general": "all-MiniLM-L6-v2",  # Fast and good enough
            "high_quality": "all-mpnet-base-v2",  # Best performance
            "qa": "multi-qa-MiniLM-L6-cos-v1",  # Optimized for Q&A
            "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # Multi-language
            "fast": "all-MiniLM-L6-v2",  # Fastest option
        }
        
        model_key = recommendations.get(use_case.lower(), "all-MiniLM-L6-v2")
        return cls.EMBEDDING_MODELS[model_key]["model_name"]