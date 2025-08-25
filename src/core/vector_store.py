"""Vector store management with multiple backend support."""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import faiss
import numpy as np
from langchain.schema import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain.vectorstores.base import VectorStore

from .embeddings import EmbeddingManager
from ..config.settings import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector stores with multiple backend support."""
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        store_type: str = None,
        store_path: Optional[Path] = None
    ):
        self.embedding_manager = embedding_manager
        self.store_type = store_type or settings.vector_store_type
        self.store_path = store_path or settings.vector_store_path
        self.vector_store: Optional[VectorStore] = None
        
        # Ensure store directory exists
        self.store_path.mkdir(parents=True, exist_ok=True)
    
    def create_vector_store(
        self, 
        documents: List[Document], 
        store_name: str = "default"
    ) -> VectorStore:
        """Create a new vector store from documents."""
        if not documents:
            raise ValueError("Cannot create vector store with empty document list")
        
        logger.info(f"Creating {self.store_type} vector store with {len(documents)} documents")
        
        try:
            if self.store_type.lower() == "faiss":
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embedding_manager.embeddings,
                    distance_strategy="COSINE"
                )
            
            elif self.store_type.lower() == "chroma":
                persist_directory = str(self.store_path / store_name)
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embedding_manager.embeddings,
                    persist_directory=persist_directory,
                    collection_name=store_name
                )
            
            else:
                raise ValueError(f"Unsupported vector store type: {self.store_type}")
            
            logger.info(f"Successfully created vector store: {store_name}")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def save_vector_store(self, store_name: str = "default") -> None:
        """Save the vector store to disk."""
        if not self.vector_store:
            raise ValueError("No vector store to save")
        
        try:
            if self.store_type.lower() == "faiss":
                store_path = self.store_path / f"{store_name}_faiss"
                self.vector_store.save_local(str(store_path))
                
            elif self.store_type.lower() == "chroma":
                # Chroma auto-persists if persist_directory is specified
                pass
            
            logger.info(f"Saved vector store: {store_name}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def load_vector_store(self, store_name: str = "default") -> Optional[VectorStore]:
        """Load a vector store from disk."""
        try:
            if self.store_type.lower() == "faiss":
                store_path = self.store_path / f"{store_name}_faiss"
                if store_path.exists():
                    self.vector_store = FAISS.load_local(
                        str(store_path),
                        embeddings=self.embedding_manager.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    logger.info(f"Loaded FAISS vector store: {store_name}")
                    return self.vector_store
            
            elif self.store_type.lower() == "chroma":
                persist_directory = str(self.store_path / store_name)
                if Path(persist_directory).exists():
                    self.vector_store = Chroma(
                        embedding_function=self.embedding_manager.embeddings,
                        persist_directory=persist_directory,
                        collection_name=store_name
                    )
                    logger.info(f"Loaded Chroma vector store: {store_name}")
                    return self.vector_store
            
            logger.warning(f"Vector store not found: {store_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return None
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to existing vector store."""
        if not self.vector_store:
            raise ValueError("No vector store loaded. Create or load one first.")
        
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores."""
        if not self.vector_store:
            raise ValueError("No vector store loaded")
        
        try:
            # Use similarity_search_with_score for better results
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            # Filter by score threshold if specified
            if score_threshold > 0:
                results = [
                    (doc, score) for doc, score in results 
                    if score >= score_threshold
                ]
            
            logger.debug(
                f"Found {len(results)} documents for query: {query[:50]}..."
            )
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
    
    def mmr_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 10,
        lambda_mult: float = 0.5
    ) -> List[Document]:
        """Maximal Marginal Relevance search for diverse results."""
        if not self.vector_store:
            raise ValueError("No vector store loaded")
        
        try:
            results = self.vector_store.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult
            )
            
            logger.debug(
                f"MMR search returned {len(results)} diverse documents"
            )
            return results
            
        except Exception as e:
            logger.error(f"Error in MMR search: {str(e)}")
            raise
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the current vector store."""
        if not self.vector_store:
            return {"status": "No vector store loaded"}
        
        stats = {
            "store_type": self.store_type,
            "embedding_model": self.embedding_manager.model_name,
            "store_path": str(self.store_path),
        }
        
        try:
            if self.store_type.lower() == "faiss":
                # FAISS specific stats
                if hasattr(self.vector_store, 'index'):
                    stats["total_vectors"] = self.vector_store.index.ntotal
                    stats["vector_dimension"] = self.vector_store.index.d
            
            elif self.store_type.lower() == "chroma":
                # Chroma specific stats
                collection = self.vector_store._collection
                stats["total_vectors"] = collection.count()
                
        except Exception as e:
            logger.warning(f"Could not get detailed stats: {str(e)}")
        
        return stats
    
    def delete_vector_store(self, store_name: str = "default") -> bool:
        """Delete a vector store from disk."""
        try:
            if self.store_type.lower() == "faiss":
                store_path = self.store_path / f"{store_name}_faiss"
                if store_path.exists():
                    import shutil
                    shutil.rmtree(store_path)
                    logger.info(f"Deleted FAISS vector store: {store_name}")
                    return True
            
            elif self.store_type.lower() == "chroma":
                persist_directory = self.store_path / store_name
                if persist_directory.exists():
                    import shutil
                    shutil.rmtree(persist_directory)
                    logger.info(f"Deleted Chroma vector store: {store_name}")
                    return True
            
            logger.warning(f"Vector store not found: {store_name}")
            return False
            
        except Exception as e:
            logger.error(f"Error deleting vector store: {str(e)}")
            return False