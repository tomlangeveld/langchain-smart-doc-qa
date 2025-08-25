"""Complete RAG system orchestration."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from .document_processor import DocumentProcessor
from .embeddings import EmbeddingManager
from .vector_store import VectorStoreManager
from .rag_chain import RAGChain
from ..config.settings import settings

logger = logging.getLogger(__name__)


class RAGSystem:
    """Complete RAG system orchestrating all components."""
    
    def __init__(
        self,
        embedding_model: str = None,
        vector_store_type: str = None,
        llm_provider: str = None,
        **kwargs
    ):
        """Initialize the complete RAG system."""
        self.embedding_model = embedding_model or settings.embedding_model
        self.vector_store_type = vector_store_type or settings.vector_store_type
        self.llm_provider = llm_provider or settings.default_llm
        
        # Initialize components
        self.document_processor = DocumentProcessor(**kwargs.get("processor_kwargs", {}))
        self.embedding_manager = EmbeddingManager(model_name=self.embedding_model)
        self.vector_store_manager = VectorStoreManager(
            embedding_manager=self.embedding_manager,
            store_type=self.vector_store_type
        )
        
        # RAG chain will be initialized after vector store is ready
        self.rag_chain: Optional[RAGChain] = None
        
        logger.info("RAG System initialized successfully")
    
    def process_documents(
        self,
        file_paths: Union[List[Path], List[str], Path, str],
        store_name: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """Process documents and create/update vector store."""
        # Convert to Path objects
        if isinstance(file_paths, (str, Path)):
            file_paths = [Path(file_paths)]
        else:
            file_paths = [Path(p) for p in file_paths]
        
        # Expand directories
        all_files = []
        for path in file_paths:
            if path.is_dir():
                # Recursively find all supported files
                for ext in DocumentProcessor.SUPPORTED_EXTENSIONS.keys():
                    all_files.extend(path.rglob(f"*{ext}"))
            elif path.is_file():
                all_files.append(path)
            else:
                logger.warning(f"Path not found: {path}")
        
        if not all_files:
            raise ValueError("No valid documents found to process")
        
        logger.info(f"Processing {len(all_files)} documents")
        
        try:
            # Process documents
            chunks = self.document_processor.process_documents(all_files)
            
            if not chunks:
                raise ValueError("No content extracted from documents")
            
            # Create or update vector store
            existing_store = self.vector_store_manager.load_vector_store(store_name)
            
            if existing_store:
                logger.info(f"Adding {len(chunks)} chunks to existing vector store")
                self.vector_store_manager.add_documents(chunks)
            else:
                logger.info(f"Creating new vector store with {len(chunks)} chunks")
                self.vector_store_manager.create_vector_store(chunks, store_name)
            
            # Save the vector store
            self.vector_store_manager.save_vector_store(store_name)
            
            # Initialize RAG chain if not already done
            if not self.rag_chain:
                self.rag_chain = RAGChain(
                    vector_store_manager=self.vector_store_manager,
                    llm_provider=self.llm_provider
                )
            
            # Get processing statistics
            stats = self.document_processor.get_document_stats(chunks)
            stats.update({
                "store_name": store_name,
                "files_processed": len(all_files),
                "processing_status": "success"
            })
            
            logger.info(f"Successfully processed documents: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise
    
    def load_existing_store(self, store_name: str = "default") -> bool:
        """Load an existing vector store."""
        try:
            store = self.vector_store_manager.load_vector_store(store_name)
            
            if store:
                # Initialize RAG chain
                self.rag_chain = RAGChain(
                    vector_store_manager=self.vector_store_manager,
                    llm_provider=self.llm_provider
                )
                
                logger.info(f"Loaded existing vector store: {store_name}")
                return True
            else:
                logger.warning(f"Vector store not found: {store_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def query(
        self,
        question: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Query the RAG system."""
        if not self.rag_chain:
            raise ValueError(
                "RAG chain not initialized. Process documents or load an existing store first."
            )
        
        return self.rag_chain.query(question, **kwargs)
    
    def batch_query(
        self,
        questions: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process multiple queries."""
        if not self.rag_chain:
            raise ValueError(
                "RAG chain not initialized. Process documents or load an existing store first."
            )
        
        return self.rag_chain.batch_query(questions, **kwargs)
    
    def add_documents(
        self,
        file_paths: Union[List[Path], List[str], Path, str],
        **kwargs
    ) -> Dict[str, Any]:
        """Add new documents to existing vector store."""
        if not self.vector_store_manager.vector_store:
            raise ValueError("No vector store loaded. Process documents first.")
        
        # Convert to Path objects
        if isinstance(file_paths, (str, Path)):
            file_paths = [Path(file_paths)]
        else:
            file_paths = [Path(p) for p in file_paths]
        
        # Process new documents
        chunks = self.document_processor.process_documents(file_paths)
        
        if chunks:
            self.vector_store_manager.add_documents(chunks)
            logger.info(f"Added {len(chunks)} new chunks to vector store")
            
            return {
                "chunks_added": len(chunks),
                "files_processed": len(file_paths),
                "status": "success"
            }
        else:
            logger.warning("No new content extracted")
            return {
                "chunks_added": 0,
                "files_processed": len(file_paths),
                "status": "no_content"
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "initialized": True,
            "embedding_model": self.embedding_model,
            "vector_store_type": self.vector_store_type,
            "llm_provider": self.llm_provider,
            "vector_store_loaded": self.vector_store_manager.vector_store is not None,
            "rag_chain_ready": self.rag_chain is not None,
        }
        
        # Add embedding model info
        if self.embedding_manager:
            status["embedding_info"] = self.embedding_manager.get_model_info()
        
        # Add vector store stats
        if self.vector_store_manager.vector_store:
            status["vector_store_stats"] = self.vector_store_manager.get_vector_store_stats()
        
        # Add RAG chain info
        if self.rag_chain:
            status["rag_info"] = self.rag_chain.get_system_info()
        
        return status
    
    def benchmark_system(self, test_questions: List[str] = None) -> Dict[str, Any]:
        """Benchmark system performance."""
        if not self.rag_chain:
            return {"error": "RAG system not ready"}
        
        # Default test questions if none provided
        if not test_questions:
            test_questions = [
                "What is this document about?",
                "Can you summarize the main points?",
                "What are the key findings mentioned?"
            ]
        
        benchmark_results = {
            "test_questions_count": len(test_questions),
            "embedding_benchmark": {},
            "retrieval_benchmark": {},
            "e2e_benchmark": {}
        }
        
        try:
            # Benchmark embedding speed
            test_texts = ["This is a test sentence for benchmarking."] * 10
            benchmark_results["embedding_benchmark"] = \
                self.embedding_manager.benchmark_embedding_speed(test_texts)
            
            # Benchmark retrieval quality
            benchmark_results["retrieval_benchmark"] = \
                self.rag_chain.evaluate_retrieval_quality(test_questions)
            
            # End-to-end benchmark
            import time
            start_time = time.time()
            
            responses = self.batch_query(test_questions[:3])  # Limit for speed
            
            total_time = time.time() - start_time
            avg_response_time = total_time / len(responses)
            
            benchmark_results["e2e_benchmark"] = {
                "total_time": total_time,
                "average_response_time": avg_response_time,
                "queries_per_second": len(responses) / total_time if total_time > 0 else 0,
                "successful_queries": len([r for r in responses if not r.get("error")])
            }
            
            logger.info(f"Benchmark completed: {benchmark_results}")
            
        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}")
            benchmark_results["error"] = str(e)
        
        return benchmark_results
    
    def export_conversation_history(self) -> List[Dict[str, str]]:
        """Export conversation history."""
        if self.rag_chain:
            return self.rag_chain.get_conversation_history()
        return []
    
    def clear_conversation(self):
        """Clear conversation history."""
        if self.rag_chain:
            self.rag_chain.clear_conversation_history()
    
    def update_config(
        self,
        embedding_model: str = None,
        llm_provider: str = None,
        **kwargs
    ):
        """Update system configuration (requires reinitialization)."""
        logger.warning("Configuration update requires system reinitialization")
        
        if embedding_model and embedding_model != self.embedding_model:
            self.embedding_model = embedding_model
            # Would need to reinitialize embedding manager and reprocess documents
        
        if llm_provider and llm_provider != self.llm_provider:
            self.llm_provider = llm_provider
            # Would need to reinitialize RAG chain
        
        # For production use, implement proper config updates
        logger.info(f"Config update requested: embedding_model={embedding_model}, llm_provider={llm_provider}")
    
    @classmethod
    def create_demo_system(
        cls,
        demo_docs_path: Optional[Path] = None
    ) -> 'RAGSystem':
        """Create a demo system with sample documents."""
        # Initialize with lightweight models for demo
        system = cls(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            vector_store_type="faiss",
            llm_provider="openai"  # Will fall back gracefully if no API key
        )
        
        # If demo documents provided, process them
        if demo_docs_path and demo_docs_path.exists():
            try:
                system.process_documents(demo_docs_path, store_name="demo")
                logger.info("Demo system created with sample documents")
            except Exception as e:
                logger.warning(f"Could not process demo documents: {str(e)}")
        
        return system