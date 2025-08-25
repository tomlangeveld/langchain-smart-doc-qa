"""RAG chain implementation with LangChain."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI as ChatOpenAINew

from .vector_store import VectorStoreManager
from ..config.settings import settings

logger = logging.getLogger(__name__)


class RAGMetricsCallback(BaseCallbackHandler):
    """Callback to collect RAG performance metrics."""
    
    def __init__(self):
        self.metrics = {
            "total_tokens": 0,
            "retrieval_time": 0,
            "llm_time": 0,
            "total_time": 0,
            "retrieved_docs": 0,
            "queries_processed": 0
        }
        self.start_time = None
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        self.start_time = datetime.now()
        self.metrics["queries_processed"] += 1
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        if self.start_time:
            self.metrics["total_time"] = (datetime.now() - self.start_time).total_seconds()
    
    def on_retriever_end(self, documents: List[Document], **kwargs):
        self.metrics["retrieved_docs"] = len(documents)


class RAGChain:
    """Advanced RAG chain with source attribution and conversation memory."""
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        llm_provider: str = None,
        model_name: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ):
        self.vector_store_manager = vector_store_manager
        self.llm_provider = llm_provider or settings.default_llm
        self.temperature = temperature or settings.temperature
        self.max_tokens = max_tokens or settings.max_tokens
        
        # Initialize LLM
        self.llm = self._initialize_llm(model_name, **kwargs)
        
        # Initialize retriever
        self.retriever = self._initialize_retriever()
        
        # Create RAG chain
        self.rag_chain = self._create_rag_chain()
        
        # Metrics callback
        self.metrics_callback = RAGMetricsCallback()
        
        # Conversation history (simple implementation)
        self.conversation_history: List[Dict[str, str]] = []
    
    def _initialize_llm(self, model_name: str = None, **kwargs):
        """Initialize the LLM based on provider."""
        try:
            if self.llm_provider.lower() == "openai":
                model = model_name or settings.openai_model
                return ChatOpenAINew(
                    model=model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **kwargs
                )
            
            elif self.llm_provider.lower() == "anthropic":
                from langchain_anthropic import ChatAnthropic
                model = model_name or settings.anthropic_model
                return ChatAnthropic(
                    model=model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **kwargs
                )
            
            elif self.llm_provider.lower() == "ollama":
                model = model_name or "llama2"
                return Ollama(
                    model=model,
                    temperature=self.temperature,
                    **kwargs
                )
            
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
    
    def _initialize_retriever(self, search_type: str = "similarity", **kwargs):
        """Initialize the document retriever."""
        if not self.vector_store_manager.vector_store:
            raise ValueError("Vector store not loaded. Load or create a vector store first.")
        
        # Configure retriever based on search type
        if search_type == "similarity":
            return self.vector_store_manager.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": kwargs.get("k", 5),
                    "score_threshold": kwargs.get("score_threshold", 0.3)
                }
            )
        
        elif search_type == "mmr":
            return self.vector_store_manager.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": kwargs.get("k", 5),
                    "fetch_k": kwargs.get("fetch_k", 10),
                    "lambda_mult": kwargs.get("lambda_mult", 0.5)
                }
            )
        
        else:
            raise ValueError(f"Unsupported search type: {search_type}")
    
    def _create_rag_chain(self):
        """Create the RAG chain using LangChain Expression Language (LCEL)."""
        
        # Enhanced RAG prompt template
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an intelligent assistant that answers questions based on provided context documents.

Instructions:
1. Use ONLY the information provided in the context documents to answer questions
2. If the answer cannot be found in the context, clearly state "I cannot find this information in the provided documents"
3. Always cite your sources by mentioning the relevant document name or section
4. Provide comprehensive answers when the context supports it
5. If multiple sources contain relevant information, synthesize them coherently
6. Maintain a helpful and professional tone

Context Documents:
{context}

Conversation History:
{chat_history}
"""),
            ("human", "{question}")
        ])
        
        # Chain for formatting documents
        def format_docs(docs: List[Document]) -> str:
            """Format documents for context."""
            formatted = []
            for i, doc in enumerate(docs, 1):
                content = doc.page_content.strip()
                metadata = doc.metadata
                
                source_info = f"Document {i}"
                if metadata.get("file_name"):
                    source_info += f" ({metadata['file_name']})"
                if metadata.get("chunk_id") is not None:
                    source_info += f" [Chunk {metadata['chunk_id']}]"
                
                formatted.append(f"--- {source_info} ---\n{content}")
            
            return "\n\n".join(formatted)
        
        # Format conversation history
        def format_chat_history() -> str:
            """Format conversation history."""
            if not self.conversation_history:
                return "No previous conversation."
            
            formatted = []
            for i, exchange in enumerate(self.conversation_history[-3:], 1):  # Last 3 exchanges
                formatted.append(f"Q{i}: {exchange['question']}")
                formatted.append(f"A{i}: {exchange['answer']}")
            
            return "\n".join(formatted)
        
        # Create the RAG chain using LCEL
        rag_chain = (
            RunnableParallel({
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: format_chat_history()
            })
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def query(
        self,
        question: str,
        include_sources: bool = True,
        return_source_documents: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Query the RAG system with comprehensive response."""
        try:
            start_time = datetime.now()
            
            # Retrieve relevant documents first
            retrieved_docs = self.retriever.invoke(question)
            retrieval_time = (datetime.now() - start_time).total_seconds()
            
            # Generate answer using RAG chain
            llm_start = datetime.now()
            answer = self.rag_chain.invoke(question)
            llm_time = (datetime.now() - llm_start).total_seconds()
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Process sources
            sources = []
            if include_sources and retrieved_docs:
                for i, doc in enumerate(retrieved_docs, 1):
                    metadata = doc.metadata
                    source = {
                        "rank": i,
                        "content_preview": doc.page_content[:200] + "...",
                        "file_name": metadata.get("file_name", "Unknown"),
                        "file_path": metadata.get("file_path", "Unknown"),
                        "chunk_id": metadata.get("chunk_id", 0),
                        "chunk_size": metadata.get("chunk_size", len(doc.page_content))
                    }
                    sources.append(source)
            
            # Build response
            response = {
                "answer": answer,
                "question": question,
                "sources": sources if include_sources else [],
                "metadata": {
                    "retrieval_time": retrieval_time,
                    "llm_time": llm_time,
                    "total_time": total_time,
                    "documents_retrieved": len(retrieved_docs),
                    "llm_provider": self.llm_provider,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            if return_source_documents:
                response["source_documents"] = retrieved_docs
            
            # Update conversation history
            self.conversation_history.append({
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only last 10 exchanges to manage memory
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            logger.info(f"Processed query in {total_time:.2f}s with {len(retrieved_docs)} documents")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def batch_query(
        self, 
        questions: List[str], 
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process multiple queries in batch."""
        results = []
        
        for question in questions:
            try:
                result = self.query(question, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing question '{question}': {str(e)}")
                results.append({
                    "question": question,
                    "answer": f"Error processing question: {str(e)}",
                    "error": True
                })
        
        return results
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self.conversation_history.copy()
    
    def update_retriever_config(self, **kwargs):
        """Update retriever configuration."""
        self.retriever = self._initialize_retriever(**kwargs)
        self.rag_chain = self._create_rag_chain()
        logger.info("Retriever configuration updated")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the RAG system."""
        return {
            "llm_provider": self.llm_provider,
            "llm_model": getattr(self.llm, "model_name", "unknown"),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "vector_store_type": self.vector_store_manager.store_type,
            "embedding_model": self.vector_store_manager.embedding_manager.model_name,
            "conversation_length": len(self.conversation_history),
            "vector_store_stats": self.vector_store_manager.get_vector_store_stats()
        }
    
    def evaluate_retrieval_quality(
        self, 
        test_questions: List[str], 
        expected_docs: List[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate retrieval quality with test questions."""
        if not test_questions:
            return {}
        
        metrics = {
            "avg_retrieval_time": 0,
            "avg_documents_retrieved": 0,
            "retrieval_consistency": 0
        }
        
        retrieval_times = []
        doc_counts = []
        
        for question in test_questions:
            start_time = datetime.now()
            docs = self.retriever.invoke(question)
            retrieval_time = (datetime.now() - start_time).total_seconds()
            
            retrieval_times.append(retrieval_time)
            doc_counts.append(len(docs))
        
        metrics["avg_retrieval_time"] = sum(retrieval_times) / len(retrieval_times)
        metrics["avg_documents_retrieved"] = sum(doc_counts) / len(doc_counts)
        
        # Consistency: how stable are the document counts
        if len(doc_counts) > 1:
            import statistics
            metrics["retrieval_consistency"] = 1.0 - (statistics.stdev(doc_counts) / max(doc_counts))
        else:
            metrics["retrieval_consistency"] = 1.0
        
        return metrics