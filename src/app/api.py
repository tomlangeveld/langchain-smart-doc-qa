"""FastAPI backend for the RAG system."""

import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core.rag_system import RAGSystem
from ..config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LangChain Smart Document Q&A API",
    description="Enterprise RAG system for intelligent document analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system instance
rag_system: Optional[RAGSystem] = None


# Pydantic models for API
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    include_sources: bool = Field(True, description="Include source documents in response")
    return_source_documents: bool = Field(False, description="Include full source documents")
    store_name: str = Field("default", description="Vector store name to query")


class BatchQueryRequest(BaseModel):
    questions: List[str] = Field(..., min_items=1, max_items=10, description="List of questions")
    include_sources: bool = Field(True, description="Include source documents")
    store_name: str = Field("default", description="Vector store name to query")


class ProcessDocumentsRequest(BaseModel):
    store_name: str = Field("default", description="Name for the vector store")
    chunk_size: Optional[int] = Field(None, description="Text chunk size")
    chunk_overlap: Optional[int] = Field(None, description="Chunk overlap size")
    embedding_model: Optional[str] = Field(None, description="Embedding model to use")


class SystemStatusResponse(BaseModel):
    status: str
    initialized: bool
    vector_store_loaded: bool
    rag_chain_ready: bool
    system_info: Dict[str, Any]
    timestamp: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    global rag_system
    
    try:
        logger.info("Initializing RAG system...")
        rag_system = RAGSystem()
        
        # Try to load existing default vector store
        if rag_system.load_existing_store("default"):
            logger.info("Loaded existing default vector store")
        else:
            logger.info("No existing vector store found - ready for document processing")
        
        logger.info("RAG system initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        # Continue startup but system won't be fully functional


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_ready": rag_system is not None
    }


# System status endpoint
@app.get("/api/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get comprehensive system status."""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    status_info = rag_system.get_system_status()
    
    return SystemStatusResponse(
        status="ready" if status_info.get("rag_chain_ready") else "initializing",
        initialized=status_info.get("initialized", False),
        vector_store_loaded=status_info.get("vector_store_loaded", False),
        rag_chain_ready=status_info.get("rag_chain_ready", False),
        system_info=status_info,
        timestamp=datetime.now().isoformat()
    )


# Document processing endpoint
@app.post("/api/process-documents")
async def process_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    request_data: ProcessDocumentsRequest = ProcessDocumentsRequest()
):
    """Process uploaded documents and add to vector store."""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate file sizes
    max_size = settings.max_upload_size * 1024 * 1024  # Convert MB to bytes
    
    temp_files = []
    try:
        # Save uploaded files temporarily
        for file in files:
            if file.size > max_size:
                raise HTTPException(
                    status_code=413, 
                    detail=f"File {file.filename} exceeds maximum size of {settings.max_upload_size}MB"
                )
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
            shutil.copyfileobj(file.file, temp_file)
            temp_file.close()
            
            temp_files.append(Path(temp_file.name))
        
        logger.info(f"Processing {len(temp_files)} uploaded files")
        
        # Process documents
        processing_stats = rag_system.process_documents(
            temp_files,
            store_name=request_data.store_name
        )
        
        # Schedule cleanup of temporary files
        background_tasks.add_task(cleanup_temp_files, temp_files)
        
        return {
            "success": True,
            "message": f"Processed {len(files)} documents successfully",
            "statistics": processing_stats,
            "store_name": request_data.store_name,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        # Cleanup on error
        cleanup_temp_files(temp_files)
        logger.error(f"Error processing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Query endpoint
@app.post("/api/query")
async def query_documents(request: QueryRequest):
    """Query the RAG system with a single question."""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not rag_system.rag_chain:
        raise HTTPException(
            status_code=400, 
            detail="No documents processed yet. Upload documents first."
        )
    
    try:
        # Load specific store if different from current
        if request.store_name != "default":
            if not rag_system.load_existing_store(request.store_name):
                raise HTTPException(
                    status_code=404, 
                    detail=f"Vector store '{request.store_name}' not found"
                )
        
        response = rag_system.query(
            question=request.question,
            include_sources=request.include_sources,
            return_source_documents=request.return_source_documents
        )
        
        return {
            "success": True,
            "data": response,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Batch query endpoint
@app.post("/api/batch-query")
async def batch_query_documents(request: BatchQueryRequest):
    """Query the RAG system with multiple questions."""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not rag_system.rag_chain:
        raise HTTPException(
            status_code=400, 
            detail="No documents processed yet. Upload documents first."
        )
    
    try:
        responses = rag_system.batch_query(
            questions=request.questions,
            include_sources=request.include_sources
        )
        
        return {
            "success": True,
            "data": responses,
            "total_questions": len(request.questions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing batch query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Conversation history endpoint
@app.get("/api/conversation-history")
async def get_conversation_history():
    """Get conversation history."""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    history = rag_system.export_conversation_history()
    
    return {
        "success": True,
        "data": history,
        "total_exchanges": len(history),
        "timestamp": datetime.now().isoformat()
    }


# Clear conversation endpoint
@app.delete("/api/conversation-history")
async def clear_conversation_history():
    """Clear conversation history."""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    rag_system.clear_conversation()
    
    return {
        "success": True,
        "message": "Conversation history cleared",
        "timestamp": datetime.now().isoformat()
    }


# Benchmark endpoint
@app.post("/api/benchmark")
async def benchmark_system(
    test_questions: Optional[List[str]] = None
):
    """Benchmark system performance."""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        benchmark_results = rag_system.benchmark_system(test_questions)
        
        return {
            "success": True,
            "data": benchmark_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error running benchmark: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Notification endpoint (for n8n workflow)
@app.post("/api/notify")
async def receive_notification(
    event: str,
    data: Dict[str, Any],
    timestamp: str
):
    """Receive notifications from external systems (like n8n)."""
    logger.info(f"Received notification: {event} at {timestamp}")
    logger.debug(f"Notification data: {data}")
    
    # Here you could implement notification handling logic
    # e.g., update databases, send alerts, etc.
    
    return {
        "success": True,
        "message": "Notification received",
        "event": event,
        "processed_at": datetime.now().isoformat()
    }


# Vector stores management
@app.get("/api/vector-stores")
async def list_vector_stores():
    """List available vector stores."""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    # List all vector store directories
    vector_stores = []
    store_path = settings.vector_store_path
    
    if store_path.exists():
        for item in store_path.iterdir():
            if item.is_dir():
                vector_stores.append({
                    "name": item.name,
                    "path": str(item),
                    "created": datetime.fromtimestamp(item.stat().st_ctime).isoformat(),
                    "size": sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                })
    
    return {
        "success": True,
        "data": vector_stores,
        "total_stores": len(vector_stores),
        "timestamp": datetime.now().isoformat()
    }


# Utility functions
def cleanup_temp_files(file_paths: List[Path]):
    """Clean up temporary files."""
    for file_path in file_paths:
        try:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {file_path}: {str(e)}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "src.app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development,
        log_level=settings.log_level.lower()
    )