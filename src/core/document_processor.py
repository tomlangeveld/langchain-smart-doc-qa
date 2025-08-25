"""Document processing pipeline for various file formats."""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain.document_loaders.base import BaseLoader

from ..config.settings import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles loading, processing, and chunking of documents."""
    
    SUPPORTED_EXTENSIONS = {
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
    }
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Optimized separators for better semantic chunking
        self.separators = separators or [
            "\n\n\n",  # Multiple newlines (section breaks)
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ". ",      # Sentence endings
            "? ",      # Question endings
            "! ",      # Exclamation endings
            "; ",      # Semicolon breaks
            ", ",      # Comma breaks
            " ",       # Word breaks
            "",        # Character breaks (last resort)
        ]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )
    
    def load_document(self, file_path: Path) -> List[Document]:
        """Load a single document from file path."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )
        
        loader_class = self.SUPPORTED_EXTENSIONS[extension]
        
        try:
            if extension == ".txt":
                loader = loader_class(str(file_path), encoding="utf-8")
            else:
                loader = loader_class(str(file_path))
            
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_type": extension,
                    "file_size": file_path.stat().st_size,
                })
            
            logger.info(f"Loaded {len(documents)} document(s) from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def load_documents(self, file_paths: List[Path]) -> List[Document]:
        """Load multiple documents."""
        all_documents = []
        
        for file_path in file_paths:
            try:
                docs = self.load_document(file_path)
                all_documents.extend(docs)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {str(e)}")
                continue
        
        logger.info(f"Loaded total of {len(all_documents)} documents")
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks for better retrieval."""
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk.page_content),
            })
        
        logger.info(
            f"Split {len(documents)} documents into {len(chunks)} chunks"
        )
        return chunks
    
    def process_documents(self, file_paths: List[Path]) -> List[Document]:
        """Complete pipeline: load and chunk documents."""
        documents = self.load_documents(file_paths)
        if not documents:
            logger.warning("No documents were loaded successfully")
            return []
        
        chunks = self.split_documents(documents)
        return chunks
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about the processed documents."""
        if not documents:
            return {}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        file_types = {}
        
        for doc in documents:
            file_type = doc.metadata.get("file_type", "unknown")
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "average_chunk_size": total_chars / len(documents),
            "file_types": file_types,
            "chunk_size_config": self.chunk_size,
            "chunk_overlap_config": self.chunk_overlap,
        }