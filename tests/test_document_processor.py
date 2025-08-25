"""Tests for document processor."""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock

from src.core.document_processor import DocumentProcessor
from langchain.schema import Document


class TestDocumentProcessor:
    """Test cases for DocumentProcessor class."""
    
    def test_init_default_settings(self):
        """Test initialization with default settings."""
        processor = DocumentProcessor()
        
        assert processor.chunk_size > 0
        assert processor.chunk_overlap >= 0
        assert len(processor.separators) > 0
        assert processor.text_splitter is not None
    
    def test_init_custom_settings(self):
        """Test initialization with custom settings."""
        processor = DocumentProcessor(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n"]
        )
        
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 100
        assert processor.separators == ["\n\n", "\n"]
    
    def test_supported_extensions(self):
        """Test that supported extensions are properly defined."""
        processor = DocumentProcessor()
        
        expected_extensions = {".pdf", ".docx", ".doc", ".txt", ".md"}
        actual_extensions = set(processor.SUPPORTED_EXTENSIONS.keys())
        
        assert actual_extensions == expected_extensions
    
    def test_load_text_document(self, sample_documents):
        """Test loading a text document."""
        processor = DocumentProcessor()
        txt_file = next(f for f in sample_documents if f.suffix == ".txt")
        
        documents = processor.load_document(txt_file)
        
        assert len(documents) > 0
        assert isinstance(documents[0], Document)
        assert len(documents[0].page_content) > 0
        assert documents[0].metadata["file_name"] == txt_file.name
        assert documents[0].metadata["file_type"] == ".txt"
    
    def test_load_markdown_document(self, sample_documents):
        """Test loading a markdown document."""
        processor = DocumentProcessor()
        md_file = next(f for f in sample_documents if f.suffix == ".md")
        
        documents = processor.load_document(md_file)
        
        assert len(documents) > 0
        assert isinstance(documents[0], Document)
        assert "Document Processing Pipeline" in documents[0].page_content
        assert documents[0].metadata["file_type"] == ".md"
    
    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        processor = DocumentProcessor()
        
        with pytest.raises(FileNotFoundError):
            processor.load_document(Path("/nonexistent/file.txt"))
    
    def test_load_unsupported_file(self, temp_dir):
        """Test loading an unsupported file type."""
        processor = DocumentProcessor()
        
        # Create an unsupported file
        unsupported_file = temp_dir / "test.xyz"
        unsupported_file.write_text("test content")
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            processor.load_document(unsupported_file)
    
    def test_load_multiple_documents(self, sample_documents):
        """Test loading multiple documents."""
        processor = DocumentProcessor()
        
        documents = processor.load_documents(sample_documents)
        
        assert len(documents) >= len(sample_documents)  # May be more due to splitting
        assert all(isinstance(doc, Document) for doc in documents)
    
    def test_split_documents(self, sample_documents):
        """Test document splitting functionality."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        
        # Load documents first
        documents = processor.load_documents(sample_documents)
        
        # Split them
        chunks = processor.split_documents(documents)
        
        assert len(chunks) >= len(documents)  # Should have more chunks than docs
        assert all(isinstance(chunk, Document) for chunk in chunks)
        assert all("chunk_id" in chunk.metadata for chunk in chunks)
        assert all("chunk_size" in chunk.metadata for chunk in chunks)
    
    def test_process_documents_complete_pipeline(self, sample_documents):
        """Test the complete document processing pipeline."""
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        
        chunks = processor.process_documents(sample_documents)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)
        
        # Verify metadata is properly set
        for chunk in chunks:
            assert "file_name" in chunk.metadata
            assert "file_type" in chunk.metadata
            assert "chunk_id" in chunk.metadata
            assert "chunk_size" in chunk.metadata
    
    def test_get_document_stats(self, sample_documents):
        """Test document statistics generation."""
        processor = DocumentProcessor()
        
        chunks = processor.process_documents(sample_documents)
        stats = processor.get_document_stats(chunks)
        
        assert "total_documents" in stats
        assert "total_characters" in stats
        assert "average_chunk_size" in stats
        assert "file_types" in stats
        assert "chunk_size_config" in stats
        assert "chunk_overlap_config" in stats
        
        assert stats["total_documents"] == len(chunks)
        assert stats["total_characters"] > 0
        assert stats["average_chunk_size"] > 0
        assert isinstance(stats["file_types"], dict)
    
    def test_empty_document_list(self):
        """Test handling of empty document lists."""
        processor = DocumentProcessor()
        
        stats = processor.get_document_stats([])
        assert stats == {}
        
        chunks = processor.split_documents([])
        assert chunks == []
    
    @patch('src.core.document_processor.PyPDFLoader')
    def test_pdf_loading_error_handling(self, mock_pdf_loader, temp_dir):
        """Test error handling during PDF loading."""
        processor = DocumentProcessor()
        
        # Create a fake PDF file
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_text("fake pdf content")
        
        # Mock the loader to raise an exception
        mock_pdf_loader.side_effect = Exception("PDF loading failed")
        
        with pytest.raises(Exception, match="PDF loading failed"):
            processor.load_document(pdf_file)