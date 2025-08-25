# 📚 Usage Examples

## Quick Start Guide

### 1. **Basic Document Processing**

```python
from src.core.rag_system import RAGSystem

# Initialize the system
rag_system = RAGSystem(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    vector_store_type="faiss",
    llm_provider="openai"
)

# Process documents
document_paths = [
    "data/company_handbook.pdf",
    "data/technical_specs.docx",
    "data/meeting_notes.txt"
]

stats = rag_system.process_documents(
    document_paths, 
    store_name="company_docs"
)

print(f"Processed {stats['files_processed']} files")
print(f"Created {stats['total_documents']} chunks")
```

### 2. **Ask Questions**

```python
# Single query
response = rag_system.query(
    question="What is our remote work policy?",
    include_sources=True
)

print(f"Answer: {response['answer']}")
print(f"Sources: {[s['file_name'] for s in response['sources']]}")
print(f"Response time: {response['metadata']['total_time']:.2f}s")

# Batch queries
questions = [
    "What are the key product features?",
    "Who are our main competitors?",
    "What is the project timeline?"
]

responses = rag_system.batch_query(questions)
for i, resp in enumerate(responses):
    print(f"Q{i+1}: {resp['question']}")
    print(f"A{i+1}: {resp['answer'][:100]}...\n")
```

### 3. **Web Interface Usage**

```bash
# Start the web interface
streamlit run src/app/streamlit_app.py
```

1. **Upload Documents**: Drag & drop PDFs, Word docs, or text files
2. **Process**: Click "Process Documents" and wait for completion
3. **Chat**: Ask questions in natural language
4. **Review Sources**: See which documents were used for each answer

---

## 🏢 Enterprise Use Cases

### **Legal Document Analysis**

```python
# Specialized setup for legal documents
legal_rag = RAGSystem(
    embedding_model="sentence-transformers/all-mpnet-base-v2",  # Higher quality
    vector_store_type="chroma",  # Better for large datasets
    llm_provider="anthropic"  # Claude for nuanced understanding
)

# Process legal documents
legal_docs = [
    "contracts/service_agreement_2024.pdf",
    "contracts/privacy_policy.pdf",
    "legal/compliance_guide.docx"
]

legal_rag.process_documents(legal_docs, store_name="legal_kb")

# Legal-specific queries
legal_questions = [
    "What are the termination clauses in our service agreements?",
    "What data retention policies do we have?",
    "Are there any compliance requirements for EU customers?"
]

for question in legal_questions:
    response = legal_rag.query(question, include_sources=True)
    print(f"\n🏛️ Legal Query: {question}")
    print(f"📋 Answer: {response['answer']}")
    print(f"📄 Sources: {', '.join([s['file_name'] for s in response['sources']])}")
```

### **Technical Documentation Search**

```python
# Setup for technical documentation
tech_rag = RAGSystem(
    embedding_model="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",  # Optimized for Q&A
    llm_provider="openai"
)

# Process technical docs
tech_docs = [
    "docs/api_documentation.md",
    "docs/architecture_guide.pdf",
    "docs/troubleshooting.txt",
    "docs/deployment_guide.docx"
]

tech_rag.process_documents(tech_docs, store_name="tech_docs")

# Technical queries with code context
tech_queries = [
    "How do I authenticate with the API?",
    "What's the recommended deployment architecture?",
    "How to troubleshoot connection timeouts?",
    "What are the rate limits for API calls?"
]

for query in tech_queries:
    response = tech_rag.query(
        question=query,
        include_sources=True,
        return_source_documents=True  # Get full context
    )
    
    print(f"\n🔧 Tech Query: {query}")
    print(f"💡 Answer: {response['answer']}")
    
    # Show code snippets if available
    for doc in response.get('source_documents', [])[:2]:
        if any(lang in doc.page_content.lower() for lang in ['python', 'javascript', 'curl']):
            print(f"📝 Code Context: {doc.page_content[:200]}...")
```

### **HR Knowledge Base**

```python
# HR-specific configuration
hr_rag = RAGSystem(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Fast for frequent queries
    vector_store_type="faiss"
)

# Process HR documents
hr_docs = [
    "hr/employee_handbook.pdf",
    "hr/benefits_guide.docx",
    "hr/vacation_policy.txt",
    "hr/performance_review_process.md"
]

hr_rag.process_documents(hr_docs, store_name="hr_kb")

# Common HR queries
hr_scenarios = [
    ("New Employee", "What benefits am I eligible for as a new hire?"),
    ("Time Off", "How do I request vacation time?"),
    ("Performance", "When are performance reviews conducted?"),
    ("Policy", "What is the dress code policy?"),
    ("Benefits", "How does the health insurance plan work?")
]

for category, question in hr_scenarios:
    response = hr_rag.query(question)
    print(f"\n👥 {category}: {question}")
    print(f"✅ HR Response: {response['answer']}")
    print(f"⏱️ Response Time: {response['metadata']['total_time']:.2f}s")
```

---

## 🔧 Advanced Configuration Examples

### **Multi-Language Support**

```python
# Setup for multilingual documents
multilang_rag = RAGSystem(
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    vector_store_type="chroma"
)

# Process documents in multiple languages
multi_docs = [
    "docs/guide_en.pdf",      # English
    "docs/guide_es.pdf",      # Spanish
    "docs/guide_fr.pdf",      # French
    "docs/guide_de.pdf"       # German
]

multilang_rag.process_documents(multi_docs, store_name="multilang_kb")

# Query in different languages
questions = [
    ("English", "What are the main features of this product?"),
    ("Spanish", "¿Cuáles son las características principales de este producto?"),
    ("French", "Quelles sont les principales caractéristiques de ce produit?"),
    ("German", "Was sind die Hauptmerkmale dieses Produkts?")
]

for lang, question in questions:
    response = multilang_rag.query(question)
    print(f"\n🌍 {lang}: {question}")
    print(f"📝 Answer: {response['answer']}")
```

### **Custom Embedding Models**

```python
from src.core.embeddings import EmbeddingManager

# Compare different embedding models
models_to_test = [
    "sentence-transformers/all-MiniLM-L6-v2",      # Fast, lightweight
    "sentence-transformers/all-mpnet-base-v2",      # High quality
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"  # Q&A optimized
]

for model_name in models_to_test:
    print(f"\n🧠 Testing Model: {model_name}")
    
    # Create system with specific model
    rag = RAGSystem(embedding_model=model_name)
    rag.process_documents(["data/test_document.pdf"])
    
    # Benchmark performance
    test_questions = [
        "What is the main topic of this document?",
        "What are the key recommendations?"
    ]
    
    benchmark = rag.benchmark_system(test_questions)
    
    print(f"📊 Performance:")
    print(f"   Avg Response Time: {benchmark['e2e_benchmark']['average_response_time']:.2f}s")
    print(f"   Embedding Speed: {benchmark['embedding_benchmark']['documents_per_second']:.1f} docs/sec")
    print(f"   Model Dimensions: {rag.embedding_manager.get_model_info()['dimensions']}")
```

### **Custom Chunking Strategies**

```python
from src.core.document_processor import DocumentProcessor
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SpacyTextSplitter
)

# Strategy 1: Semantic-based chunking
semantic_processor = DocumentProcessor(
    chunk_size=800,
    chunk_overlap=150,
    separators=[
        "\n\n\n",  # Section breaks
        "\n\n",    # Paragraph breaks  
        "\n",      # Line breaks
        ". ",      # Sentence endings
        ", ",      # Phrase breaks
        " ",       # Word breaks
    ]
)

# Strategy 2: Token-based chunking (for precise token limits)
token_processor = DocumentProcessor(
    chunk_size=1000,  # tokens, not characters
    chunk_overlap=100
)

# Custom text splitter integration
token_splitter = TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    model_name="gpt-4"  # Use GPT-4 tokenizer
)
token_processor.text_splitter = token_splitter

# Compare chunking strategies
doc_path = "data/large_document.pdf"

for name, processor in [("Semantic", semantic_processor), ("Token", token_processor)]:
    chunks = processor.process_documents([doc_path])
    stats = processor.get_document_stats(chunks)
    
    print(f"\n📝 {name} Chunking Strategy:")
    print(f"   Total Chunks: {stats['total_documents']}")
    print(f"   Avg Chunk Size: {stats['average_chunk_size']:.0f}")
    print(f"   Total Characters: {stats['total_characters']:,}")
```

---

## 🔌 API Integration Examples

### **REST API Usage**

```python
import requests
import json

API_BASE = "http://localhost:8000"

# 1. Upload and process documents
files = [
    ('files', ('document1.pdf', open('data/doc1.pdf', 'rb'), 'application/pdf')),
    ('files', ('document2.txt', open('data/doc2.txt', 'rb'), 'text/plain'))
]

process_response = requests.post(
    f"{API_BASE}/api/process-documents",
    files=files,
    data={
        'store_name': 'api_test',
        'chunk_size': 1000,
        'chunk_overlap': 200
    }
)

print(f"Processing Status: {process_response.json()['success']}")
print(f"Files Processed: {process_response.json()['statistics']['files_processed']}")

# 2. Query the system
query_data = {
    "question": "What are the main topics covered in these documents?",
    "include_sources": True,
    "store_name": "api_test"
}

query_response = requests.post(
    f"{API_BASE}/api/query",
    json=query_data
)

result = query_response.json()
print(f"\nAnswer: {result['data']['answer']}")
print(f"Sources: {len(result['data']['sources'])} documents referenced")

# 3. Batch queries
batch_data = {
    "questions": [
        "What is the main purpose of these documents?",
        "Are there any action items mentioned?",
        "What are the key dates or deadlines?"
    ],
    "include_sources": True
}

batch_response = requests.post(
    f"{API_BASE}/api/batch-query",
    json=batch_data
)

batch_results = batch_response.json()
for i, result in enumerate(batch_results['data']):
    print(f"\nQ{i+1}: {result['question']}")
    print(f"A{i+1}: {result['answer'][:100]}...")

# 4. Get system status
status_response = requests.get(f"{API_BASE}/api/status")
status = status_response.json()
print(f"\nSystem Status: {status['status']}")
print(f"Documents Loaded: {status['system_info'].get('vector_store_loaded', False)}")

# 5. Performance benchmark
benchmark_response = requests.post(
    f"{API_BASE}/api/benchmark",
    json={
        "test_questions": [
            "What is this about?",
            "Summarize the key points",
            "What are the main findings?"
        ]
    }
)

benchmark = benchmark_response.json()
print(f"\nPerformance Benchmark:")
print(f"Avg Response Time: {benchmark['data']['e2e_benchmark']['average_response_time']:.2f}s")
print(f"Queries/Second: {benchmark['data']['e2e_benchmark']['queries_per_second']:.2f}")
```

### **Webhook Integration (n8n)**

```javascript
// n8n webhook node configuration for automatic document processing
{
  "nodes": [
    {
      "name": "Document Upload Webhook",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "path": "document-upload",
        "httpMethod": "POST"
      }
    },
    {
      "name": "Process with RAG System",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "http://rag-api:8000/api/process-documents",
        "method": "POST",
        "bodyParameters": {
          "files": "={{ $json.files }}",
          "store_name": "automated_upload",
          "processing_options": {
            "chunk_size": 1000,
            "include_metadata": true
          }
        }
      }
    },
    {
      "name": "Send Notification",
      "type": "n8n-nodes-base.slack",
      "parameters": {
        "channel": "#ai-updates",
        "text": "📄 New documents processed: {{ $json.files_processed }} files, {{ $json.total_documents }} chunks created"
      }
    }
  ]
}
```

---

## 🎛️ Customization Examples

### **Custom Prompt Templates**

```python
from langchain.prompts import ChatPromptTemplate

# Create custom RAG prompt for specific domain
custom_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a specialized AI assistant for technical documentation analysis.

Instructions:
1. Focus on technical accuracy and precision
2. Always provide code examples when available
3. Mention version numbers and compatibility information
4. If information is unclear, ask for clarification
5. Structure responses with clear headings and bullet points

Context Documents:
{context}

Previous Conversation:
{chat_history}
"""),
    ("human", "{question}")
])

# Apply custom prompt to RAG chain
rag_system = RAGSystem()
# Note: In actual implementation, you'd modify the RAGChain class
# to accept custom prompts via the constructor
```

### **Custom Document Filters**

```python
from pathlib import Path
from datetime import datetime, timedelta

class SmartDocumentProcessor:
    """Enhanced document processor with smart filtering."""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
    
    def process_with_filters(
        self,
        directory: Path,
        file_age_days: int = 30,
        min_file_size: int = 1000,
        exclude_patterns: list = None
    ):
        """Process documents with intelligent filtering."""
        
        exclude_patterns = exclude_patterns or ['temp_', 'draft_', '.tmp']
        cutoff_date = datetime.now() - timedelta(days=file_age_days)
        
        valid_files = []
        
        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue
                
            # Filter by extension
            if file_path.suffix.lower() not in ['.pdf', '.docx', '.txt', '.md']:
                continue
                
            # Filter by age
            if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_date:
                continue
                
            # Filter by size
            if file_path.stat().st_size < min_file_size:
                continue
                
            # Filter by patterns
            if any(pattern in file_path.name.lower() for pattern in exclude_patterns):
                continue
                
            valid_files.append(file_path)
        
        print(f"📁 Found {len(valid_files)} valid documents")
        
        # Process in batches to avoid memory issues
        batch_size = 10
        processed_stats = []
        
        for i in range(0, len(valid_files), batch_size):
            batch = valid_files[i:i + batch_size]
            print(f"📄 Processing batch {i//batch_size + 1}: {len(batch)} files")
            
            try:
                stats = self.rag_system.process_documents(
                    batch, 
                    store_name=f"filtered_docs_batch_{i//batch_size + 1}"
                )
                processed_stats.append(stats)
            except Exception as e:
                print(f"⚠️ Error processing batch {i//batch_size + 1}: {e}")
                continue
        
        return processed_stats

# Usage
smart_processor = SmartDocumentProcessor(rag_system)
stats = smart_processor.process_with_filters(
    directory=Path("data/documents"),
    file_age_days=7,  # Only recent files
    min_file_size=5000,  # At least 5KB
    exclude_patterns=['draft_', 'temp_', 'backup_']
)
```

### **Performance Monitoring Integration**

```python
import time
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
query_counter = Counter('rag_queries_total', 'Total queries processed')
response_time_histogram = Histogram('rag_response_time_seconds', 'Response time distribution')
active_sessions = Gauge('rag_active_sessions', 'Number of active user sessions')
error_counter = Counter('rag_errors_total', 'Total errors', ['error_type'])

class MonitoredRAGSystem:
    """RAG System with comprehensive monitoring."""
    
    def __init__(self, *args, **kwargs):
        self.rag_system = RAGSystem(*args, **kwargs)
        self.active_queries = 0
    
    def monitor_performance(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            query_counter.inc()
            active_sessions.inc()
            
            start_time = time.time()
            
            try:
                result = func(self, *args, **kwargs)
                return result
            except Exception as e:
                error_counter.labels(error_type=type(e).__name__).inc()
                raise
            finally:
                response_time = time.time() - start_time
                response_time_histogram.observe(response_time)
                active_sessions.dec()
        
        return wrapper
    
    @monitor_performance
    def query(self, question: str, **kwargs):
        """Monitored query method."""
        return self.rag_system.query(question, **kwargs)
    
    @monitor_performance
    def process_documents(self, file_paths, **kwargs):
        """Monitored document processing."""
        return self.rag_system.process_documents(file_paths, **kwargs)
    
    def get_performance_metrics(self):
        """Get current performance metrics."""
        return {
            'total_queries': query_counter._value._value,
            'active_sessions': active_sessions._value._value,
            'avg_response_time': response_time_histogram._sum._value / max(response_time_histogram._count._value, 1),
            'error_rate': sum(metric._value._value for metric in error_counter._metrics.values()) / max(query_counter._value._value, 1)
        }

# Usage with monitoring
monitored_rag = MonitoredRAGSystem(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    vector_store_type="faiss"
)

# Process documents with monitoring
monitored_rag.process_documents(["data/sample.pdf"])

# Query with monitoring
response = monitored_rag.query("What is this document about?")

# Get performance insights
metrics = monitored_rag.get_performance_metrics()
print(f"📊 Performance Metrics: {metrics}")
```

---

## 🚀 Production Tips

### **Error Handling Best Practices**

```python
import logging
from typing import Optional, Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)

def handle_rag_errors(func):
    """Decorator for comprehensive error handling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return {"error": "Document not found", "success": False}
        except MemoryError as e:
            logger.error(f"Out of memory: {e}")
            return {"error": "Document too large to process", "success": False}
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            return {"error": "Internal processing error", "success": False}
    return wrapper

class ProductionRAGSystem:
    """Production-ready RAG system with robust error handling."""
    
    def __init__(self):
        try:
            self.rag_system = RAGSystem()
            self.is_healthy = True
        except Exception as e:
            logger.critical(f"Failed to initialize RAG system: {e}")
            self.is_healthy = False
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        if not self.is_healthy:
            return {"status": "unhealthy", "issues": ["System initialization failed"]}
        
        issues = []
        
        # Check vector store
        if not hasattr(self.rag_system, 'vector_store_manager') or \
           not self.rag_system.vector_store_manager.vector_store:
            issues.append("No vector store loaded")
        
        # Check LLM availability
        try:
            test_response = self.rag_system.query("test", timeout=5)
        except Exception:
            issues.append("LLM not responding")
        
        return {
            "status": "healthy" if not issues else "degraded",
            "issues": issues,
            "timestamp": datetime.now().isoformat()
        }
    
    @handle_rag_errors
    def safe_query(self, question: str, **kwargs) -> Dict[str, Any]:
        """Query with comprehensive error handling."""
        if not self.is_healthy:
            return {"error": "System not healthy", "success": False}
        
        # Validate input
        if not question or len(question.strip()) == 0:
            return {"error": "Empty question provided", "success": False}
        
        if len(question) > 10000:  # Reasonable limit
            return {"error": "Question too long", "success": False}
        
        # Process query
        result = self.rag_system.query(question, **kwargs)
        result["success"] = True
        return result

# Usage
production_rag = ProductionRAGSystem()

# Always check health before processing
health = production_rag.health_check()
if health["status"] == "healthy":
    response = production_rag.safe_query("What is the main topic?")
    if response["success"]:
        print(f"Answer: {response['answer']}")
    else:
        print(f"Error: {response['error']}")
else:
    print(f"System issues: {health['issues']}")
```

---

**🎯 Ready to build intelligent document systems with LangChain!**

*These examples cover everything from basic usage to advanced production deployments. Choose the patterns that best fit your use case.*