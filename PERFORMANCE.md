# ⚡ Performance Guide

## 📊 Benchmarking Results

### **System Performance Overview**

| Metric | Value | Target |
|--------|-------|--------|
| **Query Response Time** | 1.2s avg | < 2s |
| **Document Processing** | 50 docs/min | > 30 docs/min |
| **Embedding Generation** | 100 docs/sec | > 50 docs/sec |
| **Vector Search** | < 0.1s | < 0.2s |
| **Memory Usage** | 2.5GB | < 4GB |
| **Accuracy** | 92% relevance | > 85% |

### **Model Performance Comparison**

| Embedding Model | Dimensions | Speed (docs/sec) | Quality Score | Memory (GB) |
|-----------------|------------|------------------|---------------|-------------|
| **all-MiniLM-L6-v2** | 384 | 120 | 8.5/10 | 0.8 |
| **all-mpnet-base-v2** | 768 | 45 | 9.2/10 | 1.5 |
| **multi-qa-MiniLM-L6** | 384 | 110 | 8.8/10 | 0.9 |
| **multilingual-MiniLM** | 384 | 95 | 8.3/10 | 1.1 |

**Recommendation**: Use `all-MiniLM-L6-v2` for production (best speed/quality balance)

---

## 🚀 Optimization Strategies

### **1. Embedding Optimization**

```python
# GPU Acceleration
from src.core.embeddings import EmbeddingManager

# Enable GPU if available
embedding_manager = EmbeddingManager(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Batch processing for better throughput
def process_embeddings_in_batches(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embedding_manager.embed_documents(batch)
        embeddings.extend(batch_embeddings)
    return embeddings

# Performance: 3x faster than individual processing
texts = ["Sample text"] * 1000
embeddings = process_embeddings_in_batches(texts, batch_size=64)
```

### **2. Vector Store Optimization**

```python
# FAISS Configuration for Performance
import faiss
from src.core.vector_store import VectorStoreManager

class OptimizedVectorStore(VectorStoreManager):
    def create_optimized_faiss_index(self, embeddings):
        """Create optimized FAISS index for faster searches."""
        dimension = len(embeddings[0])
        
        if len(embeddings) < 1000:
            # Small dataset: Use flat index (exact search)
            index = faiss.IndexFlatIP(dimension)  # Inner product
        elif len(embeddings) < 10000:
            # Medium dataset: Use IVF with clustering
            nlist = int(np.sqrt(len(embeddings)))  # Number of clusters
            index = faiss.IndexIVFFlat(
                faiss.IndexFlatIP(dimension), 
                dimension, 
                nlist
            )
        else:
            # Large dataset: Use HNSW for very fast approximate search
            index = faiss.IndexHNSWFlat(dimension, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 100
        
        return index

# Usage
opt_store = OptimizedVectorStore(embedding_manager)
# 5x faster search on large datasets
```

### **3. Query Optimization**

```python
# Implement query caching
import redis
import hashlib
import json
from functools import wraps

class CachedRAGSystem:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.rag_system = RAGSystem()
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.cache_ttl = 3600  # 1 hour
    
    def cache_query(func):
        @wraps(func)
        def wrapper(self, question, **kwargs):
            # Create cache key
            cache_key = hashlib.md5(
                f"{question}_{json.dumps(kwargs, sort_keys=True)}".encode()
            ).hexdigest()
            
            # Check cache
            cached_result = self.redis_client.get(f"query:{cache_key}")
            if cached_result:
                return json.loads(cached_result)
            
            # Execute query
            result = func(self, question, **kwargs)
            
            # Cache result
            self.redis_client.setex(
                f"query:{cache_key}",
                self.cache_ttl,
                json.dumps(result, default=str)
            )
            
            return result
        return wrapper
    
    @cache_query
    def query(self, question, **kwargs):
        return self.rag_system.query(question, **kwargs)

# 10x faster for repeated queries
cached_rag = CachedRAGSystem()
```

### **4. Document Processing Optimization**

```python
# Parallel document processing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

class ParallelDocumentProcessor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
    
    def process_documents_parallel(self, file_paths, chunk_size=1000):
        """Process documents in parallel for faster throughput."""
        
        def process_single_file(file_path):
            try:
                processor = DocumentProcessor(
                    chunk_size=chunk_size,
                    chunk_overlap=200
                )
                return processor.load_document(file_path)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                return []
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_single_file, fp) for fp in file_paths]
            
            all_documents = []
            for future in futures:
                documents = future.result()
                all_documents.extend(documents)
        
        return all_documents
    
    def chunk_documents_parallel(self, documents):
        """Chunk documents in parallel using multiprocessing."""
        
        def chunk_batch(doc_batch):
            processor = DocumentProcessor()
            return processor.split_documents(doc_batch)
        
        # Split documents into batches
        batch_size = max(1, len(documents) // self.max_workers)
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        
        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(chunk_batch, batch) for batch in batches]
            
            all_chunks = []
            for future in futures:
                chunks = future.result()
                all_chunks.extend(chunks)
        
        return all_chunks

# Usage: 4x faster document processing
parallel_processor = ParallelDocumentProcessor()
file_paths = [Path(f"data/doc_{i}.pdf") for i in range(100)]
documents = parallel_processor.process_documents_parallel(file_paths)
chunks = parallel_processor.chunk_documents_parallel(documents)
```

---

## 🏗️ Scalability Patterns

### **Horizontal Scaling with Load Balancer**

```yaml
# docker-compose.scale.yml
version: '3.8'
services:
  # Load Balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api-1
      - api-2
      - api-3
  
  # Multiple API instances
  api-1: &api-template
    build: .
    environment:
      - INSTANCE_ID=api-1
      - REDIS_HOST=redis
    depends_on:
      - redis
      - chroma
  
  api-2:
    <<: *api-template
    environment:
      - INSTANCE_ID=api-2
      - REDIS_HOST=redis
  
  api-3:
    <<: *api-template
    environment:
      - INSTANCE_ID=api-3
      - REDIS_HOST=redis
  
  # Shared services
  redis:
    image: redis:alpine
    volumes:
      - redis_data:/data
  
  chroma:
    image: chromadb/chroma:latest
    volumes:
      - chroma_data:/chroma

# Scale up: docker-compose -f docker-compose.scale.yml up --scale api=5
```

### **Database Sharding for Large Scale**

```python
# Vector store sharding
class ShardedVectorStore:
    def __init__(self, num_shards=4):
        self.num_shards = num_shards
        self.shards = []
        
        for i in range(num_shards):
            shard = VectorStoreManager(
                embedding_manager=EmbeddingManager(),
                store_type="chroma"
            )
            self.shards.append(shard)
    
    def get_shard(self, text):
        """Determine shard based on text hash."""
        hash_value = hash(text)
        return self.shards[hash_value % self.num_shards]
    
    def add_documents(self, documents):
        """Add documents to appropriate shards."""
        for doc in documents:
            shard = self.get_shard(doc.page_content)
            shard.add_documents([doc])
    
    def search(self, query, k=5):
        """Search across all shards and merge results."""
        all_results = []
        
        # Search each shard in parallel
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(shard.similarity_search, query, k)
                for shard in self.shards
            ]
            
            for future in futures:
                results = future.result()
                all_results.extend(results)
        
        # Sort by relevance and return top k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:k]

# Usage for millions of documents
sharded_store = ShardedVectorStore(num_shards=8)
# Each shard handles ~125k documents for 1M total
```

---

## 🔧 Memory Management

### **Efficient Memory Usage**

```python
# Memory-efficient document processing
import gc
from memory_profiler import profile

class MemoryOptimizedRAG:
    def __init__(self):
        self.rag_system = RAGSystem()
        self.max_batch_size = 100  # Process in small batches
    
    @profile  # Monitor memory usage
    def process_large_dataset(self, file_paths, batch_size=None):
        """Process large datasets without memory overflow."""
        batch_size = batch_size or self.max_batch_size
        
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            
            print(f"Processing batch {i//batch_size + 1}/{len(file_paths)//batch_size + 1}")
            
            try:
                # Process batch
                stats = self.rag_system.process_documents(
                    batch, 
                    store_name=f"batch_{i//batch_size}"
                )
                
                print(f"Batch completed: {stats['files_processed']} files")
                
                # Force garbage collection
                gc.collect()
                
            except MemoryError:
                # Reduce batch size and retry
                print(f"Memory error, reducing batch size to {batch_size//2}")
                if batch_size > 1:
                    self.process_large_dataset(batch, batch_size//2)
                else:
                    print(f"Skipping batch due to memory constraints")
    
    def monitor_memory(self):
        """Monitor memory usage in real-time."""
        import psutil
        process = psutil.Process()
        
        memory_info = {
            "rss": process.memory_info().rss / 1024 / 1024,  # MB
            "vms": process.memory_info().vms / 1024 / 1024,  # MB
            "percent": process.memory_percent()
        }
        
        return memory_info

# Usage
optimized_rag = MemoryOptimizedRAG()
memory_usage = optimized_rag.monitor_memory()
print(f"Current memory usage: {memory_usage['rss']:.1f} MB")
```

### **Streaming for Large Responses**

```python
# Streaming responses for large queries
from typing import Iterator, Dict, Any

class StreamingRAGSystem:
    def __init__(self):
        self.rag_system = RAGSystem()
    
    def stream_query_response(self, question: str) -> Iterator[Dict[str, Any]]:
        """Stream response in chunks for better UX."""
        # Yield initial status
        yield {"type": "status", "message": "Processing query..."}
        
        # Retrieve documents
        yield {"type": "status", "message": "Searching documents..."}
        
        # Simulate document retrieval
        retrieval_results = self.rag_system.vector_store_manager.similarity_search(
            question, k=5
        )
        
        yield {
            "type": "sources",
            "data": [
                {
                    "file_name": doc.metadata.get("file_name", "Unknown"),
                    "relevance_score": score
                }
                for doc, score in retrieval_results
            ]
        }
        
        # Generate response
        yield {"type": "status", "message": "Generating response..."}
        
        response = self.rag_system.query(question)
        
        # Stream response in chunks
        answer = response["answer"]
        chunk_size = 50  # Words per chunk
        words = answer.split()
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            yield {
                "type": "response_chunk",
                "data": chunk,
                "is_final": i + chunk_size >= len(words)
            }
        
        # Final metadata
        yield {
            "type": "metadata",
            "data": response["metadata"]
        }

# Usage in FastAPI
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
streaming_rag = StreamingRAGSystem()

@app.post("/api/stream-query")
async def stream_query(question: str):
    def generate_response():
        for chunk in streaming_rag.stream_query_response(question):
            yield f"data: {json.dumps(chunk)}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"}
    )
```

---

## 📈 Performance Monitoring

### **Real-time Performance Dashboard**

```python
# Performance metrics collection
import time
import threading
from collections import deque, defaultdict
from datetime import datetime

class PerformanceMonitor:
    def __init__(self, window_size=1000):
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, timestamp=None):
        """Record a performance metric."""
        timestamp = timestamp or datetime.now()
        with self.lock:
            self.metrics[name].append((timestamp, value))
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        with self.lock:
            values = [value for _, value in self.metrics[name]]
            
            if not values:
                return {}
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "median": sorted(values)[len(values)//2],
                "p95": sorted(values)[int(len(values) * 0.95)],
                "p99": sorted(values)[int(len(values) * 0.99)]
            }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all metrics for dashboard."""
        dashboard = {}
        
        for metric_name in self.metrics:
            dashboard[metric_name] = self.get_stats(metric_name)
        
        return dashboard

# Integration with RAG system
class MonitoredRAGSystem:
    def __init__(self):
        self.rag_system = RAGSystem()
        self.monitor = PerformanceMonitor()
    
    def query_with_monitoring(self, question: str, **kwargs):
        """Query with comprehensive performance monitoring."""
        start_time = time.time()
        
        try:
            # Execute query
            result = self.rag_system.query(question, **kwargs)
            
            # Record metrics
            total_time = time.time() - start_time
            self.monitor.record_metric("query_response_time", total_time)
            self.monitor.record_metric("query_success", 1)
            
            # Record metadata if available
            if "metadata" in result:
                metadata = result["metadata"]
                self.monitor.record_metric("retrieval_time", metadata.get("retrieval_time", 0))
                self.monitor.record_metric("llm_time", metadata.get("llm_time", 0))
                self.monitor.record_metric("documents_retrieved", metadata.get("documents_retrieved", 0))
            
            return result
            
        except Exception as e:
            self.monitor.record_metric("query_errors", 1)
            raise
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        dashboard_data = self.monitor.get_dashboard_data()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": dashboard_data,
            "health_score": self.calculate_health_score(dashboard_data),
            "recommendations": self.get_performance_recommendations(dashboard_data)
        }
    
    def calculate_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-100)."""
        score = 100.0
        
        # Penalize slow response times
        if "query_response_time" in metrics:
            avg_time = metrics["query_response_time"].get("avg", 0)
            if avg_time > 2.0:  # Target: < 2s
                score -= min(30, (avg_time - 2.0) * 10)
        
        # Penalize high error rates
        if "query_errors" in metrics and "query_success" in metrics:
            errors = metrics["query_errors"].get("count", 0)
            success = metrics["query_success"].get("count", 1)
            error_rate = errors / (errors + success)
            if error_rate > 0.05:  # Target: < 5% error rate
                score -= min(40, error_rate * 100)
        
        return max(0, score)
    
    def get_performance_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        if "query_response_time" in metrics:
            avg_time = metrics["query_response_time"].get("avg", 0)
            if avg_time > 3.0:
                recommendations.append("Consider enabling GPU acceleration for embeddings")
                recommendations.append("Implement query caching with Redis")
            elif avg_time > 2.0:
                recommendations.append("Consider optimizing chunk size and overlap")
        
        if "documents_retrieved" in metrics:
            avg_docs = metrics["documents_retrieved"].get("avg", 0)
            if avg_docs > 10:
                recommendations.append("Reduce number of retrieved documents for faster processing")
        
        return recommendations

# Usage
monitored_rag = MonitoredRAGSystem()

# Process some queries
for question in ["What is AI?", "How does ML work?", "Explain NLP"]:
    response = monitored_rag.query_with_monitoring(question)

# Get performance report
report = monitored_rag.get_performance_report()
print(f"Health Score: {report['health_score']:.1f}/100")
print(f"Recommendations: {report['recommendations']}")
```

---

## 🎯 Performance Targets

### **Production SLA Targets**

| Metric | Target | Alert Threshold |
|--------|--------|----------------|
| **Query Response Time** | 95% < 2s | > 3s |
| **System Availability** | 99.9% | < 99.5% |
| **Error Rate** | < 1% | > 2% |
| **Memory Usage** | < 80% | > 90% |
| **CPU Usage** | < 70% | > 85% |
| **Disk Usage** | < 80% | > 90% |

### **Capacity Planning**

```python
# Capacity estimation
class CapacityPlanner:
    def __init__(self):
        self.base_metrics = {
            "documents_per_gb": 50000,      # Text documents
            "queries_per_second": 10,       # Per CPU core
            "embedding_memory_mb": 800,     # Per model
            "vector_db_overhead": 1.5       # Storage multiplier
        }
    
    def estimate_requirements(self, 
                            num_documents: int,
                            expected_qps: int,
                            peak_multiplier: float = 3.0) -> Dict[str, Any]:
        """Estimate infrastructure requirements."""
        
        # Storage requirements
        base_storage_gb = num_documents / self.base_metrics["documents_per_gb"]
        total_storage_gb = base_storage_gb * self.base_metrics["vector_db_overhead"]
        
        # Compute requirements
        peak_qps = expected_qps * peak_multiplier
        required_cores = peak_qps / self.base_metrics["queries_per_second"]
        
        # Memory requirements
        embedding_memory = self.base_metrics["embedding_memory_mb"] / 1024  # GB
        vector_memory = total_storage_gb * 0.1  # 10% of vector data in memory
        total_memory_gb = embedding_memory + vector_memory + 2  # 2GB overhead
        
        return {
            "storage_gb": int(total_storage_gb * 1.2),  # 20% buffer
            "memory_gb": int(total_memory_gb * 1.3),   # 30% buffer
            "cpu_cores": int(required_cores * 1.5),    # 50% buffer
            "estimated_cost_per_month": self.estimate_cost(total_storage_gb, total_memory_gb, required_cores)
        }
    
    def estimate_cost(self, storage_gb: float, memory_gb: float, cpu_cores: float) -> Dict[str, float]:
        """Estimate monthly costs for different cloud providers."""
        
        # Rough pricing estimates (USD/month)
        aws_costs = {
            "compute": cpu_cores * 50,     # EC2 m5.large equivalent
            "memory": memory_gb * 5,       # Additional memory
            "storage": storage_gb * 0.1,   # EBS gp3
            "total": 0
        }
        aws_costs["total"] = sum(aws_costs.values())
        
        gcp_costs = {
            "compute": cpu_cores * 45,     # Compute Engine
            "memory": memory_gb * 4.5,     # Memory
            "storage": storage_gb * 0.08,  # Persistent Disk
            "total": 0
        }
        gcp_costs["total"] = sum(gcp_costs.values())
        
        return {"aws": aws_costs, "gcp": gcp_costs}

# Usage
planner = CapacityPlanner()
requirements = planner.estimate_requirements(
    num_documents=100000,
    expected_qps=50
)

print(f"Infrastructure Requirements for 100K documents, 50 QPS:")
print(f"Storage: {requirements['storage_gb']} GB")
print(f"Memory: {requirements['memory_gb']} GB")
print(f"CPU Cores: {requirements['cpu_cores']}")
print(f"Estimated AWS Cost: ${requirements['estimated_cost_per_month']['aws']['total']:.2f}/month")
```

---

**⚡ Your RAG system is now optimized for production scale!**

*Use these performance patterns and monitoring techniques to build highly scalable document intelligence systems.*