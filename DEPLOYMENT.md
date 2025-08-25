# 🚀 Deployment Guide

## Quick Start (5 minutes)

### 1. **Local Development Setup**

```bash
# Clone the repository
git clone https://github.com/tomlangeveld/langchain-smart-doc-qa.git
cd langchain-smart-doc-qa

# Set up Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### 2. **Start the System**

```bash
# Terminal 1: Start FastAPI backend
uvicorn src.app.api:app --reload --port 8000

# Terminal 2: Start Streamlit frontend
streamlit run src/app/streamlit_app.py --server.port 8501
```

**Access Points:**
- 🌐 **Web Interface**: http://localhost:8501
- 📚 **API Documentation**: http://localhost:8000/docs
- 🔧 **API Health Check**: http://localhost:8000/health

---

## 🐳 Docker Deployment (Recommended)

### **Single Command Deployment**

```bash
# Start entire stack
docker-compose up -d

# View logs
docker-compose logs -f
```

**Services Available:**
- **Frontend**: http://localhost:8501 (Streamlit)
- **API**: http://localhost:8000 (FastAPI)
- **Vector DB**: http://localhost:8001 (Chroma)
- **Automation**: http://localhost:5678 (n8n)
- **Monitoring**: http://localhost:9090 (Prometheus)
- **Dashboards**: http://localhost:3000 (Grafana)

### **Production Docker Setup**

```bash
# Build production image
docker build -t rag-system:latest --target production .

# Run with production settings
docker run -d \
  --name rag-system \
  -p 8000:8000 \
  -e APP_ENV=production \
  -e OPENAI_API_KEY=your_key_here \
  -v ./data:/app/data \
  rag-system:latest
```

---

## ☁️ Cloud Deployment Options

### **AWS Deployment**

#### **Option 1: ECS Fargate**
```bash
# Build and push to ECR
aws ecr create-repository --repository-name rag-system
docker build -t rag-system:latest .
docker tag rag-system:latest $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com/rag-system:latest
docker push $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com/rag-system:latest

# Deploy with ECS CLI
ecs-cli compose --project-name rag-system service up
```

#### **Option 2: Lambda + API Gateway**
```bash
# Install serverless framework
npm install -g serverless

# Deploy serverless version
serverless deploy
```

### **GCP Deployment**

#### **Cloud Run**
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/$PROJECT_ID/rag-system

# Deploy to Cloud Run
gcloud run deploy rag-system \
  --image gcr.io/$PROJECT_ID/rag-system \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### **GKE (Kubernetes)**
```bash
# Create GKE cluster
gcloud container clusters create rag-cluster \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10

# Deploy with Helm
helm install rag-system ./helm-chart/
```

### **Azure Deployment**

#### **Container Instances**
```bash
# Create resource group
az group create --name rag-system-rg --location eastus

# Deploy container
az container create \
  --resource-group rag-system-rg \
  --name rag-system \
  --image ragregistry.azurecr.io/rag-system:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000
```

---

## 🔧 Configuration

### **Environment Variables**

```bash
# Core API Keys
OPENAI_API_KEY=sk-...                 # Required for OpenAI models
ANTHROPIC_API_KEY=sk-...              # Required for Claude models
HUGGINGFACE_API_TOKEN=hf_...          # Optional for private models

# Application Settings
APP_ENV=production                     # development|production
LOG_LEVEL=INFO                        # DEBUG|INFO|WARNING|ERROR
MAX_UPLOAD_SIZE=50                    # MB

# Processing Configuration
CHUNK_SIZE=1000                       # Text chunk size
CHUNK_OVERLAP=200                     # Overlap between chunks
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector Database
VECTOR_STORE_TYPE=faiss               # faiss|chroma
CHROMA_HOST=localhost                 # For Chroma deployment
CHROMA_PORT=8000

# LLM Configuration
DEFAULT_LLM=openai                    # openai|anthropic|ollama
OPENAI_MODEL=gpt-4                    # Model name
TEMPERATURE=0.1                       # Response creativity
MAX_TOKENS=2000                       # Response length

# Monitoring (Optional)
LANGSMITH_API_KEY=ls_...              # LangSmith tracing
LANGSMITH_PROJECT=langchain-doc-qa
```

### **Performance Tuning**

```python
# config/performance.py
PERFORMANCE_CONFIG = {
    # Embedding Settings
    "embedding_batch_size": 32,        # Process embeddings in batches
    "embedding_device": "cuda",        # Use GPU if available
    
    # Vector Store Settings
    "faiss_index_type": "IndexIVFFlat", # FAISS index type
    "vector_cache_size": 1000,         # Cache frequently accessed vectors
    
    # LLM Settings
    "llm_max_concurrent": 5,           # Concurrent LLM requests
    "llm_timeout": 30,                 # Request timeout (seconds)
    
    # API Settings
    "api_workers": 4,                  # Uvicorn workers
    "api_max_requests": 100,           # Max requests per minute
}
```

---

## 📊 Monitoring & Observability

### **Health Checks**

```bash
# API Health
curl http://localhost:8000/health

# System Status
curl http://localhost:8000/api/status

# Performance Benchmark
curl -X POST http://localhost:8000/api/benchmark
```

### **Logging Configuration**

```python
# Enhanced logging setup
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "json": {
            "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "detailed"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json"
        }
    },
    "loggers": {
        "src": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False
        }
    }
}
```

### **Metrics Collection**

```python
# Add to your application
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
query_counter = Counter('rag_queries_total', 'Total RAG queries')
response_time = Histogram('rag_response_time_seconds', 'Response time')
active_users = Gauge('rag_active_users', 'Active users')

# Use in endpoints
@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    query_counter.inc()
    
    with response_time.time():
        result = rag_system.query(request.question)
    
    return result
```

---

## 🔐 Security & Production Checklist

### **Security Hardening**

- [ ] **API Keys**: Store in secure vault (AWS Secrets Manager, Azure Key Vault)
- [ ] **HTTPS**: Enable TLS/SSL certificates
- [ ] **Authentication**: Implement JWT or OAuth2
- [ ] **Rate Limiting**: Configure per-user limits
- [ ] **CORS**: Configure allowed origins
- [ ] **Input Validation**: Sanitize all inputs
- [ ] **File Upload**: Validate file types and sizes
- [ ] **Error Handling**: Don't expose internal details

### **Performance Optimization**

- [ ] **Caching**: Implement Redis for query caching
- [ ] **Connection Pooling**: Database connection limits
- [ ] **Async Processing**: Use background tasks
- [ ] **Load Balancing**: Multiple API instances
- [ ] **CDN**: Static asset delivery
- [ ] **Database Indexing**: Optimize vector searches
- [ ] **Memory Management**: Monitor resource usage
- [ ] **GPU Acceleration**: Use CUDA for embeddings

### **Scalability Considerations**

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-system
  template:
    metadata:
      labels:
        app: rag-system
    spec:
      containers:
      - name: rag-system
        image: rag-system:latest
        ports:
        - containerPort: 8000
        env:
        - name: APP_ENV
          value: "production"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

---

## 🚨 Troubleshooting

### **Common Issues**

#### **"API Key Not Found"**
```bash
# Check environment variables
echo $OPENAI_API_KEY

# Verify .env file
cat .env | grep OPENAI_API_KEY

# Test API key
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

#### **"Vector Store Not Found"**
```bash
# Check vector store directory
ls -la data/vector_store/

# Recreate vector store
rm -rf data/vector_store/default_faiss
# Re-upload documents through web interface
```

#### **"Out of Memory"**
```bash
# Monitor memory usage
docker stats

# Increase container memory
docker run -m 4g rag-system:latest

# Use smaller embedding model
export EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

#### **"Slow Response Times"**
```bash
# Enable GPU acceleration
export EMBEDDING_DEVICE=cuda

# Increase chunk size
export CHUNK_SIZE=1500

# Add Redis caching
docker run -d --name redis -p 6379:6379 redis:alpine
```

### **Performance Debugging**

```python
# Add performance profiling
from memory_profiler import profile
from line_profiler import LineProfiler

@profile
def debug_query_performance(question):
    # Your RAG query code here
    pass

# Run with profiling
python -m memory_profiler src/debug.py
```

### **Log Analysis**

```bash
# View application logs
tail -f logs/app.log

# Filter errors only
grep "ERROR" logs/app.log

# Monitor real-time performance
grep "total_time" logs/app.log | tail -20
```

---

## 📞 Support & Community

- 📖 **Documentation**: [GitHub Wiki](https://github.com/tomlangeveld/langchain-smart-doc-qa/wiki)
- 🐛 **Issues**: [GitHub Issues](https://github.com/tomlangeveld/langchain-smart-doc-qa/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/tomlangeveld/langchain-smart-doc-qa/discussions)
- 🔄 **Updates**: Watch the repository for releases

### **Contributing**

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Run tests: `pytest tests/`
4. Submit pull request

---

**🎉 Ready to deploy your enterprise RAG system!**

*Choose your deployment method above and start processing documents intelligently with LangChain.*