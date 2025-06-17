# Local RAG System Setup Script for Windows

## Prerequisites
```powershell
# Check Docker
docker --version
docker-compose --version

# Check system resources
systeminfo | findstr "Total Physical Memory"
wmic cpu get name
```

## Step 1: Setup Environment for Local Deployment
```powershell
# Copy environment file
copy .env.example .env

# Edit .env file - ensure LOCAL MODEL CONFIGURATION is uncommented
# Comment out NVIDIA_API_KEY and cloud configurations
```

## Step 2: Start Local AI Infrastructure
```powershell
# Navigate to langchain directory
cd basic_rag_system\langchain

# Start all services (this will take time on first run)
docker compose up -d --build

# Check service status
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

## Step 3: Setup Local Models
```powershell
# Setup Ollama models (run after containers are up)
cd ..\local_deploy\ollama_config
setup_models.bat

# Alternative: Manual model setup
docker exec ollama-server ollama pull llama3.2:3b
docker exec ollama-server ollama pull phi3:mini
```

## Step 4: Verify Local Services
Check these endpoints:

### Core Services
- 🌐 **RAG Playground**: http://localhost:8090
- 📡 **Chain Server API**: http://localhost:8081/docs
- 🗄️ **Milvus Dashboard**: http://localhost:9091

### Local AI Services
- 🤖 **Ollama API**: http://localhost:11434/api/tags
- 🔗 **Embedding Service**: http://localhost:8002/health
- 🎮 **Text Generation WebUI**: http://localhost:7860 (optional)

## Step 5: Test Local Setup
```powershell
# Test Ollama
curl "http://localhost:11434/api/tags"

# Test Embedding Service  
curl "http://localhost:8002/health"

# Test Milvus
curl "http://localhost:9091/healthz"

# Test Chain Server
curl "http://localhost:8081/health"
```

## Step 6: Upload and Test Documents
1. Open http://localhost:8090
2. Go to "Knowledge Base" tab
3. Upload a test document (PDF/TXT)
4. Wait for local processing (will be slower than cloud)
5. Chat with your documents!

## Performance Tips

### System Requirements
- **Minimum**: 8GB RAM, 4 CPU cores, 20GB disk space
- **Recommended**: 16GB+ RAM, 8+ CPU cores, SSD storage
- **With GPU**: NVIDIA GPU with 8GB+ VRAM for better performance

### Model Selection
```powershell
# Lightweight models (good for 8GB RAM systems)
docker exec ollama-server ollama pull phi3:mini          # ~2GB
docker exec ollama-server ollama pull llama3.2:3b       # ~2GB

# Medium models (good for 16GB+ RAM systems)  
docker exec ollama-server ollama pull mistral:7b        # ~4GB
docker exec ollama-server ollama pull llama3.1:8b       # ~5GB

# Large models (requires 32GB+ RAM)
docker exec ollama-server ollama pull llama3.1:70b      # ~40GB
```

### Performance Optimization
```powershell
# Update .env for better performance
APP_TEXTSPLITTER_CHUNKSIZE=256  # Smaller chunks = faster processing
APP_RETRIEVER_TOPK=3            # Fewer results = faster retrieval
TEMPERATURE=0.3                 # Lower temperature = faster inference
MAX_TOKENS=1024                 # Fewer tokens = faster generation
```

## Troubleshooting

### Container Issues
```powershell
# Check logs
docker logs ollama-server
docker logs embedding-server
docker logs chain-server
docker logs milvus-standalone

# Restart services
docker compose restart

# Full reset
docker compose down -v
docker compose up -d --build
```

### Model Issues
```powershell
# Check available models
docker exec ollama-server ollama list

# Re-pull models
docker exec ollama-server ollama pull llama3.2:3b

# Test model directly
docker exec -it ollama-server ollama run llama3.2:3b "Hello!"
```

### Memory Issues
```powershell
# Check memory usage
docker stats

# Free up memory
docker system prune -f

# Adjust Docker memory limits in Docker Desktop:
# Settings > Resources > Advanced > Memory
```

### Performance Issues
```powershell
# Use smaller models
APP_LLM_MODELNAME=phi3:mini

# Reduce context length
CONTEXT_LENGTH=2048

# Use CPU-only mode
DEVICE=cpu
```

## Production Considerations

### Security
- Change default passwords in .env
- Use HTTPS in production
- Implement authentication
- Secure API endpoints

### Scaling
- Use Docker Swarm or Kubernetes
- Add load balancers
- Implement model caching
- Use GPU acceleration

### Monitoring
- Add Prometheus metrics
- Setup log aggregation
- Monitor resource usage
- Track model performance

## Next Steps
1. ✅ System running locally
2. 📚 Test with your documents
3. 🔧 Tune model parameters
4. 🚀 Scale for production use
5. 📊 Add monitoring and observability

---

**💡 Tip**: Start with lightweight models and scale up based on your performance needs!
