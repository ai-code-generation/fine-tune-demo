# Quick Setup Script for RAG System

## Prerequisites Check
```powershell
# Check Docker
docker --version
docker-compose --version

# Check Git
git --version

# Check available ports
netstat -an | findstr "8081 8090 19530"
```

## Step 1: Setup Environment
```powershell
# Copy environment file
copy .env.example .env

# Edit .env file and set your NVIDIA API key
# NVIDIA_API_KEY=nvapi-your-actual-key-here
```

## Step 2: Get NVIDIA API Key
1. Go to https://build.nvidia.com/
2. Sign up/Login with NVIDIA account
3. Navigate to API Catalog
4. Generate API key
5. Copy the key to your .env file

## Step 3: Start the System
```powershell
# Navigate to langchain directory
cd langchain

# Start all containers
docker compose up -d --build

# Check container status
docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"
```

## Step 4: Verify Services
- RAG Playground: http://localhost:8090
- Chain Server API: http://localhost:8081/docs
- Milvus Dashboard: http://localhost:9091

## Step 5: Test Upload
1. Open http://localhost:8090
2. Click "Knowledge Base" tab
3. Upload a sample PDF/TXT file
4. Wait for processing to complete
5. Go to "Chat" tab
6. Ask questions about the document

## Step 6: Verify Complete System
Check that all components are available:

```powershell
# Check rag_playground structure
ls basic_rag_system\rag_playground\
ls basic_rag_system\rag_playground\default\

# Check local_deploy configs
ls basic_rag_system\local_deploy\

# Verify docker-compose references
cat basic_rag_system\langchain\docker-compose.yaml | findstr "local_deploy"
```

## Step 7: Component Overview
Your complete system now includes:

- **🎯 LangChain Implementation**: RAG logic và chains
- **🔌 Chain Server**: FastAPI backend với all endpoints  
- **🌐 RAG Playground**: Full web UI với chat interface
- **🗄️ Vector Database**: Milvus với etcd và minio
- **🐳 Container Setup**: Complete docker orchestration

## Troubleshooting

### Container Issues
```powershell
# View logs
docker logs chain-server
docker logs milvus-standalone
docker logs rag-playground

# Restart specific container
docker restart chain-server

# Stop all containers
docker compose down

# Remove all containers and volumes
docker compose down -v
```

### API Key Issues
```powershell
# Check if API key is set
docker exec chain-server env | findstr NVIDIA_API_KEY

# Test API key
curl -H "Authorization: Bearer nvapi-your-key" https://integrate.api.nvidia.com/v1/models
```

### Port Conflicts
```powershell
# Find processes using ports
netstat -ano | findstr ":8081"
netstat -ano | findstr ":8090"
netstat -ano | findstr ":19530"

# Kill process by PID (if needed)
taskkill /PID <process_id> /F
```

### Memory Issues
```powershell
# Check Docker memory usage
docker stats

# Increase Docker Desktop memory allocation:
# Docker Desktop > Settings > Resources > Advanced > Memory
```

## Development Commands

### View Container Logs
```powershell
# Real-time logs
docker logs -f chain-server
docker logs -f milvus-standalone

# Last 100 lines
docker logs --tail 100 chain-server
```

### Database Operations
```powershell
# Connect to Milvus
docker exec -it milvus-standalone milvus-cli

# List collections
docker exec milvus-standalone milvus-cli list collections

# Check collection info
docker exec milvus-standalone milvus-cli describe collection -c vector_db
```

### API Testing
```powershell
# Test health endpoint
curl http://localhost:8081/health

# Test document upload
curl -X POST "http://localhost:8081/uploadDocument" ^
     -H "Content-Type: multipart/form-data" ^
     -F "file=@sample.pdf" ^
     -F "filename=sample.pdf"

# Test search
curl -X POST "http://localhost:8081/search" ^
     -H "Content-Type: application/json" ^
     -d "{\"query\": \"What is this document about?\", \"num_docs\": 5}"
```

## Cleanup
```powershell
# Stop and remove containers
docker compose down -v

# Remove images (optional)
docker rmi $(docker images -q)

# Clean up temp files
rmdir /s temp_repo
```

## Next Steps
1. ✅ System running successfully
2. 📚 Upload your own documents
3. 🔧 Customize prompts in `prompt.yaml`
4. 🚀 Deploy to production environment
5. 📊 Add monitoring and observability
