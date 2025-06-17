# NVIDIA NIM Local Deployment Setup Guide

## 🎯 **NVIDIA NIM Container Architecture**

Hệ thống sử dụng **NVIDIA NIM containers** để deploy local models với GPU acceleration.

```
┌─────────────────────────────────────────────────────────────────┐
│                    NVIDIA NIM RAG SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ RAG Playground  │  │  Chain Server   │  │     Milvus      │ │
│  │   (Frontend)    │◄─┤   (Backend)     │◄─┤ (Vector Store)  │ │
│  │  Port: 8090     │  │  Port: 8081     │  │  Port: 19530    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                 │                              │
│                                 ▼                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ NIM LLM         │  │ NIM Embedding   │  │ NIM Ranking     │ │
│  │ Llama3-8B       │  │ E5-V5 Model     │  │ Mistral-4B      │ │
│  │ Port: 8000      │  │ Port: 9080      │  │ Port: 1976      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                 │
│  🚀 All models run in NVIDIA containers with GPU acceleration  │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 **Prerequisites**

### **Hardware Requirements:**
- **GPU**: NVIDIA GPU với CUDA support (RTX 3060 8GB+)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space cho models
- **CPU**: 8+ cores recommended

### **Software Requirements:**
- Docker Desktop với GPU support
- NVIDIA Container Toolkit
- NGC Account (free registration)

## 🔧 **Step 1: Setup NVIDIA Container Toolkit**

### **Windows (Docker Desktop):**
```powershell
# 1. Install NVIDIA Container Toolkit
# Download from: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# 2. Verify GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi

# 3. Enable GPU support in Docker Desktop
# Settings > Resources > WSL Integration > Enable GPU support
```

### **Linux:**
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi
```

## 🔑 **Step 2: Get NGC API Key**

1. **Register at NGC**: https://ngc.nvidia.com/
2. **Generate API Key**:
   - Login → Account → Setup → Generate API Key
   - Copy your API key
3. **Login to NGC Registry**:
   ```bash
   docker login nvcr.io
   Username: $oauthtoken
   Password: <your-ngc-api-key>
   ```

## 🏗️ **Step 3: Setup Environment**

```powershell
# Navigate to project
cd basic_rag_system

# Copy and configure environment
copy .env.example .env

# Edit .env file with your settings:
# NGC_API_KEY=your-ngc-api-key-here
# MODEL_DIRECTORY=./volumes/nim-cache
# USERID=1000
```

### **Key Environment Variables:**
```bash
# Required
NGC_API_KEY=your-ngc-api-key-here
MODEL_DIRECTORY=./volumes/nim-cache

# GPU Configuration
INFERENCE_GPU_COUNT=1
EMBEDDING_MS_GPU_ID=0
LLM_MS_GPU_ID=0

# Model Endpoints
APP_LLM_SERVERURL=http://nemollm-inference-microservice:8000
APP_EMBEDDINGS_SERVERURL=http://nemo-retriever-embedding-microservice:8000
```

## 🚀 **Step 4: Deploy NIM Containers**

```powershell
# Navigate to langchain directory
cd langchain

# Start with specific profiles
docker compose --profile local-nim up -d --build

# OR start all services including ranking
docker compose --profile nemo-retriever up -d --build

# Check container status
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### **Available Profiles:**
- `local-nim`: LLM + Embedding services
- `nemo-retriever`: Full stack including ranking
- `milvus`: Vector database only
- `pgvector`: PostgreSQL vector database

## 📊 **Step 5: Monitor Deployment**

### **Check Container Health:**
```powershell
# Check NIM LLM service
curl http://localhost:8000/v1/health/ready

# Check NIM Embedding service
curl http://localhost:9080/v1/health/ready

# Check Chain Server
curl http://localhost:8081/health

# Check Milvus
curl http://localhost:9091/healthz
```

### **Monitor Logs:**
```powershell
# NIM containers download models on first start
docker logs -f nemollm-inference-microservice
docker logs -f nemo-retriever-embedding-microservice

# Chain server logs
docker logs -f chain-server
```

## 🎯 **Step 6: Test System**

1. **Access Web Interface**: http://localhost:8090
2. **Upload Documents**: Test with sample PDF/TXT files
3. **Ask Questions**: Verify RAG functionality
4. **Check Performance**: Monitor GPU usage

### **API Testing:**
```powershell
# Test LLM directly
curl -X POST "http://localhost:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "meta/llama3-8b-instruct", "prompt": "Hello!", "max_tokens": 100}'

# Test Embedding
curl -X POST "http://localhost:9080/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"model": "nvidia/nv-embedqa-e5-v5", "input": ["Hello world"]}'
```

## ⚡ **Performance Optimization**

### **GPU Memory Optimization:**
```bash
# Adjust GPU memory allocation in .env
INFERENCE_GPU_COUNT=1  # Use single GPU
# For multiple GPUs:
INFERENCE_GPU_COUNT=2
LLM_MS_GPU_ID=0
EMBEDDING_MS_GPU_ID=1
```

### **Model Caching:**
```bash
# Persistent model cache
MODEL_DIRECTORY=/path/to/persistent/storage/nim-cache
# Models are downloaded once and cached
```

### **Resource Limits:**
```yaml
# In docker-compose, adjust resources:
deploy:
  resources:
    limits:
      memory: 20G
    reservations:
      memory: 16G
```

## 🔍 **Troubleshooting**

### **Common Issues:**

**GPU Not Available:**
```bash
# Check NVIDIA drivers
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi
```

**NGC Authentication Failed:**
```bash
# Re-login to NGC
docker logout nvcr.io
docker login nvcr.io
```

**Model Download Timeout:**
```bash
# Increase health check timeout
HEALTHCHECK --start-period=10m
```

**Out of Memory:**
```bash
# Use smaller models or reduce batch size
# Monitor with: docker stats
```

## 📈 **Available Models**

### **NIM LLM Models:**
- `nvcr.io/nim/meta/llama3-8b-instruct:1.0.3` (8GB VRAM)
- `nvcr.io/nim/meta/llama3-70b-instruct:1.0.0` (40GB VRAM)
- `nvcr.io/nim/microsoft/phi-3-mini-128k-instruct:1.0.0` (4GB VRAM)

### **NIM Embedding Models:**
- `nvcr.io/nim/nvidia/nv-embedqa-e5-v5:1.0.1`
- `nvcr.io/nim/nvidia/nv-embed-v1:1.0.0`

### **NIM Ranking Models:**
- `nvcr.io/nim/nvidia/nv-rerankqa-mistral-4b-v3:1.0.1`

## 🎯 **Production Considerations**

### **Security:**
- Secure NGC API keys
- Network isolation
- Resource quotas
- Access controls

### **Scaling:**
- Multi-GPU deployment
- Load balancing
- Model replicas
- Horizontal scaling

### **Monitoring:**
- GPU utilization
- Model latency
- Memory usage
- Error rates

---

**🎉 Your NVIDIA NIM RAG system is ready!**
**🌐 Access: http://localhost:8090**
**📡 API: http://localhost:8081/docs**
