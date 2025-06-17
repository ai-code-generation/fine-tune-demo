# NVIDIA NIM & Blueprints Complete Guide

## Table of Contents

1. [Introduction to NVIDIA NIM](#1-introduction-to-nvidia-nim)
2. [NIM Architecture & Components](#2-nim-architecture--components)
3. [Running a Basic NIM: Step-by-Step](#3-running-a-basic-nim-step-by-step)
4. [Using Custom Fine-tuned Models with NIM](#4-using-custom-fine-tuned-models-with-nim)
5. [NVIDIA Blueprints Overview](#5-nvidia-blueprints-overview)
6. [Deploying a RAG Blueprint](#6-deploying-a-rag-blueprint)
7. [Customizing Blueprints](#7-customizing-blueprints)

---

## 1. Introduction to NVIDIA NIM

### What is NVIDIA NIM?

NVIDIA NIM (NVIDIA Inference Microservices) is a set of containerized microservices designed to accelerate the deployment of AI models in production environments. NIM provides optimized inference engines that can run on various NVIDIA hardware platforms.

### Key Features

✅ **High Performance:** Optimized for NVIDIA GPUs with TensorRT acceleration  
✅ **Production Ready:** Enterprise-grade scalability and reliability  
✅ **Model Agnostic:** Supports various model formats (PyTorch, ONNX, TensorRT)  
✅ **Easy Deployment:** Containerized with Docker/Kubernetes support  
✅ **API Standardization:** REST and gRPC APIs following OpenAI standards  
✅ **Multi-Modal Support:** Text, vision, speech, and multimodal models

### Advantages

| Feature | Benefit |
|---------|---------|
| Optimized Performance | 2-10x faster inference vs vanilla deployments |
| Simplified Deployment | One-command container deployment |
| Scalability | Auto-scaling with Kubernetes |
| Cost Efficiency | Better GPU utilization and throughput |
| Enterprise Support | NVIDIA backing with SLA guarantees |
| Security | Built-in security and compliance features |

---

## 2. NIM Architecture & Components

### Layer Architecture

```
🏗️ NIM Architecture (Top to Bottom):
├── 🌐 API Layer          → REST/gRPC endpoints
├── 🔧 Service Layer      → Request processing & routing  
├── 🤖 Inference Engine   → Model execution & optimization
├── 📦 Model Layer        → Model weights & configurations
├── ⚡ Runtime Layer      → TensorRT, CUDA, drivers
└── 🖥️ Hardware Layer    → GPU/CPU infrastructure
```

---

## 3. Running a Basic NIM: Step-by-Step

### Prerequisites

- Docker installed
- NVIDIA GPU drivers (for GPU acceleration)
- NGC account and API key

### Step 1: NGC Account Setup

```bash
# 1. Create NGC account at https://catalog.ngc.nvidia.com/
# 2. Generate API key from Setup → Generate API Key
# 3. Login to NGC registry
docker login nvcr.io
# Username: $oauthtoken
# Password: nvapi-xxxxxxxxxxxxx (your API key)
```

### Step 2: Pull NIM Container

```bash
# Pull a basic text generation NIM
docker pull nvcr.io/nvidia/nim/llama2-7b-chat:latest

# Verify download
docker images | grep nim
```

### Step 3: Run NIM Container

```bash
docker run -d \
  --name nim-llama2-gpu \
  --gpus all \
  -p 8000:8000 \
  -e NIM_DEVICE=cuda \
  -e NIM_MODEL_NAME=llama2-7b-chat \
  nvcr.io/nvidia/nim/llama2-7b-chat:latest
```

### Step 4: Test NIM API

```bash
# Health check
curl http://localhost:8000/health

# Text completion
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2-7b-chat",
    "prompt": "The future of AI is",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2-7b-chat",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
  }'
```

---

## 4. Using Custom Fine-tuned Models with NIM

### Step 1: Prepare Your Fine-tuned Model

```bash
# Ensure your model has the correct structure
my-ai-project/
├── 📁 my-finetuned-model/         
│   ├── pytorch_model.bin           
│   ├── config.json                
│   ├── tokenizer.json              
│   └── tokenizer_config.json 
│
├── docker-compose.yml             
└── .env                            
```

### Step 2: Create Model Directory Structure

```bash
# Create directory for your custom model
mkdir -p ./custom-models/my-finetuned-llm

# Copy your fine-tuned model files
cp -r /path/to/your/finetuned-model/* ./custom-models/my-finetuned-llm/
```

### Step 3: Create Custom NIM Configuration

#### docker-compose.yml
```yaml
version: '3.8'
services:
  nim-service:
    image: nvcr.io/nvidia/nim/nim-llm:latest  # NIM template có sẵn
    ports:
      - "8000:8000"     # API endpoint
      - "8001:8001"     # gRPC endpoint  
    volumes:
      - ./my-finetuned-model:/opt/nim/models/my-model
    environment:
      - NIM_MODEL_NAME=my-model
      - NIM_CACHE_PATH=/tmp/nim-cache
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

#### .env
```
# Model configuration  
MODEL_NAME=my-finetuned-model
MAX_BATCH_SIZE=8
MAX_SEQUENCE_LENGTH=2048

# API configuration
API_PORT=8000
GRPC_PORT=8001

# Performance tuning
ENABLE_TENSORRT=true
OPTIMIZE_MODEL=true

```

### Step 4: Run NIM with Custom Model

```bash
# Deploy with docker-compose
docker-compose up -d
```

### Step 5: Test Custom Model

```bash
# Test your custom model
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-custom-model",
    "prompt": "Test my fine-tuned model:",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Check model info
curl http://localhost:8000/v1/models
```

## 5. NVIDIA Blueprints Overview

### What are NVIDIA Blueprints?

NVIDIA Blueprints are pre-built, production-ready AI solution templates that combine multiple NIMs and infrastructure components to solve specific business problems. They provide end-to-end workflows that can be quickly deployed and customized.

### Blueprint Components

```
🏗️ NVIDIA Blueprint Architecture:
├── 🤖 NIM Services        → Pre-configured AI microservices
├── 🔗 Orchestration      → Workflow coordination logic
├── 🌐 API Gateway        → Unified API endpoints
├── 💾 Data Pipeline      → Data ingestion and preprocessing
├── 📊 Monitoring         → Metrics and observability
├── 🐳 Infrastructure     → Docker/K8s deployment configs
└── 📚 Documentation      → Setup and customization guides
```

### Blueprint Categories

| Category | Examples | Use Cases |
|----------|----------|-----------|
| Conversational AI | Chatbots, Virtual Assistants | Customer service, FAQ automation |
| Computer Vision | Object Detection, OCR | Quality inspection, content moderation |
| RAG Systems | Document Q&A, Knowledge Base | Enterprise search, support systems |
| Multimodal | Vision + Language | Content analysis, accessibility |
| Healthcare | Medical Imaging, Drug Discovery | Diagnosis assistance, research |
| Financial | Fraud Detection, Risk Analysis | Compliance, transaction monitoring |

### RAG Blueprint Architecture Diagram

```
📊 RAG Blueprint Workflow:

┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   📄 Documents   │───▶│  🔧 Processing   │───▶│ 🗄️ Vector Store │
│                 │    │   • Chunking     │    │   • Embeddings  │
│   • PDFs        │    │   • Cleaning     │    │   • Indexing    │
│   • Text files  │    │   • Embedding    │    │   • Search      │
│   • Web pages   │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│ 💬 User Query   │───▶│  🔍 Retrieval    │◀────────────┘
│                 │    │   • Similarity   │
│ "What is X?"    │    │   • Top-K docs   │
│                 │    │   • Context      │
└─────────────────┘    └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ 🤖 Generation    │───▶│ ✅ Response     │
                       │   • LLM Model    │    │                 │
                       │   • Prompt       │    │ "Based on the   │
                       │   • Context      │    │  documents..."  │
                       └──────────────────┘    └─────────────────┘
```

### How Blueprints Work

1. **Template Deployment:** Download pre-configured blueprint with all components
2. **Configuration:** Customize settings for your specific use case
3. **Model Integration:** Plug in your custom models or use pre-trained ones
4. **Data Integration:** Connect your data sources and vector databases
5. **Deployment:** One-command deployment to your infrastructure
6. **Monitoring:** Built-in observability and performance tracking

---

## 6. Deploying a RAG Blueprint

### Step 1: Download RAG Blueprint

```bash
# Method 1: NGC CLI
ngc registry resource download-version \
  nvidia/blueprints/rag-chatbot:latest

# Method 2: Direct docker pull
docker pull nvcr.io/nvidia/blueprints/rag-chatbot:latest

# https://github.com/NVIDIA-AI-Blueprints/rag/tree/v2.1.0/docs
```

### Step 2: Setup Directory Structure

```bash
# Create project directory
mkdir my-rag-deployment
cd my-rag-deployment

# Create required directories
mkdir -p {models,documents,vector-db,configs,logs}
```

### Step 3: Configure Blueprint

```yaml
# Create config file: configs/rag-config.yaml
rag_blueprint:
  # Model configurations
  models:
    llm:
      provider: "nim"
      model_name: "llama2-7b-chat"
      endpoint: "http://llm-service:8000"
      
    embedding:
      provider: "nim" 
      model_name: "e5-large-v2"
      endpoint: "http://embedding-service:8000"
      
  # Vector database configuration
  vector_store:
    provider: "chroma"  # chroma, pinecone, weaviate, qdrant
    connection:
      host: "localhost"
      port: 8000
      collection_name: "rag_documents"
      
  # Document processing
  document_processing:
    chunk_size: 1000
    chunk_overlap: 200
    supported_formats: ["pdf", "txt", "docx", "md"]
    
  # Retrieval settings
  retrieval:
    top_k: 5
    similarity_threshold: 0.7
    search_type: "similarity"
    
  # API configuration
  api:
    host: "0.0.0.0"
    port: 8080
    enable_ui: true
    cors_origins: ["*"]
```

### Step 4: Deploy with Docker Compose

```yaml
# Create docker-compose.yml
version: '3.8'

services:
  # LLM Service (NIM)
  llm-service:
    image: nvcr.io/nvidia/nim/llama2-7b-chat:latest
    container_name: rag-llm-service
    environment:
      - NIM_DEVICE=cpu
    ports:
      - "8001:8000"
    networks:
      - rag-network

  # Embedding Service (NIM)
  embedding-service:
    image: nvcr.io/nvidia/nim/embed-qa:latest
    container_name: rag-embedding-service
    environment:
      - NIM_DEVICE=cpu
    ports:
      - "8002:8000"
    networks:
      - rag-network

  # Vector Database (ChromaDB)
  vector-db:
    image: chromadb/chroma:latest
    container_name: rag-vector-db
    ports:
      - "8003:8000"
    volumes:
      - ./vector-db:/chroma/chroma
    networks:
      - rag-network

  # RAG Orchestrator
  rag-service:
    image: nvcr.io/nvidia/blueprints/rag-chatbot:latest
    container_name: rag-orchestrator
    ports:
      - "8080:8080"  # API
      - "8501:8501"  # Streamlit UI
    volumes:
      - ./configs/rag-config.yaml:/config/config.yaml:ro
      - ./documents:/data/documents:ro
      - ./logs:/app/logs
    environment:
      - RAG_CONFIG_PATH=/config/config.yaml
      - LLM_SERVICE_URL=http://llm-service:8000
      - EMBEDDING_SERVICE_URL=http://embedding-service:8000
      - VECTOR_DB_URL=http://vector-db:8000
    depends_on:
      - llm-service
      - embedding-service
      - vector-db
    networks:
      - rag-network

networks:
  rag-network:
    driver: bridge
```

### Step 5: Add Documents and Deploy

```bash
# Add your documents
cp /path/to/your/documents/* ./documents/

# Deploy the blueprint
docker-compose up -d

# Check deployment status
docker-compose ps
```

### Step 6: Test RAG System

```bash
# Wait for services to start (may take a few minutes)
sleep 60

# Test health endpoints
curl http://localhost:8001/health  # LLM service
curl http://localhost:8002/health  # Embedding service
curl http://localhost:8003/api/v1/heartbeat  # Vector DB
curl http://localhost:8080/health  # RAG service

# Test document indexing
curl -X POST http://localhost:8080/v1/index \
  -H "Content-Type: application/json" \
  -d '{"documents_path": "/data/documents"}'

# Test RAG query
curl -X POST http://localhost:8080/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic in the documents?",
    "max_tokens": 200
  }'

# Access web UI
open http://localhost:8501
```

### Step 7: Monitor and Scale

```bash
# Monitor logs
docker-compose logs -f rag-service

# Scale services if needed
docker-compose up -d --scale llm-service=2

# Update configuration
vim configs/rag-config.yaml
docker-compose restart rag-service
```

---

## 7. Customizing Blueprints

### Customization Options

🎯 **Model Customization**
- Replace default models with your fine-tuned versions
- Adjust model parameters and inference settings
- Add new model types (vision, speech, etc.)

🔧 **Pipeline Modification**
- Custom document processing logic
- Enhanced retrieval algorithms
- Multi-step reasoning workflows

🌐 **API Extensions**
- Additional endpoints for specific use cases
- Custom authentication and authorization
- Integration with external systems

💾 **Data Integration**
- Custom data connectors
- Real-time data streaming
- Multi-source data fusion

#### https://github.com/NVIDIA/GenerativeAIExamples/tree/main/RAG/examples


**End of Document**
