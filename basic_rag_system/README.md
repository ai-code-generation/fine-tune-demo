# Basic RAG System - NVIDIA GenerativeAI Examples

Hệ thống RAG (Retrieval-Augmented Generation) cơ bản được clone từ NVIDIA GenerativeAIExamples repository.

## 🚀 **NVIDIA NIM Local Deployment (Recommended)**

This system is configured for **NVIDIA NIM container deployment** with GPU acceleration!

### **NIM Architecture:**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │ Vector Database │
│ (React/Docker)  │◄──►│ (FastAPI/Docker)│◄──►│ (Milvus/Docker) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ NVIDIA NIM      │
                       │ • Llama3-8B     │
                       │ • E5-V5 Embed   │
                       │ • GPU Accel     │
                       └─────────────────┘
```

### **NIM Components:**
- **🤖 NIM LLM**: Llama3-8B-Instruct in container
- **🔗 NIM Embedding**: NV-EmbedQA-E5-V5 model
- **🏆 NIM Ranking**: Mistral-4B reranking (optional)
- **🗄️ Milvus**: Vector database with GPU support
- **🌐 RAG Playground**: Full-featured web interface

### **Quick NIM Start:**
```bash
# 1. Get NGC API Key from https://ngc.nvidia.com/
# 2. Setup environment for NIM deployment
copy .env.example .env
# Edit .env - set NGC_API_KEY and NVIDIA NIM configuration

# 3. Start NIM containers (requires GPU)
cd langchain
docker compose --profile local-nim up -d --build

# 4. Access your GPU-accelerated RAG system
# http://localhost:8090 - Web interface
# http://localhost:8000 - NIM LLM API
# http://localhost:9080 - NIM Embedding API
```

**📋 See [NIM_SETUP_GUIDE.md](NIM_SETUP_GUIDE.md) for detailed GPU setup**

## Cấu trúc thư mục

```
basic_rag_system/
├── langchain/                    # Frontend RAG implementation
│   ├── chains.py                # RAG chain logic
│   ├── docker-compose.yaml      # Container orchestration
│   ├── prompt.yaml             # LLM prompts
│   └── README.md               # Original documentation
├── chain_server/                # Backend API server
│   ├── server.py               # FastAPI application
│   ├── base.py                 # Base classes
│   ├── utils.py                # Utility functions
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile              # Container definition
│   └── ...                     # Other support files
├── rag_playground/             # Frontend Web UI
│   ├── default/                # Default UI mode
│   │   ├── api.py             # API client
│   │   ├── chat_client.py     # Chat interface
│   │   ├── pages/             # UI pages
│   │   └── static/            # Static assets
│   ├── speech/                # Speech UI mode (optional)
│   ├── Dockerfile             # Container definition
│   └── requirements.txt       # Python dependencies
├── local_deploy/              # Docker compose configs
│   ├── docker-compose-vectordb.yaml    # Milvus services
│   └── docker-compose-nim-ms.yaml      # NVIDIA NIM microservices
└── README.md                   # This file
```

## Các file chính

### 1. LangChain Implementation (langchain/)
- **chains.py**: Chứa class `NvidiaAPICatalog` với các methods:
  - `ingest_docs()`: Upload và index documents
  - `rag_chain()`: RAG query processing
  - `document_search()`: Document search functionality
  - `get_documents()` / `delete_documents()`: Document management

- **docker-compose.yaml**: Định nghĩa các services:
  - `chain-server`: Backend API
  - `rag-playground`: Frontend UI
  - `milvus-standalone`: Vector database
  - `milvus-etcd`: Cluster coordination
  - `milvus-minio`: Object storage

- **prompt.yaml**: Template prompts cho LLM

### 2. Chain Server (chain_server/)
- **server.py**: FastAPI server với endpoints:
  - POST `/uploadDocument`: Upload documents
  - POST `/generate`: Generate RAG responses
  - POST `/search`: Search documents
  - GET `/documents`: List documents
  - DELETE `/documents`: Delete documents

- **utils.py**: Utility functions cho:
  - Vector store operations
  - LLM configuration
  - Text splitting
  - Embedding operations

- **requirements.txt**: Python dependencies

### 3. RAG Playground (Frontend Web UI)
- **default/**: Default web interface mode
  - `api.py`: API client for backend communication
  - `chat_client.py`: Main chat interface implementation
  - `pages/`: Streamlit pages for different UI sections
  - `static/`: CSS, JS, and other static assets
- **speech/**: Speech-enabled interface mode (optional)
- **Dockerfile**: Container build instructions
- **requirements.txt**: Python dependencies for UI

### 4. Local Deploy Configuration
- **docker-compose-vectordb.yaml**: Milvus vector database services
  - milvus-standalone: Main vector database
  - milvus-etcd: Cluster coordination
  - milvus-minio: Object storage
- **docker-compose-nim-ms.yaml**: NVIDIA NIM microservices (optional)
  - nemollm-inference: Local LLM hosting
  - nemollm-embedding: Local embedding model hosting

## Quick Start

1. **Chuẩn bị environment**:
   ```bash
   # Set NVIDIA API key
   export NVIDIA_API_KEY="nvapi-your-key-here"
   
   # Navigate to langchain directory
   cd basic_rag_system/langchain/
   ```

2. **Start containers**:
   ```bash
   docker compose up -d --build
   ```

3. **Access the application**:
   - RAG Playground: http://localhost:8090
   - Chain Server API: http://localhost:8081
   - Milvus Dashboard: http://localhost:9091

## Dependencies

### Main Python Packages
- fastapi==0.110.0
- uvicorn[standard]==0.27.1
- langchain==0.1.9
- langchain-nvidia-ai-endpoints==0.1.6
- unstructured[all-docs]==0.12.5
- sentence-transformers==3.0.0
- pymilvus==2.4.0
- faiss-cpu==1.7.4

## System Requirements

- **CPU**: 4+ cores (recommended: 8+ cores)
- **RAM**: 8GB+ (recommended: 16GB+)
- **Disk**: 20GB+ free space
- **Docker**: Docker Desktop or Docker Engine
- **Network**: Internet connection for NVIDIA API calls

## Configuration

### Environment Variables
- `NVIDIA_API_KEY`: Required for NVIDIA API access
- `APP_VECTORSTORE_NAME`: Vector database type (default: milvus)
- `APP_LLM_MODELNAME`: LLM model name
- `APP_EMBEDDINGS_MODELNAME`: Embedding model name

### Model Configuration
- **LLM**: meta/llama3-70b-instruct
- **Embedding**: nvidia/nv-embedqa-e5-v5
- **Vector DB**: Milvus
- **Framework**: LangChain

## Usage

1. **Upload Documents**: Use the web interface to upload PDF, TXT, or MD files
2. **Wait for Processing**: Documents will be processed and indexed automatically
3. **Ask Questions**: Use the chat interface to ask questions about your documents
4. **Review Sources**: Check the source citations in responses

## Troubleshooting

### Common Issues
1. **API Key Issues**: Ensure NVIDIA_API_KEY is set correctly
2. **Container Issues**: Check Docker daemon and memory allocation
3. **Network Issues**: Verify internet connection for API calls
4. **Port Conflicts**: Ensure ports 8081, 8090, 19530 are available

### Logs
```bash
# View container logs
docker logs chain-server
docker logs milvus-standalone
docker logs rag-playground

# Follow logs in real-time
docker logs -f chain-server
```

## Next Steps

1. **Customize Prompts**: Edit `prompt.yaml` for custom prompts
2. **Add Authentication**: Implement user authentication
3. **Scale System**: Use Kubernetes for production deployment
4. **Monitor Performance**: Add observability tools
5. **Extend Features**: Add more file type support

## References

- [Original NVIDIA Repository](https://github.com/NVIDIA/GenerativeAIExamples)
- [LangChain Documentation](https://langchain.readthedocs.io/)
- [Milvus Documentation](https://milvus.io/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Note**: This is a development setup. For production deployment, please refer to the implementation and design documents for security and scalability considerations.

## 🏠 **Local Deployment (Recommended)**

This system is now configured for **complete local deployment** without cloud dependencies!

### **Local Architecture:**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │ Vector Database │
│ (React/Docker)  │◄──►│ (FastAPI/Docker)│◄──►│ (Milvus/Docker) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Local AI Models │
                       │ • Ollama (LLM)  │
                       │ • SentenceTrans │
                       │   (Embeddings)  │
                       └─────────────────┘
```

### **Local Components:**
- **🤖 Ollama Server**: Local LLM hosting (Llama 3.2, Phi-3, Mistral)
- **🔗 Embedding Service**: Local sentence-transformers
- **🗄️ Milvus**: Local vector database
- **🌐 RAG Playground**: Web interface
- **📡 Chain Server**: FastAPI backend

### **Quick Local Start:**
```bash
# 1. Setup environment for local deployment
copy .env.example .env
# Edit .env - use LOCAL MODEL CONFIGURATION section

# 2. Start all services
cd langchain
docker compose up -d --build

# 3. Setup local models
cd ../local_deploy/ollama_config
setup_models.bat

# 4. Access your local RAG system
# http://localhost:8090 - Web interface
# http://localhost:11434 - Ollama API
```

**📋 See [LOCAL_SETUP_GUIDE.md](LOCAL_SETUP_GUIDE.md) for detailed instructions**
