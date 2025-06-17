# RAG System Design Document

## 1. Tổng quan kiến trúc hệ thống

### 1.1 High-Level Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Browser   │    │  RAG Playground │    │   Chain Server  │
│                 │◄──►│   (Frontend)    │◄──►│   (Backend)     │
│   User Interface│    │   Port: 8090    │    │   Port: 8081    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────┐             │
                       │ NVIDIA API      │◄────────────┤
                       │ Catalog         │             │
                       │ (LLM + Embed)   │             │
                       └─────────────────┘             │
                                                        │
                       ┌─────────────────┐             │
                       │ Milvus Vector   │◄────────────┘
                       │ Database        │
                       │ Port: 19530     │
                       └─────────────────┘
```

### 1.2 Component Overview
- **RAG Playground**: React-based web interface
- **Chain Server**: FastAPI backend xử lý RAG logic
- **Milvus**: Vector database lưu trữ embeddings
- **NVIDIA API**: Cloud-based LLM và embedding services

## 2. Detailed System Design

### 2.1 Data Flow Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        Document Ingestion Flow                  │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Upload    │───►│   Parse     │───►│   Chunk     │───►│  Vectorize  │
│  Document   │    │  Document   │    │   Text      │    │ & Store     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                           │                   │                   │
                           ▼                   ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
                   │UnstructuredFL│    │Text Splitter│    │   Milvus    │
                   │   Loader    │    │(LangChain)  │    │  Database   │
                   └─────────────┘    └─────────────┘    └─────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         Query Flow                              │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│User Question│───►│  Vectorize  │───►│  Retrieve   │───►│  Generate   │
│             │    │   Query     │    │  Context    │    │  Response   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                           │                   │                   │
                           ▼                   ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
                   │NVIDIA Embed │    │   Milvus    │    │NVIDIA LLM   │
                   │   API       │    │ Similarity  │    │   API       │
                   └─────────────┘    └─────────────┘    └─────────────┘
```

### 2.2 Container Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                       Docker Network: nvidia-rag               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ rag-playground  │  │  chain-server   │  │milvus-standalone│ │
│  │                 │  │                 │  │                 │ │
│  │ • React UI      │  │ • FastAPI       │  │ • Vector DB     │ │
│  │ • Port: 8090    │  │ • Port: 8081    │  │ • Port: 19530   │ │
│  │ • Nginx         │  │ • Python 3.10   │  │ • etcd cluster  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│           │                     │                     │        │
│           └─────────────────────┼─────────────────────┘        │
│                                 │                              │
│  ┌─────────────────┐            │            ┌─────────────────┐ │
│  │  milvus-etcd    │            │            │  milvus-minio   │ │
│  │                 │            │            │                 │ │
│  │ • Coordination  │            │            │ • Object Store  │ │
│  │ • Port: 2379    │            │            │ • Port: 9000    │ │
│  └─────────────────┘            │            └─────────────────┘ │
│                                 │                              │
│                    ┌─────────────────┐                         │
│                    │ External APIs   │                         │
│                    │                 │                         │
│                    │ • NVIDIA API    │                         │
│                    │   Catalog       │                         │
│                    └─────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

## 3. Component Design Details

### 3.1 Chain Server (Backend)
```
chain-server/
├── server.py                 # FastAPI application entry point
├── chains.py                 # NvidiaAPICatalog implementation
├── base.py                   # BaseExample abstract class
├── utils.py                  # Utility functions
├── tracing.py               # Observability & logging
└── requirements.txt         # Python dependencies

FastAPI Endpoints:
├── POST /uploadDocument     # Document ingestion
├── POST /generate          # RAG chat completion
├── POST /search            # Document search
├── GET /documents          # List uploaded documents
└── DELETE /documents       # Delete documents
```

### 3.2 RAG Playground (Frontend)
```
rag-playground/
├── src/
│   ├── components/
│   │   ├── ChatInterface.tsx    # Main chat UI
│   │   ├── DocumentUpload.tsx   # File upload component
│   │   ├── MessageList.tsx      # Chat messages display
│   │   └── SettingsPanel.tsx    # Configuration panel
│   ├── services/
│   │   ├── api.ts              # Backend API calls
│   │   └── websocket.ts        # Real-time communication
│   └── App.tsx                 # Main application
└── nginx.conf                  # Web server configuration
```

### 3.3 Milvus Vector Database
```
Milvus Architecture:
├── milvus-standalone        # Main vector database
├── milvus-etcd             # Cluster coordination
└── milvus-minio            # Object storage

Collections Schema:
├── documents_collection
│   ├── id: int64           # Auto-generated ID
│   ├── vector: float[]     # Document embeddings (dim: 1024)
│   ├── content: string     # Original text content
│   ├── source: string      # Document filename
│   └── metadata: JSON      # Additional information
```

## 4. User Interface Views

### 4.1 Main Dashboard View
```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Playground Dashboard                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐              ┌─────────────────────────────┐│
│  │   Knowledge     │              │         Chat Interface       ││
│  │     Base        │              │                             ││
│  │                 │              │  ┌─────────────────────────┐││
│  │ [Upload Doc]    │              │  │ User: How to install... │││
│  │                 │              │  └─────────────────────────┘││
│  │ Documents:      │              │  ┌─────────────────────────┐││
│  │ • manual.pdf ❌  │              │  │ Bot: To install the...  │││
│  │ • guide.txt  ❌  │              │  └─────────────────────────┘││
│  │ • readme.md  ❌  │              │                             ││
│  │                 │              │  ┌─────────────────────────┐││
│  │ [Clear All]     │              │  │ [Type your question...] │││
│  │                 │              │  │                    [▶]  │││
│  └─────────────────┘              │  └─────────────────────────┘││
│                                   └─────────────────────────────┘│
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                      Settings Panel                         ││
│  │                                                             ││
│  │  Model: [meta/llama3-70b-instruct ▼]  Temperature: [0.7]   ││
│  │  Max Tokens: [2048]  Top-K: [5]  Score Threshold: [0.25]   ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Document Upload View
```
┌─────────────────────────────────────────────────────────────────┐
│                       Upload Documents                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Drag & Drop Area                        ││
│  │                                                             ││
│  │              📄 Drop files here or click to browse         ││
│  │                                                             ││
│  │              Supported: PDF, TXT, MD, DOCX                 ││
│  │              Max size: 50MB per file                       ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  Upload Queue:                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 📄 installation_guide.pdf        [████████░░] 80% ⏸️ ❌    ││
│  │ 📄 user_manual.docx              [██████████] ✅ Done      ││
│  │ 📄 troubleshooting.md            [░░░░░░░░░░] Pending      ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  [Start Upload] │  │  [Pause All]    │  │  [Clear Queue]  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Chat Interface View (Detailed)
```
┌─────────────────────────────────────────────────────────────────┐
│                        Chat Interface                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Message History                          ││
│  │                                                             ││
│  │  👤 User (2:30 PM)                                          ││
│  │  How do I configure the RAG system for production?          ││
│  │                                                             ││
│  │  🤖 Assistant (2:31 PM)                                     ││
│  │  Based on the documentation you uploaded, here are the      ││
│  │  key steps for production configuration:                    ││
│  │                                                             ││
│  │  1. Environment Variables:                                  ││
│  │     - Set NVIDIA_API_KEY for production                     ││
│  │     - Configure APP_VECTORSTORE_NAME=milvus                 ││
│  │                                                             ││
│  │  2. Resource Scaling:                                       ││
│  │     - Increase memory allocation...                         ││
│  │                                                             ││
│  │  📋 Sources: installation_guide.pdf (p.15-18)              ││
│  │  ⏱️ Response time: 2.3s | 🔗 Relevance: 0.87               ││
│  │                                                             ││
│  │  👤 User (2:33 PM)                                          ││
│  │  What about security considerations?                        ││
│  │                                                             ││
│  │  🤖 Assistant (2:33 PM) [Typing...]                        ││
│  │                                                             ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  💬 Ask a question about your documents...                  ││
│  │                                                        [📤] ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  Quick Actions:                                                 │
│  [📄 Summarize Docs] [🔍 Search] [💡 Suggest Questions] [🔄]   │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Document Search View
```
┌─────────────────────────────────────────────────────────────────┐
│                       Document Search                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  🔍 Search in knowledge base...                       [Go] ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  Search Results (5 found):                                      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 📄 installation_guide.pdf                    Score: 0.92   ││
│  │ "...configure the environment variables including           ││
│  │ NVIDIA_API_KEY and set up the docker containers..."        ││
│  │ 🏷️ Tags: setup, configuration, environment                 ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 📄 user_manual.docx                          Score: 0.87   ││
│  │ "...production deployment requires careful consideration     ││
│  │ of security, scalability, and monitoring aspects..."       ││
│  │ 🏷️ Tags: production, deployment, security                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  Filters: [📄 File Type ▼] [📅 Date ▼] [🎯 Relevance ▼]       │
└─────────────────────────────────────────────────────────────────┘
```

### 4.5 Admin/Settings View
```
┌─────────────────────────────────────────────────────────────────┐
│                       System Settings                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   LLM Config    │  │  Vector Config  │  │   System Info   │ │
│  │                 │  │                 │  │                 │ │
│  │ Model:          │  │ Database:       │  │ Status: 🟢 OK   │ │
│  │ [llama3-70b ▼]  │  │ [Milvus ▼]      │  │                 │ │
│  │                 │  │                 │  │ Uptime: 2h 15m  │ │
│  │ Temperature:    │  │ Collection:     │  │                 │ │
│  │ [0.7     ━━●━━]  │  │ documents       │  │ Memory: 45%     │ │
│  │                 │  │                 │  │                 │ │
│  │ Max Tokens:     │  │ Dimensions:     │  │ CPU: 23%        │ │
│  │ [2048        ]  │  │ 1024            │  │                 │ │
│  │                 │  │                 │  │ Docs: 12        │ │
│  │ Top-K:          │  │ Index Type:     │  │                 │ │
│  │ [5           ]  │  │ IVF_FLAT        │  │ Vectors: 1,247  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    API Configuration                        ││
│  │                                                             ││
│  │  NVIDIA API Key: [••••••••••••••••••••••••nvapi-xyz] ✅     ││
│  │  Endpoint URL: [https://integrate.api.nvidia.com/v1]       ││
│  │  Rate Limits: [100 req/min] Usage: [23/100]                ││
│  │                                                             ││
│  │  [Test Connection]  [Reset to Defaults]  [Save Changes]    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 5. API Design

### 5.1 REST API Endpoints
```
Chain Server API (Port 8081):

POST /uploadDocument
├── Content-Type: multipart/form-data
├── Body: file + filename
└── Response: {"status": "success", "message": "Document uploaded"}

POST /generate  
├── Content-Type: application/json
├── Body: {"messages": [...], "use_knowledge_base": true}
└── Response: Stream of {"token": "...", "finished": false}

POST /search
├── Content-Type: application/json  
├── Body: {"query": "...", "num_docs": 5}
└── Response: [{"content": "...", "source": "...", "score": 0.9}]

GET /documents
├── Parameters: limit, offset
└── Response: {"documents": [...], "total": 12}

DELETE /documents
├── Content-Type: application/json
├── Body: {"filenames": ["doc1.pdf", "doc2.txt"]}
└── Response: {"deleted": 2, "failed": 0}
```

### 5.2 WebSocket Connections
```
WebSocket Endpoint: ws://localhost:8081/ws

Message Types:
├── document_upload_progress
│   └── {"type": "progress", "filename": "...", "percent": 75}
├── indexing_status  
│   └── {"type": "indexing", "status": "processing", "docs_count": 156}
└── chat_stream
    └── {"type": "token", "content": "Hello", "finished": false}
```

## 6. Data Models

### 6.1 Document Model
```typescript
interface Document {
    id: string;
    filename: string;
    content: string;
    metadata: {
        upload_time: string;
        file_size: number;
        file_type: string;
        chunk_count: number;
        processed: boolean;
    };
    embeddings?: number[];
}
```

### 6.2 Chat Message Model
```typescript
interface ChatMessage {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: string;
    metadata?: {
        sources?: Source[];
        response_time?: number;
        token_count?: number;
        relevance_score?: number;
    };
}

interface Source {
    filename: string;
    page?: number;
    chunk_id: string;
    relevance_score: number;
}
```

### 6.3 Configuration Model
```typescript
interface SystemConfig {
    llm: {
        model_name: string;
        temperature: number;
        max_tokens: number;
        top_k: number;
    };
    retriever: {
        top_k: number;
        score_threshold: number;
        search_type: 'similarity' | 'mmr';
    };
    vectorstore: {
        type: 'milvus' | 'faiss' | 'chroma';
        connection_string: string;
        collection_name: string;
    };
}
```

## 7. Security Design

### 7.1 Authentication Flow
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Browser   │    │  Frontend   │    │  Backend    │
└─────────────┘    └─────────────┘    └─────────────┘
        │                   │                   │
        │ 1. Login Request  │                   │
        ├──────────────────►│                   │
        │                   │ 2. Auth API Call  │
        │                   ├──────────────────►│
        │                   │                   │
        │                   │ 3. JWT Token      │
        │                   │◄──────────────────┤
        │ 4. Set Token      │                   │
        │◄──────────────────┤                   │
        │                   │                   │
        │ 5. API Calls      │                   │
        ├──────────────────►│ 6. API + Token    │
        │   (with token)    ├──────────────────►│
        │                   │                   │
```

### 7.2 Data Protection
```
Security Layers:
├── Transport Layer
│   ├── HTTPS/TLS encryption
│   └── WSS for WebSocket connections
├── Application Layer  
│   ├── JWT token validation
│   ├── Rate limiting
│   └── Input sanitization
├── Data Layer
│   ├── Vector database encryption
│   ├── Document encryption at rest
│   └── Secure API key storage
└── Infrastructure Layer
    ├── Docker container isolation
    ├── Network security groups
    └── Resource access controls
```

## 8. Performance Considerations

### 8.1 Optimization Points
```
Performance Hotspots:
├── Document Processing
│   ├── Large file parsing (PDF/DOCX)
│   ├── Text chunking algorithms
│   └── Embedding generation (API calls)
├── Vector Search
│   ├── Similarity computation
│   ├── Index optimization
│   └── Result ranking
├── LLM Generation
│   ├── Context length optimization
│   ├── Token streaming
│   └── Response caching
└── UI Rendering
    ├── Message virtualization
    ├── File upload progress
    └── Real-time updates
```

### 8.2 Scalability Design
```
Horizontal Scaling:
├── Load Balancer
│   └── Multiple Chain Server instances
├── Database Clustering
│   └── Milvus distributed deployment
├── Caching Layer
│   ├── Redis for embeddings cache
│   └── API response caching
└── CDN Integration
    └── Static asset delivery
```

---

**Note**: This design document should be updated as the system evolves and new requirements are identified.
