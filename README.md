<<<<<<< HEAD
# 🏆 FPT-Nvidia Hackathon Demo: S32DS AI CodeGen

## 🎯 Demo Overview

**Event**: FPT-Nvidia Hackathon  
**Timeline**: 2 weeks
**Team Size**: 4 people
**Demo Scope**: AI-powered SWTBot test generation for S32K144 project creation and build

## 🚀 Demo Objectives

### **Primary Goal**
Demonstrate an AI system that automatically generates Maven-based SWTBot test scripts for:
1. **S32K144 Project Creation** - Complete project setup workflow
2. **Build Execution** - Automated build process testing

### **Demo Flow (5-7 minutes)**
```
1. Input: User describes S32DS workflow in natural language
   ↓
2. AI Processing: RAG + Local LLM generates SWTBot code
   ↓
3. Output: Complete Maven project with executable tests
   ↓
4. Live Demo: Run generated tests against S32DS IDE
   ↓
5. Results: Successful S32K144 project creation and build
```

## 🎪 Demo Scenario

### **User Input Example**
```
"Create a new S32K144 project with LQFP100 package, 
configure it for ARM Cortex-M4F, and build the project"
```

### **AI Generated Output**
- Complete Maven project structure with data-driven tests
- Generic SWTBot test classes (no SoC-specific code)
- Parameterized test execution for multiple SoCs
- SoC configuration data (S32K144, S32K146, S32G274A, etc.)
- Ready-to-run test suite with JUnit Parameterized

### **Live Demonstration**
1. Show AI generating S32K144-specific code in real-time using **FPT AI Factory + Nvidia infrastructure**
2. Execute generated tests against actual S32DS IDE
3. Verify S32K144 project is created and built successfully
4. **Highlight Nvidia solutions**: GPU acceleration, TensorRT optimization, NIM services

## 📁 Demo Architecture

```
hackathon-demo/
├── README.md                    # This overview
├── demo-plan.md                 # Detailed 2-week plan
├── technical-specs.md           # Technical implementation
├── presentation/                # Demo presentation materials
├── sample-outputs/              # Example generated code
└── setup-guide.md              # Environment setup
```

## 🎯 Success Criteria

### **Must Have (MVP)**
- [x] AI generates generic SWTBot code with data-driven testing
- [x] AI generates SoC configuration data (S32K144, S32K146, S32G274A)
- [x] Generated parameterized tests run successfully in Maven
- [x] Live demo works reliably with multiple SoCs

### **Nice to Have**
- [ ] Web interface for SoC selection and input/output
- [ ] Real-time code generation visualization
- [ ] Extended SoC family support (S32K1, S32G, S32V)
- [ ] Advanced error handling and validation

## 👥 Team Roles (4 People)

### **Person 1: AI/ML Engineer**
- RAG system implementation
- Local LLM integration and fine-tuning
- Training data preparation
- Code generation pipeline

### **Person 2: S32DS/SWTBot Expert**
- S32DS workflow analysis
- SWTBot test implementation
- S32K144 specific configurations
- IDE integration testing

### **Person 3: Backend/Integration Developer**
- Maven project templates
- System integration
- API development
- Testing framework setup

### **Person 4: Frontend/Demo Lead**
- Demo presentation preparation
- User interface development
- Live demo coordination
- Documentation and presentation

## 📁 Demo Architecture & Documentation Links

```
hackathon-demo/
├── README.md                    # This overview
├── demo-plan.md                 # Detailed 2-week plan
├── technical-specs.md           # Technical implementation
├── team-assignments.md          # Individual tasks for 4 people
├── setup-guide.md              # Environment setup
├── sample-outputs/              # Example generated code
│   └── README.md               # Sample test code and outputs
└── presentation/                # Demo presentation materials (TBD)
```

### 📋 **Documentation Links**

| Document | Description | Link |
|----------|-------------|------|
| **Demo Overview** | Main overview, objectives, and team structure | [README.md](./README.md) |
| **2-Week Plan** | Detailed timeline, milestones, and deliverables | [demo-plan.md](./demo-plan.md) |
| **Technical Specs** | Architecture, data-driven testing, and implementation | [technical-specs.md](./technical-specs.md) |
| **Team Assignments** | Individual tasks for 4-person team | [team-assignments.md](./team-assignments.md) |
| **Setup Guide** | Environment setup and installation instructions | [setup-guide.md](./setup-guide.md) |
| **Sample Outputs** | Example generated SWTBot code and Maven projects | [sample-outputs/README.md](./sample-outputs/README.md) |

### 🎯 **Quick Navigation**

- **🚀 [Start Here: Demo Plan](./demo-plan.md)** - Begin with the 2-week implementation timeline
- **🔧 [Technical Implementation](./technical-specs.md)** - Data-driven testing architecture
- **👥 [Team Tasks](./team-assignments.md)** - Individual responsibilities and deliverables
- **⚙️ [Environment Setup](./setup-guide.md)** - FPT AI Factory, Nvidia, and S32DS configuration
- **📋 [Sample Code](./sample-outputs/README.md)** - Example parameterized SWTBot tests

## 📅 Key Milestones

- **Week 1**: Core AI system + Basic SWTBot templates
- **Week 2**: Integration + Demo preparation + Testing
- **Event Week**: Final polish + Live demonstration

## 🛠️ Technology Stack

- **AI/ML**: FPT AI Factory infrastructure, **Nvidia NIM** (if available), Local LLM (Ollama/LLaMA), RAG with vector DB
- **Training Infrastructure**: **FPT AI Factory** for model training and fine-tuning
- **GPU Acceleration**: **Nvidia CUDA/TensorRT** for inference optimization
- **Vector Processing**: **Nvidia Rapids** (if applicable) for embeddings
- **Testing**: SWTBot, JUnit, Maven
- **IDE**: S32DS (Eclipse-based)
- **Demo**: Live coding demonstration

Ready to dive into the detailed planning! 🚀
=======
# RAG System with NVIDIA NIM Containers

Enterprise-ready Retrieval Augmented Generation system porting from NVIDIA GenerativeAIExamples with enhanced local deployment capabilities using NVIDIA NIM containers.

## 🎯 **Features**

- **🤖 NVIDIA NIM Integration**: Local deployment with enterprise-grade containers
- **📚 Advanced RAG Pipeline**: Document ingestion, vector search, and LLM generation
- **🌐 Web Interface**: React-based RAG Playground for interactive Q&A
- **🗄️ Vector Database**: Milvus for high-performance vector storage
- **🔧 Flexible Deployment**: Support both cloud APIs and local NIM containers
- **🐳 Docker Orchestration**: Complete containerized deployment

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  RAG Playground │    │   Chain Server  │    │ Vector Database │
│ (React/Docker)  │◄──►│ (FastAPI/Docker)│◄──►│ (Milvus/Docker) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ NVIDIA NIM      │
                       │ • Llama3-8B     │
                       │ • E5-V5 Embed   │
                       │ • GPU Accel     │
                       │ • Enterprise    │
                       └─────────────────┘
```

## 🚀 **Quick Start**

### **Prerequisites**
- Docker & Docker Compose
- NVIDIA GPU (for NIM containers)
- NGC API Key ([Get here](https://ngc.nvidia.com/))

### **Deployment**

1. **Clone Repository**
   ```bash
   git clone https://github.com/ai-code-generation/fine-tune-demo.git
   cd fine-tune-demo
   git checkout porting_GenerativeAIExamples_RAG
   ```

2. **Setup Environment**
   ```bash
   cd basic_rag_system
   cp .env.example .env
   # Edit .env with your NGC_API_KEY
   ```

3. **Deploy with NIM Containers**
   ```bash
   cd langchain
   docker compose --profile local-nim up -d --build
   ```

4. **Access Application**
   - Web Interface: http://localhost:8090
   - API Documentation: http://localhost:8081/docs

## 📁 **Project Structure**

```
basic_rag_system/
├── 🎯 langchain/                    # RAG Implementation
│   ├── chains.py                   # Core RAG logic
│   ├── docker-compose.yaml         # Service orchestration
│   └── prompt.yaml                # LLM prompts
├── 🔌 chain_server/                # Backend API Server
│   ├── server.py                  # FastAPI endpoints
│   ├── utils.py                   # Utility functions
│   └── requirements.txt           # Dependencies
├── 🌐 rag_playground/              # Frontend Web UI
│   └── (React application)
├── 🏭 local_deploy/                # Infrastructure
│   ├── docker-compose-vectordb.yaml    # Milvus services
│   └── docker-compose-nim-ms.yaml      # NVIDIA NIM containers
└── 📖 Documentation files
```

## 🔧 **Configuration**

### **NIM Container Setup**
```bash
# .env configuration
NGC_API_KEY=your-ngc-api-key-here
APP_LLM_MODELNAME=meta/llama3-8b-instruct
APP_EMBEDDINGS_MODELNAME=nvidia/nv-embedqa-e5-v5
COLLECTION_NAME=rag_collection
```

### **Model Configuration**
- **LLM**: Meta Llama3-8B-Instruct (via NIM)
- **Embedding**: NVIDIA NV-EmbedQA-E5-V5 (via NIM)
- **Vector DB**: Milvus with L2 similarity
- **Text Splitter**: SentenceTransformers token-based

## 📚 **Documentation**

- [📋 RAG System Design](RAG_System_Design.md)
- [🔧 Code Flow Guide](RAG_CODE_FLOW_GUIDE.md)
- [💼 Enterprise Licensing](ENTERPRISE_LICENSING_ANALYSIS.md)
- [🔄 Project Comparison](PROJECT_COMPARISON_ANALYSIS.md)
- [🚀 NIM Setup Guide](basic_rag_system/NIM_SETUP_GUIDE.md)

## 🎯 **Use Cases**

- **📖 Document Q&A**: Upload PDFs, TXT, MD files for interactive questioning
- **🏢 Enterprise Knowledge Base**: Internal document search and retrieval
- **🔍 Research Assistant**: Academic paper analysis and summarization
- **📊 Technical Documentation**: API docs, manuals, and guides processing

## 🛡️ **Enterprise Features**

- **🔒 Local Deployment**: Complete on-premises processing
- **🚀 GPU Optimization**: NVIDIA NIM container acceleration
- **📈 Scalable Architecture**: Multi-container orchestration
- **🔧 Configurable**: Flexible model and parameter settings
- **📊 Monitoring**: Health checks and observability

## 💰 **Cost Analysis**

### **Free Alternative Stack**
- **Total Cost**: $0/month
- **LLM**: Local models (Ollama)
- **Embedding**: Sentence-transformers
- **Performance**: Good

### **NVIDIA NIM Stack**
- **Total Cost**: $5,000-$50,000+/year
- **LLM**: NVIDIA NIM containers
- **Embedding**: NVIDIA NIM embedding
- **Performance**: Excellent

## 🔄 **Migration from NVIDIA Examples**

This project is a direct port of [NVIDIA GenerativeAIExamples](https://github.com/NVIDIA/GenerativeAIExamples/tree/main/RAG) with:

- **✅ 100% Core Logic Preservation**: RAG algorithms unchanged
- **🆕 Enhanced Deployment**: Added NIM container support
- **🔧 Simplified Structure**: Easier development and deployment
- **📚 Comprehensive Documentation**: Detailed guides and analysis

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 **License**

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Original NVIDIA GenerativeAIExamples code is also under Apache 2.0 license.

## 🙏 **Acknowledgments**

- [NVIDIA GenerativeAIExamples](https://github.com/NVIDIA/GenerativeAIExamples) - Original implementation
- [LangChain](https://langchain.com/) - RAG framework
- [Milvus](https://milvus.io/) - Vector database
- [FastAPI](https://fastapi.tiangolo.com/) - API framework

## 📞 **Support**

For issues and questions:
- 🐛 [GitHub Issues](https://github.com/ai-code-generation/fine-tune-demo/issues)
- 📖 [Documentation](RAG_CODE_FLOW_GUIDE.md)
- 💼 [Enterprise Licensing Guide](ENTERPRISE_LICENSING_ANALYSIS.md)

---

**🎯 Ready for production-grade RAG deployment with NVIDIA NIM containers!**
>>>>>>> 2c0c044 (porting GenerativeAIExamples RAG)
