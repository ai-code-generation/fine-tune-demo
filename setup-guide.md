# 🛠️ Demo Environment Setup Guide

## 🎯 Overview

Complete setup guide for the FPT-Nvidia Hackathon demo environment. This guide ensures all team members have consistent development and demo environments.

## 💻 System Requirements

### **Hardware Requirements**
- **CPU**: Intel i5/AMD Ryzen 5 or better
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB free space for S32DS and tools
- **Display**: Dual monitor setup recommended for demo

### **Operating System**
- **Primary**: Windows 10/11 (S32DS compatibility)
- **Alternative**: Linux with Windows VM for S32DS

## 🔧 Core Software Installation

### **1. Java Development Kit**
```bash
# Download and install JDK 11 or higher
# Verify installation
java -version
javac -version

# Set JAVA_HOME environment variable
export JAVA_HOME=/path/to/jdk
```

### **2. Maven**
```bash
# Download Maven 3.6+
# Add to PATH
export PATH=$PATH:/path/to/maven/bin

# Verify installation
mvn -version
```

### **3. Git**
```bash
# Install Git
# Configure user
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### **4. Python Environment**
```bash
# Install Python 3.8+
python --version

# Create virtual environment
python -m venv hackathon-demo
source hackathon-demo/bin/activate  # Linux/Mac
# or
hackathon-demo\Scripts\activate     # Windows

# Install required packages (including Nvidia support)
pip install ollama transformers sentence-transformers numpy pandas
pip install nvidia-ml-py cupy-cuda11x  # Nvidia GPU support
pip install tensorrt  # Nvidia TensorRT (if available)
```

## 🏭 S32DS IDE Setup

### **1. Download S32DS**
- Download from NXP official website
- Version: Latest stable release
- License: Evaluation license for demo

### **2. S32DS Installation**
```bash
# Run installer as administrator
# Select installation directory: C:\NXP\S32DS
# Install with default components
```

### **3. S32K144 SDK Installation**
- Open S32DS
- Go to Help → Install New Software
- Install S32K1 SDK (latest version)
- Verify S32K144 processor support

### **4. Workspace Configuration**
```
# Create clean workspace for demo
Workspace: C:\S32DS_Demo_Workspace

# Configure workspace settings:
- Auto-build: Disabled (for demo control)
- Perspectives: S32DS perspective
- Views: Project Explorer, Console, Problems
```

## 🤖 AI Environment Setup

### **1. FPT AI Factory Access**
```bash
# Setup FPT AI Factory credentials
export FPT_AI_FACTORY_API_KEY="your-api-key"
export FPT_AI_FACTORY_ENDPOINT="https://ai-factory.fpt.com"

# Test FPT AI Factory connection
curl -H "Authorization: Bearer $FPT_AI_FACTORY_API_KEY" \
     $FPT_AI_FACTORY_ENDPOINT/health
```

### **2. Nvidia Environment Setup**
```bash
# Check Nvidia GPU availability
nvidia-smi

# Install Nvidia Container Toolkit (if using Docker)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Test Nvidia CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### **3. Ollama Installation (Backup)**
```bash
# Install Ollama as backup
curl -fsSL https://ollama.ai/install.sh | sh

# Pull CodeLlama model
ollama pull codellama:7b

# Test installation
ollama run codellama:7b "Hello, world!"
```

### **4. Vector Database Setup**
```python
# Vector store with Nvidia acceleration support
pip install faiss-gpu chromadb  # GPU-accelerated FAISS
pip install cudf cuml  # Nvidia Rapids (if available)

# Test vector store
python -c "import faiss; print('FAISS-GPU installed successfully')"
python -c "import cudf; print('Nvidia Rapids available')" || echo "Rapids not available, using CPU"
```

### **5. RAG System Setup**
```bash
# Clone demo repository
git clone <demo-repo-url>
cd hackathon-demo

# Install dependencies with Nvidia support
pip install -r requirements.txt
pip install -r requirements-nvidia.txt  # Nvidia-specific packages

# Setup FPT AI Factory integration
python setup_fpt_ai_factory.py

# Test RAG system with S32K144 data
python test_s32k144_rag.py
```

## 🧪 SWTBot Testing Setup

### **1. SWTBot Dependencies**
```xml
<!-- Add to Maven pom.xml -->
<dependency>
    <groupId>org.eclipse.swtbot</groupId>
    <artifactId>org.eclipse.swtbot.eclipse.finder</artifactId>
    <version>3.1.0</version>
</dependency>
```

### **2. Test Environment Configuration**
```bash
# Set system properties for SWTBot
export SWT_GTK3=0  # Linux only
export DISPLAY=:0  # Linux only

# Windows: No additional configuration needed
```

### **3. S32DS Test Configuration**
```java
// Create test base class
@Before
public void setUp() {
    bot = new SWTWorkbenchBot();
    bot.perspectiveByLabel("S32DS").activate();
    
    // Set timeouts
    SWTBotPreferences.TIMEOUT = 30000;
    SWTBotPreferences.PLAYBACK_DELAY = 100;
}
```

## 📁 Project Structure Setup

### **1. Demo Project Structure**
```
hackathon-demo/
├── ai-components/           # AI/ML code
│   ├── rag_system.py
│   ├── code_generator.py
│   └── models/
├── swtbot-templates/        # SWTBot test templates
│   ├── project_creation.java
│   └── build_execution.java
├── maven-generator/         # Maven project generator
│   ├── generator.py
│   └── templates/
├── demo-interface/          # CLI and demo interface
│   ├── cli.py
│   └── demo_runner.py
├── test-data/              # Sample inputs and outputs
│   ├── sample_inputs/
│   └── expected_outputs/
└── docs/                   # Documentation
    └── setup-guide.md
```

### **2. Environment Variables**
```bash
# Create .env file
S32DS_PATH=C:\NXP\S32DS
WORKSPACE_PATH=C:\S32DS_Demo_Workspace
OLLAMA_HOST=localhost:11434
DEMO_OUTPUT_DIR=./generated-projects
```

## 🎪 Demo Day Setup

### **1. Hardware Setup**
- **Primary Monitor**: S32DS IDE display
- **Secondary Monitor**: Demo interface and presentation
- **Backup Laptop**: Pre-configured with same environment
- **Network**: Stable internet for LLM if needed

### **2. Software Preparation**
```bash
# Pre-start all services
ollama serve &
cd hackathon-demo && python demo_server.py &

# Open S32DS with clean workspace
# Verify all components working
python test_demo_flow.py
```

### **3. Demo Materials**
- [ ] Presentation slides loaded
- [ ] Sample inputs prepared
- [ ] Backup videos ready
- [ ] Technical documentation printed
- [ ] Team contact information

### **4. Contingency Plans**
- **Plan A**: Live demo with real-time generation
- **Plan B**: Pre-generated code with live execution
- **Plan C**: Video demonstration with live Q&A
- **Plan D**: Presentation-only with technical discussion

## 🔍 Testing and Validation

### **1. Component Testing**
```bash
# Test AI components
python test_ai_components.py

# Test SWTBot scripts
mvn test -Dtest=ProjectCreationTest

# Test integration
python test_integration.py
```

### **2. Demo Rehearsal Checklist**
- [ ] AI generates valid code (< 10 seconds)
- [ ] Maven project builds successfully
- [ ] SWTBot tests execute without errors
- [ ] S32K144 project created in S32DS
- [ ] Build process completes successfully
- [ ] Demo completes in < 7 minutes

### **3. Performance Validation**
- [ ] Code generation: < 10 seconds
- [ ] Project creation: < 2 minutes
- [ ] Build execution: < 3 minutes
- [ ] Total demo time: < 7 minutes

## 🚨 Troubleshooting

### **Common Issues**

#### **S32DS Issues**
```bash
# S32DS won't start
- Check Java version (JDK 11+)
- Verify workspace permissions
- Clear workspace metadata

# SWTBot can't find elements
- Check S32DS perspective is active
- Verify element IDs haven't changed
- Increase timeout values
```

#### **AI Issues**
```bash
# Ollama connection failed
ollama serve --host 0.0.0.0:11434

# Model not responding
ollama pull codellama:7b --force

# RAG system errors
- Check vector database initialization
- Verify training data format
- Test with simple queries
```

#### **Integration Issues**
```bash
# Maven build failures
- Check Java version compatibility
- Verify SWTBot dependencies
- Clean and rebuild: mvn clean compile

# Demo flow broken
- Test each component individually
- Check file permissions
- Verify environment variables
```

### **Emergency Contacts**
- **Person 1 (AI)**: [Contact info]
- **Person 2 (SWTBot)**: [Contact info]
- **Person 3 (Integration)**: [Contact info]
- **Person 4 (Demo Lead)**: [Contact info]

## ✅ Setup Verification

### **Final Checklist**
- [ ] All software installed and configured
- [ ] S32DS working with S32K144 SDK
- [ ] AI components responding correctly
- [ ] SWTBot tests executing successfully
- [ ] Integration pipeline working end-to-end
- [ ] Demo rehearsal completed successfully
- [ ] Backup plans prepared and tested

Ready for the hackathon! 🚀
