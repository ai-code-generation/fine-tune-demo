# LLM Server - Optimized for Code Models

OpenAI-compatible chat completion API server using local HuggingFace models. **Specially optimized for CodeLlama-13B, StarCoder-15B, and DeepSeek-Coder models**.

## 🎯 Key Optimizations

### **Code Model Support**
- ✅ **CodeLlama**: Instruction format with `[INST]` tokens
- ✅ **StarCoder**: Code-focused prompting with `# Solution:` format  
- ✅ **DeepSeek-Coder**: `### Instruction:` / `### Response:` format
- ✅ **Auto-detection** of model type for optimal parameters

### **Smart Response Extraction**
- ✅ **Clean responses** - no prompt echo in output
- ✅ **Multiple extraction methods** with intelligent fallbacks
- ✅ **Code-specific post-processing** for better formatting
- ✅ **Repetition removal** and artifact cleanup

### **Performance Optimizations**
- ✅ **Smart device/dtype detection** - auto float32 for CPU, float16 for GPU
- ✅ **Model-specific generation parameters** - lower temperature for code models
- ✅ **Memory optimization** for large models (13B, 15B parameters)
- ✅ **Longer context** support for code models (4K vs 2K tokens)

## 📁 Project Structure

```
llm-server/
├── models/                  # Local models directory
│   └── my-finetuned-model/ # Default model location
├── app.py                  # FastAPI server
├── Dockerfile              # Container definition
├── docker-compose.yaml     # Docker deployment
├── .env                    # Environment variables
├── deploy.sh               # Deployment script
├── run_manual.sh           # Manual development mode
├── test_manual.sh          # Test script for manual mode
├── API_DOCUMENTATION.md    # Complete API reference
├── MANUAL_MODE.md          # Development guide
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Copy environment template
cp .env.example .env

# Edit configuration - all variables are loaded automatically
nano .env
```

### 2. Configure Environment Variables
Edit the `.env` file with your specific settings. The Docker container will automatically load all variables from this file.

### 3. Prepare Model
```bash
# Option A: Use existing model in models/
ls models/

# Option B: Download new model
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
model.save_pretrained('./models/my-finetuned-model')
tokenizer.save_pretrained('./models/my-finetuned-model')
"
```

### 4. Deploy with Docker
```bash
# Auto-deploy with GPU detection
./deploy.sh

# Or manual deployment
docker-compose up -d --build
```

### 5. Test API
```bash
# Health check
curl http://localhost:8884/v1/health/ready

# Chat completion test
curl -X POST http://localhost:8884/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## ⚙️ Configuration (.env)

All configuration is managed through environment variables in the `.env` file:

```env
# Model Configuration (Local models take priority)
HF_MODEL_LOCAL_PATH=./models/my-finetuned-model
HF_MODEL_NAME=microsoft/DialoGPT-medium

# HuggingFace Settings  
TRUST_REMOTE_CODE=false     # Set to true for models requiring custom code

# Server Configuration
HOST=0.0.0.0
PORT=8884

# Performance Settings
DEVICE=auto                 # auto, cpu, cuda
TORCH_DTYPE=float16         # float16, float32
LOW_CPU_MEM_USAGE=true      # Optimize memory usage

# Logging
LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR
```

**Note**: All environment variables are loaded from the `.env` file automatically by Docker Compose.

## 🛠️ Development Mode

For development without Docker:

```bash
# One-time setup
./setup_dev.sh

# Start development server
./run_manual.sh

# Test in another terminal
./test_manual.sh
```

## 🐳 Docker Operations

```bash
# Deploy
./deploy.sh                           # Smart deployment

# Manual operations
docker-compose up -d --build          # Start services
docker-compose down                   # Stop services
docker logs -f llm-server             # View logs
docker stats llm-server               # Monitor resources
```

**Environment Configuration**: Docker Compose automatically loads all environment variables from the `.env` file using `env_file` directive. No need to declare variables individually in docker-compose.yaml.

## 📚 API Endpoints

- **Health**: `GET /v1/health/ready`
- **Models**: `GET /v1/models`  
- **Chat**: `POST /v1/chat/completions`

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| GPU errors in WSL2 | Deploy script auto-handles CPU fallback |
| Port 8884 in use | Change `PORT=8001` in `.env` |
| Model not found | Check `HF_MODEL_LOCAL_PATH` in `.env` |
| Out of memory | Use `DEVICE=cpu` or smaller model |
| Permission denied | `chmod -R 755 models/` |

### Debug Commands
```bash
# Container status
docker ps

# Detailed logs  
docker logs llm-server

# Test connectivity
curl http://localhost:8884/v1/health/ready

# Resource usage
docker stats llm-server
```

## 📖 Additional Documentation

- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) - Complete API reference
- [MANUAL_MODE.md](MANUAL_MODE.md) - Development without Docker

---

**Quick Deploy**: `./deploy.sh` → Test: `curl localhost:8884/v1/health/ready` → Ready! 🚀
