# 🔧 Environment Variables Management

This project now uses **deployment-specific environment variable management** for better organization and consistency.

## 📋 Overview

| Deployment Method | Configuration Location | Benefits |
|-------------------|------------------------|----------|
| 🐳 **Docker** | `docker-compose.yaml` | Single file to edit, no .env confusion |
| 🔧 **Manual** | `run_manual.sh` | Embedded variables, command-line overrides |

## 🐳 Docker Deployment

### Configuration Location
All environment variables are embedded directly in `docker-compose.yaml`:

```yaml
services:
  llm-server:
    environment:
      # Model Configuration
      HF_MODEL_LOCAL_PATH: "./models/CodeLlama-13b-Instruct-hf"
      HF_MODEL_NAME: "codellama/CodeLlama-13b-Instruct-hf"
      
      # HuggingFace Settings
      TRUST_REMOTE_CODE: "true"
      
      # Server Configuration
      HOST: "0.0.0.0"
      PORT: "8884"
      
      # Performance Settings
      DEVICE: "auto"
      TORCH_DTYPE: "auto"
      LOW_CPU_MEM_USAGE: "true"
      
      # Logging
      LOG_LEVEL: "INFO"
```

### How to Change Configuration
1. Edit `docker-compose.yaml`
2. Modify values in the `environment` section
3. Restart: `docker-compose down && docker-compose up -d --build`

### Example: Change Port to 8001
```yaml
environment:
  PORT: "8001"
```

The ports mapping and healthcheck will automatically use the PORT environment variable:
```yaml
ports:
  - "${PORT:-8884}:${PORT:-8884}"  # Automatically uses PORT env var
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:$$PORT/v1/health/ready"]
```

## 🔧 Manual Deployment

### Configuration Location
All environment variables are embedded in `run_manual.sh` script in the `setup_environment()` function.

### How to Change Configuration

#### Method 1: Edit Script (Permanent)
Edit `run_manual.sh` and modify the `setup_environment()` function:

```bash
# Server Configuration
export HOST="${HOST}"
export PORT="${PORT}"
```

#### Method 2: Command Line (Temporary)
Use command-line arguments to override defaults:

```bash
# Change port
./run_manual.sh --port 8001

# Change device to CPU only
./run_manual.sh --device cpu

# Multiple overrides
./run_manual.sh --port 8001 --device cpu --log-level debug
```

### Available Command Line Options
```bash
--port PORT         # Server port (default: 8884)
--host HOST         # Server host (default: 127.0.0.1)
--device DEVICE     # Device: cpu, cuda, auto (default: auto)
--model-path PATH   # Path to local model
--log-level LEVEL   # Log level: debug, info, warning, error
--reload            # Enable auto-reload for development
```

## 📊 Environment Variables Reference

| Variable | Docker Default | Manual Default | Description |
|----------|----------------|----------------|-------------|
| `HF_MODEL_LOCAL_PATH` | `./models/CodeLlama-13b-Instruct-hf` | `./models/CodeLlama-13b-Instruct-hf` | Path to local model |
| `HF_MODEL_NAME` | `codellama/CodeLlama-13b-Instruct-hf` | `codellama/CodeLlama-13b-Instruct-hf` | HuggingFace model name |
| `TRUST_REMOTE_CODE` | `true` | `true` | Allow custom model code |
| `HOST` | `0.0.0.0` | `127.0.0.1` | Server host |
| `PORT` | `8884` | `8884` | Server port |
| `DEVICE` | `auto` | `auto` | Device: auto/cpu/cuda |
| `TORCH_DTYPE` | `auto` | `auto` | Torch data type |
| `LOW_CPU_MEM_USAGE` | `true` | `true` | Optimize memory usage |
| `LOG_LEVEL` | `INFO` | `INFO` | Log level |

## ✅ Benefits of This Approach

### ✅ Consistency
- Each deployment method has its own single configuration file
- No confusion about which .env file is being used
- Clear separation of concerns

### ✅ Simplicity
- **Docker**: Edit `docker-compose.yaml` → restart
- **Manual**: Edit `run_manual.sh` or use command-line args

### ✅ Version Control Friendly
- All configuration is in tracked files
- No .env files to ignore or manage
- Easy to see configuration changes in git diffs

### ✅ No .env File Confusion
- No more wondering which .env file is being loaded
- No more .env vs .env.example synchronization issues
- No more environment variable precedence confusion

## 🚀 Quick Start Examples

### Docker Deployment
```bash
# Use default configuration (port 8884)
./deploy.sh

# To change port to 8001:
# 1. Edit docker-compose.yaml:
#    environment:
#      PORT: "8001"
# 2. Restart: ./deploy.sh --down && ./deploy.sh
# 3. Access at: http://localhost:8001
```

### Manual Deployment
```bash
# Use default configuration
./run_manual.sh

# Override specific settings
./run_manual.sh --port 8001 --device cpu

# Development mode
./run_manual.sh --reload --log-level debug
```

## 🔄 Migration from .env Files

The old `.env` file approach has been completely removed. If you have existing `.env` files:

1. **For Docker**: Copy values to `docker-compose.yaml` environment section
2. **For Manual**: Values are already set in `run_manual.sh` or use command-line args
3. **Remove**: Delete any `.env` files to avoid confusion

This new approach provides better organization, clearer configuration management, and eliminates common .env file issues.
