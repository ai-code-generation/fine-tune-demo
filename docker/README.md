# Docker Setup for NeMo Fine-Tuning Pipeline

This directory contains Docker configuration for running the NeMo fine-tuning pipeline in a containerized environment with persistent model outputs.

## 🐳 Quick Start

### Prerequisites

1. **Docker & Docker Compose**
   ```bash
   # Install Docker (Ubuntu/Debian)
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

2. **NVIDIA Docker (for GPU support)**
   ```bash
   # Install NVIDIA Docker runtime
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

3. **Verify GPU Support**
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
   ```

### Basic Usage

1. **Build and Run Fine-Tuning**
   ```bash
   # Make script executable
   chmod +x scripts/docker_run.sh
   
   # Build Docker image and run S32K144 fine-tuning
   ./scripts/docker_run.sh run
   ```

2. **Monitor Training Progress**
   ```bash
   # View logs
   ./scripts/docker_run.sh logs
   
   # Or directly with Docker
   docker logs -f nemo-finetune-s32k144
   ```

3. **Access Results**
   ```bash
   # Trained models will be in:
   ls outputs/checkpoints/
   
   # Deployment artifacts:
   ls outputs/deploy/
   
   # Training logs:
   ls outputs/logs/
   ```

## 📁 Volume Mounts

The Docker setup uses persistent volumes to preserve your training results:

```
Host Directory          -> Container Path           Purpose
./data                  -> /workspace/data          Training data (read-only)
./outputs/checkpoints   -> /workspace/checkpoints   Model checkpoints
./outputs/logs          -> /workspace/logs          Training logs
./outputs/deploy        -> /workspace/deploy        Deployed models
./outputs/cache         -> /workspace/.cache        Model cache
./configs               -> /workspace/configs       Configuration files (read-only)
```

## 🚀 Available Commands

### Build & Run
```bash
# Build Docker image
./scripts/docker_run.sh build

# Run fine-tuning with default settings (LLaMA2 7B)
./scripts/docker_run.sh run

# Run with custom parameters
./scripts/docker_run.sh --model-type llama3 --model-size 8b --epochs 5 --gpus 2 run
```

### Development & Debugging
```bash
# Start interactive shell
./scripts/docker_run.sh shell

# Start Jupyter Lab (accessible at http://localhost:8888)
./scripts/docker_run.sh jupyter

# Start TensorBoard (accessible at http://localhost:6006)
./scripts/docker_run.sh tensorboard
```

### Monitoring & Management
```bash
# View training logs
./scripts/docker_run.sh logs

# Stop all containers
./scripts/docker_run.sh stop

# Clean up containers and images
./scripts/docker_run.sh clean
```

## ⚙️ Configuration Options

### Model Configuration
```bash
# LLaMA2 7B (default)
./scripts/docker_run.sh --model-type llama2 --model-size 7b run

# LLaMA3 8B
./scripts/docker_run.sh --model-type llama3 --model-size 8b run

# CodeLlama 7B for code generation
./scripts/docker_run.sh --model-type codellama --model-size 7b run
```

### Training Parameters
```bash
# Extended training
./scripts/docker_run.sh --epochs 10 --batch-size 4 run

# Multi-GPU training
./scripts/docker_run.sh --gpus 2 run

# Custom container name
./scripts/docker_run.sh --container my-finetune-job run
```

## 🔧 Docker Compose Alternative

For more complex setups, use Docker Compose:

```bash
# Start all services (fine-tuning + TensorBoard + Jupyter)
docker-compose up -d

# Start only fine-tuning
docker-compose up -d nemo-finetune

# View logs
docker-compose logs -f nemo-finetune

# Stop all services
docker-compose down
```

## 📊 Monitoring Training

### 1. Real-time Logs
```bash
# Follow training logs
docker logs -f nemo-finetune-s32k144

# Or using the script
./scripts/docker_run.sh logs
```

### 2. TensorBoard
```bash
# Start TensorBoard
./scripts/docker_run.sh tensorboard

# Access at: http://localhost:6006
```

### 3. Jupyter Lab
```bash
# Start Jupyter Lab
./scripts/docker_run.sh jupyter

# Access at: http://localhost:8888
```

### 4. Resource Monitoring
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor container resources
docker stats nemo-finetune-s32k144
```

## 📂 Output Structure

After training, your outputs will be organized as:

```
outputs/
├── checkpoints/           # Model checkpoints
│   ├── epoch_1.ckpt
│   ├── epoch_2.ckpt
│   └── final_model.nemo
├── logs/                  # Training logs
│   ├── tensorboard/
│   ├── training.log
│   └── evaluation_results.json
├── deploy/                # Deployed models
│   ├── model_name/
│   │   ├── model.nemo
│   │   ├── tokenizer/
│   │   ├── config.yaml
│   │   ├── inference.py
│   │   └── README.md
└── cache/                 # Model cache
    ├── nemo/
    ├── huggingface/
    └── wandb/
```

## 🐛 Troubleshooting

### Common Issues

1. **GPU Not Available**
   ```bash
   # Check NVIDIA Docker installation
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
   
   # If fails, reinstall nvidia-docker2
   sudo apt-get purge nvidia-docker2
   sudo apt-get install nvidia-docker2
   sudo systemctl restart docker
   ```

2. **Out of Memory**
   ```bash
   # Reduce batch size
   ./scripts/docker_run.sh --batch-size 1 run
   
   # Or use gradient accumulation in config
   ```

3. **Permission Issues**
   ```bash
   # Fix output directory permissions
   sudo chown -R $USER:$USER outputs/
   chmod -R 755 outputs/
   ```

4. **Container Won't Start**
   ```bash
   # Check container logs
   docker logs nemo-finetune-s32k144
   
   # Remove and recreate
   docker rm -f nemo-finetune-s32k144
   ./scripts/docker_run.sh run
   ```

### Performance Optimization

1. **Multi-GPU Training**
   ```bash
   # Use all available GPUs
   ./scripts/docker_run.sh --gpus $(nvidia-smi -L | wc -l) run
   ```

2. **Memory Optimization**
   ```bash
   # Enable mixed precision and gradient checkpointing in config
   # Reduce sequence length for memory-constrained environments
   ```

3. **Storage Optimization**
   ```bash
   # Use SSD for outputs directory
   # Clean up old checkpoints periodically
   ```

## 🔒 Security Considerations

1. **Non-root User**: Container runs as non-root user (uid 1000)
2. **Read-only Mounts**: Training data and configs are mounted read-only
3. **Network Isolation**: Container uses bridge network by default
4. **Resource Limits**: Memory and GPU limits can be configured

## 📝 Environment Variables

Key environment variables in the container:

```bash
NVIDIA_VISIBLE_DEVICES=all          # GPU visibility
CUDA_VISIBLE_DEVICES=all            # CUDA device selection
PYTHONPATH=/workspace/src           # Python path
NEMO_CACHE_DIR=/workspace/.cache/nemo    # NeMo cache
HF_HOME=/workspace/.cache/huggingface   # HuggingFace cache
WANDB_PROJECT=nemo-s32k144-finetune     # Weights & Biases project
```

## 🎯 Next Steps

1. **Start Training**: `./scripts/docker_run.sh run`
2. **Monitor Progress**: `./scripts/docker_run.sh logs`
3. **Access Results**: Check `outputs/` directory
4. **Deploy Model**: Use generated inference scripts in `outputs/deploy/`

For advanced usage and customization, see the main documentation in `docs/USAGE.md`.
