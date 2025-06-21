# Docker Setup Guide - Self-Contained CUDA

This guide covers the complete Docker setup for the CodeLlama fine-tuning pipeline with **self-contained CUDA installation**. No CUDA installation required on the host machine!

## Prerequisites

### 1. Docker Only (Required)

- **Docker** (any recent version)
- **Docker Compose** (optional, for easier management)

Install Docker:
```bash
# Linux
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Or use your package manager
sudo apt-get install docker.io docker-compose
```

### 2. NVIDIA GPU and Drivers (Optional, for GPU acceleration)

- **NVIDIA GPU** with compute capability 6.0+ (Pascal architecture or newer)
- **NVIDIA drivers** version 450.80.02 or newer (on host only)
- **At least 8GB GPU memory** (recommended for CodeLlama 7B)

Check your GPU (if available):
```bash
nvidia-smi
```

**Note:** The container includes its own CUDA toolkit, so you don't need CUDA installed on the host!

### 3. Docker GPU Support (Optional)

For GPU acceleration, install NVIDIA Container Toolkit:

```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Without GPU:** The container will automatically fall back to CPU mode.

## Quick Setup

### 1. Clone and Build

```bash
git clone <your-repo>
cd fine-tune-pipeline_nnemo

# Build Docker image with all dependencies
./scripts/build_docker.sh
```

### 2. Start Container

```bash
# Start the container
./scripts/docker_setup.sh start

# Verify installation
docker exec -it codellama-finetune python scripts/verify_installation.py
```

### 3. Create Sample Data and Train

```bash
# Create sample training data
./scripts/docker_setup.sh sample-data

# Start training
./scripts/docker_setup.sh train
```

## Detailed Setup Steps

### Step 1: Build the Docker Image

The build script handles everything automatically:

```bash
./scripts/build_docker.sh
```

This will:
- ✅ Check Docker installation
- ✅ Build self-contained image with CUDA 12.1 toolkit
- ✅ Install PyTorch with CUDA support
- ✅ Install all ML libraries (Transformers, PEFT, etc.)
- ✅ Verify the installation
- ✅ Test functionality (GPU or CPU mode)

**Build options:**
```bash
./scripts/build_docker.sh --no-cache    # Clean build
./scripts/build_docker.sh --build-only  # Build without testing
./scripts/build_docker.sh --test-only   # Test existing image
./scripts/build_docker.sh --dev         # Setup development mode
```

### Step 2: Container Management

**Start the container:**
```bash
docker-compose up -d codellama-finetune
```

**Access the container:**
```bash
docker exec -it codellama-finetune bash
```

**Check container status:**
```bash
docker-compose ps
```

**View logs:**
```bash
docker-compose logs -f codellama-finetune
```

### Step 3: Verification

Run the comprehensive verification script:
```bash
docker exec -it codellama-finetune python scripts/verify_installation.py
```

This checks:
- ✅ Python environment
- ✅ CUDA installation and drivers
- ✅ PyTorch CUDA support
- ✅ All required packages
- ✅ Pipeline modules
- ✅ File structure
- ✅ Basic functionality

## Container Features

### Installed Software

**Base System:**
- Ubuntu 22.04 LTS
- **CUDA 12.1 Toolkit** (self-contained, no host dependency)
- Python 3.10 with development headers
- Build tools (gcc, cmake, ninja)

**Python Packages:**
- PyTorch 2.1.0 with CUDA 12.1 support
- Transformers 4.35.2
- PEFT 0.6.2
- Datasets 2.14.6
- BitsAndBytes 0.41.3
- All other ML dependencies

**System Utilities:**
- vim, nano (text editors)
- htop, tmux (system monitoring)
- git, wget, curl (utilities)
- tree (directory visualization)

**Key Features:**
- ✅ **No host CUDA required** - everything is self-contained
- ✅ **Automatic GPU/CPU detection** - works in both modes
- ✅ **Complete ML stack** - ready to use out of the box

### Volume Mounts

The container automatically mounts:
- `./output` → `/workspace/output` (trained models)
- `./data` → `/workspace/data` (training data)
- `./configs` → `/workspace/configs` (configurations)
- `./cache` → `/workspace/cache` (model cache)
- `./logs` → `/workspace/logs` (training logs)

### Environment Variables

Pre-configured environment:
- `CUDA_HOME=/usr/local/cuda`
- `HF_HOME=/workspace/cache`
- `TRANSFORMERS_CACHE=/workspace/cache`
- `TORCH_HOME=/workspace/cache`
- `PYTHONPATH=/workspace/src`

## Training Workflow

### 1. Prepare Data

```bash
# Create sample data
docker exec -it codellama-finetune python train.py --create-sample-data

# Or copy your own data to ./data/ directory
cp your_training_data.yaml ./data/train.yaml
```

### 2. Configure Training

Edit configuration files in `./configs/`:
- `model_configs/codellama_7b.yaml` - Model settings
- `lora_configs/lora_default.yaml` - LoRA parameters

### 3. Start Training

```bash
# Using helper script
./scripts/docker_setup.sh train

# Or directly
docker exec -it codellama-finetune python train.py \
    --model-config configs/model_configs/codellama_7b.yaml \
    --train-data data/train.yaml \
    --eval-data data/validation.yaml
```

### 4. Monitor Training

**TensorBoard:**
```bash
./scripts/docker_setup.sh tensorboard
# Access at http://localhost:6006
```

**Container logs:**
```bash
docker-compose logs -f codellama-finetune
```

**GPU monitoring:**
```bash
watch -n 1 nvidia-smi
```

## Troubleshooting

### Common Issues

**1. "nvidia-smi not found" or "CUDA not available"**
- Check NVIDIA drivers: `nvidia-smi`
- Restart Docker: `sudo systemctl restart docker`
- Rebuild image: `./scripts/build_docker.sh --no-cache`

**2. "Out of memory" errors**
- Reduce batch size in model config
- Enable gradient checkpointing
- Use 4-bit quantization

**3. "Permission denied" errors**
- Check file permissions: `ls -la`
- Fix ownership: `sudo chown -R $USER:$USER .`

**4. Container won't start**
- Check Docker daemon: `sudo systemctl status docker`
- Check GPU support: `docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi`

### Debug Commands

**Check GPU in container:**
```bash
docker exec -it codellama-finetune nvidia-smi
```

**Test CUDA in container:**
```bash
docker exec -it codellama-finetune python -c "import torch; print(torch.cuda.is_available())"
```

**Check container resources:**
```bash
docker stats codellama-finetune
```

**Inspect container:**
```bash
docker inspect codellama-finetune
```

## Advanced Configuration

### Custom Build Arguments

```bash
# Build with proxy
HTTP_PROXY=http://proxy:8080 ./scripts/build_docker.sh

# Build with custom CUDA version
docker build --build-arg CUDA_VERSION=11.8 -t codellama-finetune:latest .
```

### Development Mode

```bash
# Create development override
./scripts/build_docker.sh --dev

# Start with source code mounted
docker-compose up -d
```

### Multi-GPU Setup

Edit `docker-compose.yml`:
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1,2,3  # Use specific GPUs
```

### Resource Limits

Edit `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 64G
    reservations:
      memory: 16G
```

## Production Deployment

### Build Production Image

```bash
# Build optimized production image
docker build -f Dockerfile.prod -t codellama-finetune:prod .
```

### Deploy with Docker Swarm

```bash
docker stack deploy -c docker-compose.prod.yml codellama-stack
```

### Health Monitoring

The container includes health checks:
```bash
docker inspect --format='{{.State.Health.Status}}' codellama-finetune
```

## Support

For issues with Docker setup:
1. Check the troubleshooting section above
2. Run the verification script: `python scripts/verify_installation.py`
3. Check container logs: `docker-compose logs codellama-finetune`
4. Test GPU access: `docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi`
