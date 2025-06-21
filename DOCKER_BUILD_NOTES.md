# Docker Build Notes

## Overview

This document explains the Docker build process and how to handle optional dependencies that may cause build issues.

## Build Process

### Using the Main Dockerfile

The main `Dockerfile` has been updated with the robust build process:

```bash
docker build -t fine-tune-pipeline .
```

### Key Features

1. **Self-contained**: All dependencies are installed within the container
2. **CUDA Support**: Includes CUDA 11.8 toolkit for GPU acceleration
3. **Build Tools**: Includes all necessary compilation tools (gcc, g++, cmake, ninja)
4. **Robust Dependencies**: Uses a curated list of dependencies that build reliably
5. **Updated PyTorch**: Uses PyTorch 2.1.0 for better compatibility
6. **Import Fix**: Resolved relative import issues in the pipeline scripts

## Dependency Management

### Core Dependencies

The following dependencies are installed automatically:
- NeMo Toolkit with NLP extensions
- PyTorch 2.0.1 with CUDA 11.8 support
- Transformers, datasets, and other ML libraries
- Development tools (pytest, black, flake8, mypy)

### Optional Dependencies

#### APEX
- **Status**: ⚠️ Optional (excluded from automatic build)
- **Purpose**: Mixed precision training
- **Issue**: Build backend compatibility issues with `enscons`
- **Solution**: Can be installed manually after container is running

#### Flash Attention
- **Status**: ⚠️ Optional (excluded from automatic build)
- **Purpose**: Memory-efficient attention computation
- **Issue**: Compilation conflicts with PyTorch 2.0.1 and CUDA 11.8
- **Solution**: Can be installed manually after container is running

### Installing Optional Dependencies

You can install optional dependencies inside the running container:

```bash
# Method 1: Install all optional dependencies
./scripts/install_optional_deps.sh

# Method 2: Install individually
./scripts/install_apex.sh
./scripts/install_flash_attn.sh

# Method 3: Manual installation
pip install apex --no-build-isolation
pip install flash-attn>=2.0.0 --no-build-isolation --verbose
```

**Requirements for optional dependencies**:
- At least 6-8GB of RAM
- 10-15 minutes of compilation time
- May fail on some systems due to CUDA/PyTorch version compatibility

## Troubleshooting

### Build Failures

1. **Out of Memory**: Increase Docker memory allocation to at least 8GB
2. **CUDA Issues**: Ensure NVIDIA Docker runtime is installed
3. **Compilation Errors**: Check that all build tools are available

### Runtime Issues

1. **GPU Not Available**: Verify NVIDIA Docker runtime and GPU drivers
2. **Permission Errors**: The container runs as `nemo_user` (UID 1000)
3. **Cache Issues**: Clear Docker build cache with `docker system prune`

## File Structure

```
requirements.txt                    # Full requirements (includes problematic packages)
requirements-docker.txt            # Docker-compatible requirements (curated)
scripts/install_apex.sh            # Optional NVIDIA Apex installer
scripts/install_flash_attn.sh      # Optional flash attention installer
scripts/install_optional_deps.sh   # Install all optional dependencies
scripts/test_imports.py            # Test script to verify imports work
scripts/run_pipeline.py            # Main pipeline script
DOCKER_BUILD_NOTES.md              # This documentation file
```

## Usage Instructions

### Quick Start

```bash
# Build the Docker image
docker build -t fine-tune-pipeline .

# Run the container
docker run --gpus all -it fine-tune-pipeline

# Test that everything works
./scripts/test_imports.py

# Optional: Install performance optimizations
./scripts/install_optional_deps.sh
```

### Running the Pipeline

```bash
# Example: Fine-tune LLaMA2 7B model
python scripts/run_pipeline.py \
    --model-type llama2 \
    --model-size 7b \
    --train-file data/train.yaml \
    --val-file data/val.yaml \
    --test-file data/test.yaml \
    --max-epochs 3 \
    --gpus 1

# Training only
python scripts/run_pipeline.py \
    --model-type llama2 \
    --model-size 7b \
    --train-file data/train.yaml \
    --training-only \
    --max-epochs 3
```

## Best Practices

1. Use `requirements-docker.txt` for Docker builds
2. Keep `requirements.txt` for local development
3. Install optional dependencies manually when needed
4. Monitor memory usage during builds
5. Use multi-stage builds for production deployments
6. Test imports before running the pipeline
