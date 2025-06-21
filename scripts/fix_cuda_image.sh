#!/bin/bash

# Quick fix script for CUDA Docker image issues
# This script will automatically fix the Dockerfile with a working CUDA base image

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "========================================"
echo "CUDA Docker Image Fix Script"
echo "========================================"
echo

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

print_status "Testing CUDA base images to find a working one..."

# List of CUDA images to test (in order of preference)
CUDA_IMAGES=(
    "nvidia/cuda:11.8-cudnn8-devel-ubuntu22.04"
    "nvcr.io/nvidia/cuda:11.8-devel-ubuntu22.04"
    "nvidia/cuda:11.8-devel-ubuntu20.04"
    "nvidia/cuda:12.0-devel-ubuntu22.04"
    "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel"
)

WORKING_IMAGE=""

for image in "${CUDA_IMAGES[@]}"; do
    print_status "Testing: $image"
    
    if docker manifest inspect "$image" &> /dev/null; then
        print_success "✓ Found working image: $image"
        WORKING_IMAGE="$image"
        break
    else
        print_warning "✗ Not available: $image"
    fi
done

if [ -z "$WORKING_IMAGE" ]; then
    print_error "No working CUDA images found!"
    print_error "Creating a fallback Dockerfile with Ubuntu base + manual CUDA installation..."
    
    # Create fallback Dockerfile
    cat > Dockerfile.fixed << 'EOF'
# Fallback Dockerfile with Ubuntu base + manual CUDA installation
FROM ubuntu:22.04 as base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git wget curl build-essential cmake \
    libsndfile1 ffmpeg sox libsox-fmt-all software-properties-common gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA manually
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get -y install cuda-toolkit-11-8 && \
    rm cuda-keyring_1.0-1_all.deb && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python -m pip install --upgrade pip
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install nemo_toolkit[all]==1.20.0
RUN pip install transformers pytorch-lightning omegaconf hydra-core wandb tensorboard

WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt || true
COPY . .
RUN pip install -e . || true

RUN mkdir -p /workspace/data /workspace/logs /workspace/checkpoints /workspace/deploy /workspace/outputs
RUN useradd -m -u 1000 nemo_user && chown -R nemo_user:nemo_user /workspace
USER nemo_user

ENV PYTHONPATH=/workspace/src:$PYTHONPATH
RUN mkdir -p /workspace/.cache/nemo /workspace/.cache/huggingface /workspace/.cache/wandb

EXPOSE 8888 6006
CMD ["/bin/bash"]
EOF

    print_status "Created Dockerfile.fixed with Ubuntu base + manual CUDA installation"
    print_status "To use this, run: docker build -f Dockerfile.fixed -t nemo-finetune-pipeline:latest ."
    
else
    print_success "Found working CUDA image: $WORKING_IMAGE"
    print_status "Updating Dockerfile with working image..."
    
    # Backup original Dockerfile
    cp Dockerfile Dockerfile.backup
    
    # Update Dockerfile with working image
    sed -i "s|FROM.*as base|FROM $WORKING_IMAGE as base|" Dockerfile
    
    print_success "Updated Dockerfile with working base image: $WORKING_IMAGE"
    print_status "Original Dockerfile backed up as Dockerfile.backup"
fi

echo
print_status "Next steps:"
echo "1. Try building again:"
echo "   ./scripts/docker_run.sh build"
echo
echo "2. Or build manually:"
if [ -n "$WORKING_IMAGE" ]; then
    echo "   docker build -t nemo-finetune-pipeline:latest ."
else
    echo "   docker build -f Dockerfile.fixed -t nemo-finetune-pipeline:latest ."
fi
echo
echo "3. If still having issues, try without GPU support first:"
echo "   docker build --build-arg CUDA_VERSION=none -t nemo-finetune-pipeline:cpu ."

print_success "CUDA image fix completed!"
