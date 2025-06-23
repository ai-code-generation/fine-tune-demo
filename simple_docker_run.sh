#!/bin/bash

# Simple Docker Run Script for NeMo 24.07 Fine-tuning
# Includes --user flag for proper file permissions

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${GREEN}[STEP] $1${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# Configuration
IMAGE="nvcr.io/nvidia/nemo:24.07"
CONTAINER_NAME="nemo-finetuning"
HOST_PATH="/static-data/team_08/simple-nemo/fine-tune-demo"
CONTAINER_PATH="/workspace"

print_step "Simple NeMo 24.07 Docker Setup"
echo ""

# Step 1: Check prerequisites
print_step "1. Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    exit 1
fi

if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running"
    exit 1
fi

if ! nvidia-smi &> /dev/null; then
    print_error "NVIDIA drivers not found"
    exit 1
fi

print_info "✓ Docker and NVIDIA drivers are available"

# Step 2: Set up environment
print_step "2. Setting up environment..."

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_info "Creating .env file..."
    cat > .env << EOF
HF_TOKEN=your_huggingface_token_here
USER_ID=$(id -u)
GROUP_ID=$(id -g)
EOF
    print_warning "Please edit .env file and add your HuggingFace token"
    print_info "Get token from: https://huggingface.co/settings/tokens"
    read -p "Press Enter after setting up your HF_TOKEN in .env file..."
fi

# Load environment variables
source .env

if [ "$HF_TOKEN" = "your_huggingface_token_here" ]; then
    print_error "Please set your HuggingFace token in .env file"
    exit 1
fi

print_info "✓ Environment configured"

# Step 3: Pull image
print_step "3. Pulling NeMo 24.07 image..."
docker pull "$IMAGE"
print_info "✓ Image pulled successfully"

# Step 4: Create directories
print_step "4. Creating output directories..."
mkdir -p outputs/{models,data,experiments,logs}
mkdir -p .cache/{transformers,huggingface}
print_info "✓ Directories created"

# Step 5: Run container
print_step "5. Starting Docker container with --user flag..."

print_info "Container configuration:"
print_info "  Image: $IMAGE"
print_info "  Host path: $HOST_PATH"
print_info "  Container path: $CONTAINER_PATH"
print_info "  User: $(id -u):$(id -g)"
print_info "  GPUs: all"

# Stop existing container if running
if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
    print_info "Stopping existing container..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

# Remove existing container if exists
if docker ps -aq -f name="$CONTAINER_NAME" | grep -q .; then
    print_info "Removing existing container..."
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

# Run new container
print_info "Starting new container..."

docker run -it --rm \
  --name "$CONTAINER_NAME" \
  --user "$(id -u):$(id -g)" \
  --gpus all \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$HOST_PATH:$CONTAINER_PATH" \
  -v "$HOST_PATH/outputs:$CONTAINER_PATH/outputs" \
  -v "$HOST_PATH/.cache:$CONTAINER_PATH/.cache" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e CUDA_VISIBLE_DEVICES=all \
  -e PYTHONPATH="$CONTAINER_PATH" \
  -e TRANSFORMERS_CACHE="$CONTAINER_PATH/.cache/transformers" \
  -e HF_HOME="$CONTAINER_PATH/.cache/huggingface" \
  -e HF_DATASETS_CACHE="$CONTAINER_PATH/.cache/datasets" \
  -e TORCH_HOME="$CONTAINER_PATH/.cache/torch" \
  -e XDG_CACHE_HOME="$CONTAINER_PATH/.cache" \
  -w "$CONTAINER_PATH" \
  -p 8888:8888 \
  -p 6006:6006 \
  "$IMAGE" \
  bash -c "
    echo '🚀 NeMo 24.07 Fine-tuning Environment Ready!';
    echo '';
    echo 'Configuration:';
    echo '  Host mount: $HOST_PATH';
    echo '  User: $(id -u):$(id -g)';
    echo '  GPUs: Available';
    echo '';
    echo 'Setting up cache directories...';
    mkdir -p $CONTAINER_PATH/.cache/{transformers,huggingface,datasets,torch};
    chown -R $(id -u):$(id -g) $CONTAINER_PATH/.cache 2>/dev/null || true;
    echo 'Cache directories ready ✓';
    echo '';
    echo 'Available commands:';
    echo '  python simple_nemo_finetune_7b.py --help   # CodeLlama-7B help';
    echo '  python simple_nemo_finetune.py --help      # CodeLlama-13B help';
    echo '  python debug_jsonl_files.py                # Debug JSONL files';
    echo '';
    echo 'Available models:';
    echo '  - CodeLlama-7B:  2 GPUs, 16GB+ VRAM each (Resource Efficient)';
    echo '  - CodeLlama-13B: 4 GPUs, 24GB+ VRAM each (Higher Quality)';
    echo '';
    echo 'Example training:';
    echo '  # For CodeLlama-7B (Resource Efficient):';
    echo '  python simple_nemo_finetune_7b.py --data example_training_data.yaml --hf-token \$HF_TOKEN';
    echo '';
    echo '  # For CodeLlama-13B (Higher Quality):';
    echo '  python simple_nemo_finetune.py --data example_training_data.yaml --hf-token \$HF_TOKEN';
    echo '';
    echo 'Starting interactive shell...';
    /bin/bash
  "

print_info "Container exited. Goodbye!"
