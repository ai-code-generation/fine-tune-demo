#!/bin/bash

# Docker run script for NeMo Fine-Tuning Pipeline
# This script sets up and runs the fine-tuning pipeline in a Docker container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODEL_TYPE="llama2"
MODEL_SIZE="7b"
MAX_EPOCHS=3
GPUS=1
BATCH_SIZE=2
CONTAINER_NAME="nemo-finetune-s32k144"
IMAGE_NAME="nemo-finetune-pipeline:latest"

# Function to print colored output
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

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Docker runner for NeMo Fine-Tuning Pipeline

COMMANDS:
    build           Build the Docker image
    run             Run the fine-tuning pipeline
    shell           Start interactive shell in container
    jupyter         Start Jupyter Lab server
    tensorboard     Start TensorBoard server
    stop            Stop and remove containers
    clean           Clean up containers and images

OPTIONS:
    --model-type    Model type (llama2, llama3, codellama) [default: llama2]
    --model-size    Model size (7b, 8b, 13b, etc.) [default: 7b]
    --epochs        Number of training epochs [default: 3]
    --gpus          Number of GPUs to use [default: 1]
    --batch-size    Batch size for training [default: 2]
    --container     Container name [default: nemo-finetune-s32k144]
    --image         Docker image name [default: nemo-finetune-pipeline:latest]
    --help          Show this help message

EXAMPLES:
    # Build the Docker image
    $0 build

    # Run S32K144 fine-tuning with default settings
    $0 run

    # Run with custom parameters
    $0 --model-type llama3 --model-size 8b --epochs 5 --gpus 2 run

    # Start interactive shell
    $0 shell

    # Start Jupyter Lab
    $0 jupyter

    # Clean up everything
    $0 clean

VOLUME MOUNTS:
    ./data              -> /workspace/data (training data, read-only)
    ./outputs/checkpoints -> /workspace/checkpoints (model checkpoints)
    ./outputs/logs      -> /workspace/logs (training logs)
    ./outputs/deploy    -> /workspace/deploy (deployed models)
    ./outputs/cache     -> /workspace/.cache (model cache)

EOF
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if NVIDIA Docker is available (for GPU support)
    if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        print_warning "NVIDIA Docker runtime not available. GPU acceleration will not work."
        print_warning "Install nvidia-docker2 for GPU support."
    else
        print_success "NVIDIA Docker runtime is available"
    fi
    
    # Check if required directories exist
    if [ ! -d "data" ]; then
        print_error "Data directory not found. Please ensure training data is in ./data/"
        exit 1
    fi
    
    # Create output directories
    mkdir -p outputs/{checkpoints,logs,deploy,cache}
    print_success "Output directories created"
}

# Function to build Docker image
build_image() {
    print_status "Building Docker image: $IMAGE_NAME"

    # First, try to build with the main Dockerfile
    print_status "Attempting to build with main Dockerfile..."
    if docker build -t "$IMAGE_NAME" . 2>/dev/null; then
        print_success "Docker image built successfully with main Dockerfile"
        return 0
    fi

    print_warning "Main Dockerfile failed. Checking CUDA image availability..."

    # Check if the CUDA image checker script exists and run it
    if [ -f "scripts/check_cuda_images.sh" ]; then
        chmod +x scripts/check_cuda_images.sh
        print_status "Running CUDA image checker..."
        ./scripts/check_cuda_images.sh
    fi

    # Try alternative Dockerfile if it exists
    if [ -f "Dockerfile.alternative" ]; then
        print_status "Attempting to build with alternative Dockerfile..."
        if docker build -f Dockerfile.alternative -t "$IMAGE_NAME" .; then
            print_success "Docker image built successfully with alternative Dockerfile"
            return 0
        fi
    fi

    # Try with NVIDIA Container Registry
    print_status "Attempting to build with NVIDIA Container Registry base image..."

    # Create temporary Dockerfile with nvcr.io base
    cat > Dockerfile.nvcr << EOF
FROM nvcr.io/nvidia/cuda:11.8-devel-ubuntu22.04 as base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=\${CUDA_HOME}/bin:\${PATH}
ENV LD_LIBRARY_PATH=\${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y \\
    python3 python3-pip python3-dev git wget curl build-essential cmake \\
    libsndfile1 ffmpeg sox libsox-fmt-all && rm -rf /var/lib/apt/lists/*

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

ENV PYTHONPATH=/workspace/src:\$PYTHONPATH
RUN mkdir -p /workspace/.cache/nemo /workspace/.cache/huggingface /workspace/.cache/wandb

EXPOSE 8888 6006
CMD ["/bin/bash"]
EOF

    if docker build -f Dockerfile.nvcr -t "$IMAGE_NAME" .; then
        print_success "Docker image built successfully with NVIDIA Container Registry"
        rm -f Dockerfile.nvcr
        return 0
    fi

    rm -f Dockerfile.nvcr

    print_error "All Docker build attempts failed!"
    print_error "Please check:"
    print_error "1. Internet connectivity"
    print_error "2. Docker Hub access"
    print_error "3. NVIDIA Docker installation"
    print_error ""
    print_error "Try running: ./scripts/check_cuda_images.sh"
    exit 1
}

# Function to run fine-tuning
run_finetune() {
    print_status "Starting S32K144 fine-tuning pipeline..."
    print_status "Model: $MODEL_TYPE $MODEL_SIZE"
    print_status "Epochs: $MAX_EPOCHS, GPUs: $GPUS, Batch Size: $BATCH_SIZE"
    
    # Stop existing container if running
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    
    # Run the fine-tuning pipeline
    docker run -d \
        --name "$CONTAINER_NAME" \
        --gpus all \
        -v "$(pwd)/data:/workspace/data:ro" \
        -v "$(pwd)/outputs/checkpoints:/workspace/checkpoints" \
        -v "$(pwd)/outputs/logs:/workspace/logs" \
        -v "$(pwd)/outputs/deploy:/workspace/deploy" \
        -v "$(pwd)/outputs/cache:/workspace/.cache" \
        -v "$(pwd)/configs:/workspace/configs:ro" \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e CUDA_VISIBLE_DEVICES=all \
        -e PYTHONPATH=/workspace/src \
        -e WANDB_PROJECT=nemo-s32k144-finetune \
        "$IMAGE_NAME" \
        python scripts/run_pipeline.py \
            --model-type "$MODEL_TYPE" \
            --model-size "$MODEL_SIZE" \
            --train-file data/train.yaml \
            --val-file data/val.yaml \
            --test-file data/test.yaml \
            --max-epochs "$MAX_EPOCHS" \
            --gpus "$GPUS"
    
    if [ $? -eq 0 ]; then
        print_success "Fine-tuning started in container: $CONTAINER_NAME"
        print_status "Monitor progress with: docker logs -f $CONTAINER_NAME"
        print_status "Or use: $0 logs"
    else
        print_error "Failed to start fine-tuning"
        exit 1
    fi
}

# Function to start interactive shell
start_shell() {
    print_status "Starting interactive shell in container..."
    
    docker run -it --rm \
        --name "${CONTAINER_NAME}-shell" \
        --gpus all \
        -v "$(pwd)/data:/workspace/data:ro" \
        -v "$(pwd)/outputs/checkpoints:/workspace/checkpoints" \
        -v "$(pwd)/outputs/logs:/workspace/logs" \
        -v "$(pwd)/outputs/deploy:/workspace/deploy" \
        -v "$(pwd)/outputs/cache:/workspace/.cache" \
        -v "$(pwd)/configs:/workspace/configs:ro" \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e CUDA_VISIBLE_DEVICES=all \
        -e PYTHONPATH=/workspace/src \
        "$IMAGE_NAME" \
        /bin/bash
}

# Function to start Jupyter Lab
start_jupyter() {
    print_status "Starting Jupyter Lab server..."
    
    # Stop existing Jupyter container if running
    docker stop "${CONTAINER_NAME}-jupyter" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}-jupyter" 2>/dev/null || true
    
    docker run -d \
        --name "${CONTAINER_NAME}-jupyter" \
        --gpus all \
        -p 8888:8888 \
        -v "$(pwd)/data:/workspace/data:ro" \
        -v "$(pwd)/outputs:/workspace/outputs" \
        -v "$(pwd)/configs:/workspace/configs:ro" \
        -v "$(pwd)/notebooks:/workspace/notebooks" \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e CUDA_VISIBLE_DEVICES=all \
        -e PYTHONPATH=/workspace/src \
        "$IMAGE_NAME" \
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
            --NotebookApp.token='' --NotebookApp.password=''
    
    if [ $? -eq 0 ]; then
        print_success "Jupyter Lab started at: http://localhost:8888"
        print_status "Container name: ${CONTAINER_NAME}-jupyter"
    else
        print_error "Failed to start Jupyter Lab"
        exit 1
    fi
}

# Function to start TensorBoard
start_tensorboard() {
    print_status "Starting TensorBoard server..."
    
    # Stop existing TensorBoard container if running
    docker stop "${CONTAINER_NAME}-tensorboard" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}-tensorboard" 2>/dev/null || true
    
    docker run -d \
        --name "${CONTAINER_NAME}-tensorboard" \
        -p 6006:6006 \
        -v "$(pwd)/outputs/logs:/logs:ro" \
        tensorflow/tensorflow:latest \
        tensorboard --logdir=/logs --host=0.0.0.0 --port=6006
    
    if [ $? -eq 0 ]; then
        print_success "TensorBoard started at: http://localhost:6006"
        print_status "Container name: ${CONTAINER_NAME}-tensorboard"
    else
        print_error "Failed to start TensorBoard"
        exit 1
    fi
}

# Function to show logs
show_logs() {
    if docker ps | grep -q "$CONTAINER_NAME"; then
        print_status "Showing logs for container: $CONTAINER_NAME"
        docker logs -f "$CONTAINER_NAME"
    else
        print_error "Container $CONTAINER_NAME is not running"
        exit 1
    fi
}

# Function to stop containers
stop_containers() {
    print_status "Stopping containers..."
    
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker stop "${CONTAINER_NAME}-jupyter" 2>/dev/null || true
    docker stop "${CONTAINER_NAME}-tensorboard" 2>/dev/null || true
    
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}-jupyter" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}-tensorboard" 2>/dev/null || true
    
    print_success "Containers stopped and removed"
}

# Function to clean up
cleanup() {
    print_status "Cleaning up containers and images..."
    
    stop_containers
    
    # Remove image
    docker rmi "$IMAGE_NAME" 2>/dev/null || true
    
    # Clean up dangling images
    docker image prune -f
    
    print_success "Cleanup completed"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --model-size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --container)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        build|run|shell|jupyter|tensorboard|logs|stop|clean)
            COMMAND="$1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if command is provided
if [ -z "$COMMAND" ]; then
    print_error "No command provided"
    show_usage
    exit 1
fi

# Execute command
case $COMMAND in
    build)
        check_prerequisites
        build_image
        ;;
    run)
        check_prerequisites
        build_image
        run_finetune
        ;;
    shell)
        check_prerequisites
        start_shell
        ;;
    jupyter)
        check_prerequisites
        start_jupyter
        ;;
    tensorboard)
        check_prerequisites
        start_tensorboard
        ;;
    logs)
        show_logs
        ;;
    stop)
        stop_containers
        ;;
    clean)
        cleanup
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac
