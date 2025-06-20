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
    
    docker build -t "$IMAGE_NAME" .
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
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
