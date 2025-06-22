#!/bin/bash

# Docker Start Script for NeMo 24.07
# Optimized for CodeLlama-13B and Llama3 models (8B-70B)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

print_status() {
    echo -e "${GREEN}[✓] $1${NC}"
}

# Check if Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
    
    print_status "Docker is available"
}

# Check for NVIDIA Docker support
check_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
        print_error "NVIDIA Docker support not available"
        print_error "Make sure nvidia-docker2 is installed and configured"
        exit 1
    fi
    
    print_status "NVIDIA Docker support available"
}

# Check if HF token is available
check_hf_token() {
    if [ -f ".env" ] && grep -q "HF_TOKEN=" .env; then
        print_status "HF_TOKEN found in .env file"
        return 0
    fi
    
    if [ -n "$HF_TOKEN" ]; then
        print_status "HF_TOKEN found in environment"
        return 0
    fi
    
    print_warning "No HF_TOKEN found"
    print_info "You'll need to set it up inside the container"
    print_info "Run: python setup_hf_auth.py inside the container"
}

# Start the container
start_container() {
    print_info "Starting NeMo 24.07 container..."
    
    # Container configuration
    CONTAINER_NAME="nemo-finetuning-24.07"
    IMAGE="nvcr.io/nvidia/nemo:24.07"
    HOST_PATH="/static-data/team_08/simple-nemo/fine-tune-demo"
    CONTAINER_PATH="/workspace"
    
    # Check if container already exists
    if docker ps -a --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        print_info "Container $CONTAINER_NAME already exists"
        
        if docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
            print_info "Container is already running"
            print_info "Connecting to existing container..."
            docker exec -it "$CONTAINER_NAME" /bin/bash
            return 0
        else
            print_info "Starting existing container..."
            docker start "$CONTAINER_NAME"
            docker exec -it "$CONTAINER_NAME" /bin/bash
            return 0
        fi
    fi
    
    # Create and start new container
    print_info "Creating new container: $CONTAINER_NAME"
    print_info "Image: $IMAGE"
    print_info "Host path: $HOST_PATH"
    print_info "Container path: $CONTAINER_PATH"
    
    # Build docker run command
    DOCKER_CMD="docker run -it --rm"
    DOCKER_CMD="$DOCKER_CMD --name $CONTAINER_NAME"
    DOCKER_CMD="$DOCKER_CMD --gpus all"
    DOCKER_CMD="$DOCKER_CMD --shm-size=16g"
    DOCKER_CMD="$DOCKER_CMD --ulimit memlock=-1"
    DOCKER_CMD="$DOCKER_CMD --ulimit stack=67108864"
    
    # Environment variables
    DOCKER_CMD="$DOCKER_CMD -e CUDA_VISIBLE_DEVICES=all"
    DOCKER_CMD="$DOCKER_CMD -e NCCL_DEBUG=INFO"
    DOCKER_CMD="$DOCKER_CMD -e PYTHONPATH=/workspace"
    DOCKER_CMD="$DOCKER_CMD -e TRANSFORMERS_CACHE=/workspace/.cache/transformers"
    DOCKER_CMD="$DOCKER_CMD -e HF_HOME=/workspace/.cache/huggingface"
    
    # Add HF_TOKEN if available
    if [ -f ".env" ] && grep -q "HF_TOKEN=" .env; then
        HF_TOKEN=$(grep "HF_TOKEN=" .env | cut -d'=' -f2)
        DOCKER_CMD="$DOCKER_CMD -e HF_TOKEN=$HF_TOKEN"
    elif [ -n "$HF_TOKEN" ]; then
        DOCKER_CMD="$DOCKER_CMD -e HF_TOKEN=$HF_TOKEN"
    fi
    
    # Volume mounts
    DOCKER_CMD="$DOCKER_CMD -v $HOST_PATH:$CONTAINER_PATH"
    DOCKER_CMD="$DOCKER_CMD -v $HOST_PATH/outputs:$CONTAINER_PATH/outputs"
    DOCKER_CMD="$DOCKER_CMD -v $HOST_PATH/.cache:$CONTAINER_PATH/.cache"
    
    # Port mappings
    DOCKER_CMD="$DOCKER_CMD -p 8888:8888"  # Jupyter
    DOCKER_CMD="$DOCKER_CMD -p 6006:6006"  # TensorBoard
    DOCKER_CMD="$DOCKER_CMD -p 7007:7007"  # Additional services
    
    # Working directory
    DOCKER_CMD="$DOCKER_CMD -w $CONTAINER_PATH"
    
    # Image and command
    DOCKER_CMD="$DOCKER_CMD $IMAGE"
    DOCKER_CMD="$DOCKER_CMD bash -c \""
    DOCKER_CMD="$DOCKER_CMD echo 'NeMo 24.07 Fine-tuning Environment Ready';"
    DOCKER_CMD="$DOCKER_CMD echo 'Supported: CodeLlama-13B, Llama3-8B/70B';"
    DOCKER_CMD="$DOCKER_CMD echo 'Host mount: $HOST_PATH';"
    DOCKER_CMD="$DOCKER_CMD echo 'Quick start: ./quick_start_nemo24.sh';"
    DOCKER_CMD="$DOCKER_CMD echo 'Pipeline: python finetune_pipeline_nemo24.py --help';"
    DOCKER_CMD="$DOCKER_CMD /bin/bash\""
    
    print_info "Running: $DOCKER_CMD"
    eval $DOCKER_CMD
}

# Alternative: Use docker-compose
use_docker_compose() {
    print_info "Using docker-compose for NeMo 24.07..."
    
    if [ ! -f "docker-compose.yml" ]; then
        print_error "docker-compose.yml not found"
        exit 1
    fi
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        print_warning "No .env file found"
        print_info "Creating .env file template..."
        echo "HF_TOKEN=your_huggingface_token_here" > .env
        print_info "Please edit .env file and add your HuggingFace token"
        return 1
    fi
    
    print_info "Starting with docker-compose..."
    docker-compose up nemo-finetuning
}

# Show usage
show_usage() {
    echo "NeMo 24.07 Docker Start Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --compose, -c    Use docker-compose instead of direct docker run"
    echo "  --help, -h       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0               Start container with docker run"
    echo "  $0 --compose     Start container with docker-compose"
    echo ""
    echo "Requirements:"
    echo "  - Docker with NVIDIA GPU support"
    echo "  - Host path: /static-data/team_08/simple-nemo/fine-tune-demo"
    echo "  - HuggingFace token (for gated models)"
}

# Main function
main() {
    # Parse arguments
    USE_COMPOSE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --compose|-c)
                USE_COMPOSE=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    print_info "NeMo 24.07 Docker Environment Setup"
    print_info "Host mount: /static-data/team_08/simple-nemo/fine-tune-demo"
    echo ""
    
    # Check prerequisites
    check_docker
    check_nvidia_docker
    check_hf_token
    
    echo ""
    
    # Start container
    if [ "$USE_COMPOSE" = true ]; then
        use_docker_compose
    else
        start_container
    fi
}

# Run main function
main "$@"
