#!/bin/bash

# Docker startup script for NeMo Fine-tuning Pipeline
# This script helps you start the NeMo container with the updated image

set -e

echo "🐳 Starting NeMo Fine-tuning Pipeline with Docker"
echo "================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if Docker is installed and running
check_docker() {
    print_step "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    print_status "Docker is installed and running ✓"
}

# Check if NVIDIA Docker runtime is available
check_nvidia_docker() {
    print_step "Checking NVIDIA Docker runtime..."
    
    if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
        print_warning "NVIDIA Docker runtime not available or no GPUs detected."
        print_warning "GPU acceleration will not be available."
        return 1
    fi
    
    print_status "NVIDIA Docker runtime available ✓"
    return 0
}

# Pull the latest NeMo image
pull_nemo_image() {
    print_step "Pulling NeMo container image..."
    
    NEMO_IMAGE="nvcr.io/nvidia/nemo:25.04.01.llama_nemotron_nano_vl"
    
    if docker pull "$NEMO_IMAGE"; then
        print_status "NeMo image pulled successfully ✓"
    else
        print_error "Failed to pull NeMo image. Check your internet connection."
        exit 1
    fi
}

# Start container with Docker Compose
start_with_compose() {
    print_step "Starting container with Docker Compose..."
    
    if [ -f "docker-compose.yml" ]; then
        docker-compose up -d nemo-finetuning
        print_status "Container started with Docker Compose ✓"
        print_status "Jupyter Lab available at: http://localhost:8888"
        print_status "To connect to container: docker-compose exec nemo-finetuning bash"
    else
        print_error "docker-compose.yml not found"
        return 1
    fi
}

# Start container directly
start_direct() {
    print_step "Starting container directly..."
    
    NEMO_IMAGE="nvcr.io/nvidia/nemo:25.04.01.llama_nemotron_nano_vl"
    
    # Check if GPU support is available
    if check_nvidia_docker; then
        GPU_ARGS="--gpus all"
    else
        GPU_ARGS=""
        print_warning "Running without GPU support"
    fi
    
    # Get current user ID and group ID for proper permissions
    USER_ID=$(id -u)
    GROUP_ID=$(id -g)

    docker run $GPU_ARGS \
        --shm-size=8g \
        --ulimit memlock=-1 \
        --rm -it \
        --user "$USER_ID:$GROUP_ID" \
        -v "$CURRENT_DIR:/workspace" \
        -w /workspace \
        -p 8888:8888 \
        "$NEMO_IMAGE" \
        bash -c "
            echo 'NeMo Fine-tuning Pipeline Container Started';
            echo 'Creating necessary directories...';
            mkdir -p outputs data models;
            echo 'Current directory contents:';
            ls -la;
            echo '';
            echo 'Checking for pipeline files:';
            ls -la *.py *.sh *.yaml *.yml 2>/dev/null || echo 'No pipeline files found';
            echo '';
            echo 'Available models:';
            python -c 'from configs.model_configs import list_available_models; print(list_available_models())' 2>/dev/null || echo 'Model configs not loaded yet - run: cd /workspace && python -c \"from configs.model_configs import list_available_models; print(list_available_models())\"';
            echo '';
            echo 'Starting interactive shell...';
            echo 'To start Jupyter Lab: jupyter lab --no-browser --port=8888 --allow-root --ip=0.0.0.0';
            echo 'To run quick start: ./quick_start.sh';
            bash
        "
}

# Show usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --compose    Use Docker Compose (default)"
    echo "  -d, --direct     Run container directly"
    echo "  -p, --pull       Pull latest image only"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                # Start with Docker Compose"
    echo "  $0 --direct       # Start container directly"
    echo "  $0 --pull         # Pull latest image only"
}

# Main function
main() {
    local mode="compose"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--compose)
                mode="compose"
                shift
                ;;
            -d|--direct)
                mode="direct"
                shift
                ;;
            -p|--pull)
                mode="pull"
                shift
                ;;
            -h|--help)
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
    
    # Check prerequisites
    check_docker
    
    # Execute based on mode
    case $mode in
        "pull")
            pull_nemo_image
            print_status "Image pull completed!"
            ;;
        "compose")
            pull_nemo_image
            if start_with_compose; then
                print_status "Container started successfully! 🎉"
            else
                print_warning "Docker Compose failed, trying direct method..."
                start_direct
            fi
            ;;
        "direct")
            pull_nemo_image
            start_direct
            ;;
    esac
}

# Run main function with all arguments
main "$@"
