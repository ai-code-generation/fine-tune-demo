#!/bin/bash

# Docker setup script for CodeLlama fine-tuning pipeline
# This script helps with Docker operations and provides common commands

set -e

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

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Function to check if Docker is installed and running
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    print_status "Docker is installed and running"
}

# Function to check if Docker Compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    
    print_status "Docker Compose is available"
}

# Function to check Docker GPU support (optional)
check_docker_gpu_support() {
    print_status "Checking Docker GPU support (optional)..."

    # Check if Docker supports --gpus flag
    if docker run --rm --gpus all ubuntu:22.04 nvidia-smi &> /dev/null 2>&1; then
        print_status "Docker GPU support detected - GPU training will be available"
        return 0
    else
        print_warning "Docker GPU support not detected - will use CPU mode"
        print_warning "For GPU support, ensure:"
        print_warning "  1. NVIDIA drivers are installed on host"
        print_warning "  2. Docker supports --gpus flag (Docker 19.03+)"
        print_warning "  3. nvidia-container-toolkit is installed"
        return 1
    fi
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."

    mkdir -p output
    mkdir -p cache
    mkdir -p logs
    mkdir -p data
    mkdir -p configs/model_configs
    mkdir -p configs/lora_configs

    # Set proper permissions for cache and logs directories
    chmod 755 cache logs output data

    print_status "Directories created successfully"
    print_status "Cache directory: ./cache (for model downloads)"
    print_status "Logs directory: ./logs (for training logs)"
    print_status "Output directory: ./output (for trained models)"
}

# Function to build the Docker image
build_image() {
    print_header "Building Docker image..."
    
    if command -v docker-compose &> /dev/null; then
        docker-compose build codellama-finetune
    else
        docker compose build codellama-finetune
    fi
    
    print_status "Docker image built successfully"
}

# Function to start the container
start_container() {
    print_header "Starting container..."
    
    if command -v docker-compose &> /dev/null; then
        docker-compose up -d codellama-finetune
    else
        docker compose up -d codellama-finetune
    fi
    
    print_status "Container started successfully"
    print_status "Use 'docker exec -it codellama-finetune bash' to access the container"
}

# Function to stop the container
stop_container() {
    print_header "Stopping container..."
    
    if command -v docker-compose &> /dev/null; then
        docker-compose down
    else
        docker compose down
    fi
    
    print_status "Container stopped successfully"
}

# Function to show container logs
show_logs() {
    print_header "Showing container logs..."
    
    if command -v docker-compose &> /dev/null; then
        docker-compose logs -f codellama-finetune
    else
        docker compose logs -f codellama-finetune
    fi
}

# Function to run training
run_training() {
    local model_config=${1:-"configs/model_configs/codellama_7b.yaml"}
    local train_data=${2:-"data/train.yaml"}
    local eval_data=${3:-"data/validation.yaml"}
    
    print_header "Running training with:"
    echo "  Model config: $model_config"
    echo "  Training data: $train_data"
    echo "  Evaluation data: $eval_data"
    
    docker exec -it codellama-finetune python train.py \
        --model-config "$model_config" \
        --train-data "$train_data" \
        --eval-data "$eval_data" \
        --validate-data
}

# Function to create sample data
create_sample_data() {
    print_header "Creating sample training data..."

    docker exec -it codellama-finetune python train.py --create-sample-data

    print_status "Sample data created in data/ directory"
}

# Function to merge LoRA model with base model
merge_model() {
    local lora_model=${1:-""}
    local output_path=${2:-""}

    if [ -z "$lora_model" ] || [ -z "$output_path" ]; then
        echo "Enter merge parameters:"
        read -p "LoRA model path: " lora_model
        read -p "Output path for merged model: " output_path
    fi

    if [ -z "$lora_model" ] || [ -z "$output_path" ]; then
        print_error "Both LoRA model path and output path are required"
        return 1
    fi

    print_header "Merging LoRA model with base model..."
    echo "  LoRA model: $lora_model"
    echo "  Output path: $output_path"

    docker exec -it codellama-finetune python scripts/merge_lora_model.py \
        --lora-model "$lora_model" \
        --output "$output_path"

    print_status "Model merge completed"
}



# Function to clean up Docker resources
cleanup() {
    print_header "Cleaning up Docker resources..."
    
    # Stop and remove containers
    if command -v docker-compose &> /dev/null; then
        docker-compose down --remove-orphans
    else
        docker compose down --remove-orphans
    fi
    
    # Remove unused images (optional)
    read -p "Remove unused Docker images? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker image prune -f
        print_status "Unused images removed"
    fi
    
    print_status "Cleanup completed"
}

# Function to show help
show_help() {
    print_header "CodeLlama Fine-tuning Pipeline - Docker Setup Script"
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  setup           - Complete setup (check dependencies, create dirs, build image)"
    echo "  build           - Build Docker image"
    echo "  start           - Start the container"
    echo "  stop            - Stop the container"
    echo "  logs            - Show container logs"
    echo "  shell           - Open shell in container"
    echo "  train           - Run training with default settings"
    echo "  train-custom    - Run training with custom parameters"
    echo "  sample-data     - Create sample training data"
    echo "  merge           - Merge LoRA model with base model"
    echo "  cleanup         - Clean up Docker resources"
    echo "  help            - Show this help message"
    echo
    echo "Examples:"
    echo "  $0 setup                    # Complete setup"
    echo "  $0 train                    # Run training with defaults"
    echo "  $0 shell                    # Access container shell"
}

# Main script logic
case "${1:-help}" in
    setup)
        print_header "Setting up CodeLlama fine-tuning pipeline..."
        check_docker
        check_docker_compose
        check_docker_gpu_support  # Optional check
        create_directories
        build_image
        print_status "Setup completed successfully!"
        echo
        print_status "The container includes:"
        echo "  ✅ CUDA 12.1 toolkit (self-contained)"
        echo "  ✅ All ML dependencies pre-installed"
        echo "  ✅ No host CUDA installation required"
        echo
        print_status "Next steps:"
        echo "  1. Run '$0 sample-data' to create sample training data"
        echo "  2. Run '$0 start' to start the container"
        echo "  3. Run '$0 train' to start training"
        ;;
    build)
        check_docker
        check_docker_compose
        build_image
        ;;
    start)
        check_docker
        create_directories
        start_container
        ;;
    stop)
        stop_container
        ;;
    logs)
        show_logs
        ;;
    shell)
        print_status "Opening shell in container..."
        docker exec -it codellama-finetune bash
        ;;
    train)
        print_status "Running training with default settings..."
        run_training
        ;;
    train-custom)
        echo "Enter training parameters (or press Enter for defaults):"
        read -p "Model config [configs/model_configs/codellama_7b.yaml]: " model_config
        read -p "Training data [data/train.yaml]: " train_data
        read -p "Evaluation data [data/validation.yaml]: " eval_data
        
        run_training "${model_config:-configs/model_configs/codellama_7b.yaml}" \
                    "${train_data:-data/train.yaml}" \
                    "${eval_data:-data/validation.yaml}"
        ;;
    sample-data)
        create_sample_data
        ;;
    merge)
        merge_model "$2" "$3"
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo
        show_help
        exit 1
        ;;
esac
