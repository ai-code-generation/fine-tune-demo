#!/bin/bash

# Docker build script for CodeLlama fine-tuning pipeline
# This script builds the Docker image with all requirements and CUDA support

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

# Function to check NVIDIA Docker support
check_nvidia_docker() {
    print_header "Checking NVIDIA Docker support..."
    
    # Check if nvidia-docker2 or Docker with GPU support is available
    if command -v nvidia-docker &> /dev/null; then
        print_status "nvidia-docker found"
        return 0
    fi
    
    # Check if Docker supports --gpus flag
    if docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_status "Docker GPU support detected"
        return 0
    fi
    
    print_warning "NVIDIA Docker support not detected. GPU training may not work."
    print_warning "Please install nvidia-docker2 or ensure Docker supports --gpus flag"
    return 1
}

# Function to check available GPU memory
check_gpu_memory() {
    print_header "Checking GPU memory..."
    
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
        
        # Check if we have enough memory (at least 8GB recommended)
        total_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        if [ "$total_memory" -lt 8192 ]; then
            print_warning "GPU has less than 8GB memory. Consider using smaller models or batch sizes."
        else
            print_status "GPU memory looks sufficient for training"
        fi
    else
        print_warning "nvidia-smi not found. Cannot check GPU memory."
    fi
}

# Function to build the Docker image
build_image() {
    print_header "Building Docker image..."
    
    # Build arguments
    BUILD_ARGS=""
    
    # Add build arguments if needed
    if [ ! -z "$HTTP_PROXY" ]; then
        BUILD_ARGS="$BUILD_ARGS --build-arg HTTP_PROXY=$HTTP_PROXY"
    fi
    
    if [ ! -z "$HTTPS_PROXY" ]; then
        BUILD_ARGS="$BUILD_ARGS --build-arg HTTPS_PROXY=$HTTPS_PROXY"
    fi
    
    # Build the image
    echo "Building image with command:"
    echo "docker build $BUILD_ARGS -t codellama-finetune:latest ."
    
    if docker build $BUILD_ARGS -t codellama-finetune:latest .; then
        print_status "Docker image built successfully"
        return 0
    else
        print_error "Docker image build failed"
        return 1
    fi
}

# Function to test the built image
test_image() {
    print_header "Testing the built image..."
    
    # Test basic functionality
    print_status "Testing basic Python imports..."
    if docker run --rm codellama-finetune:latest python -c "import torch, transformers, peft; print('✅ All imports successful')"; then
        print_status "Basic imports test passed"
    else
        print_error "Basic imports test failed"
        return 1
    fi
    
    # Test CUDA availability (if GPU is available)
    print_status "Testing CUDA availability..."
    if docker run --rm --gpus all codellama-finetune:latest python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"; then
        print_status "CUDA test completed"
    else
        print_warning "CUDA test failed or no GPU available"
    fi
    
    # Test pipeline imports
    print_status "Testing pipeline imports..."
    if docker run --rm codellama-finetune:latest python -c "from src import ConversationDataHandler, ModelSetup, CodeLlamaTrainer; print('✅ Pipeline imports successful')"; then
        print_status "Pipeline imports test passed"
    else
        print_error "Pipeline imports test failed"
        return 1
    fi
    
    print_status "All tests passed!"
    return 0
}

# Function to show image information
show_image_info() {
    print_header "Docker image information:"
    
    # Show image size
    image_size=$(docker images codellama-finetune:latest --format "table {{.Size}}" | tail -1)
    echo "Image size: $image_size"
    
    # Show image layers
    echo ""
    echo "Image layers:"
    docker history codellama-finetune:latest --format "table {{.CreatedBy}}\t{{.Size}}" | head -10
    
    # Show installed packages
    echo ""
    echo "Key installed packages:"
    docker run --rm codellama-finetune:latest python -c "
import torch, transformers, peft, datasets
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'Datasets: {datasets.__version__}')
"
}

# Function to create docker-compose override for development
create_dev_override() {
    print_header "Creating development docker-compose override..."
    
    cat > docker-compose.override.yml << EOF
version: '3.8'

services:
  codellama-finetune:
    # Override for development
    volumes:
      # Mount source code for live editing
      - ./src:/workspace/src
      - ./tests:/workspace/tests
      
    # Enable development mode
    environment:
      - PYTHONPATH=/workspace/src
      - DEVELOPMENT=true
      
    # Override command for development
    command: >
      bash -c "
        echo '🔧 Development mode enabled';
        echo 'Source code is mounted for live editing';
        echo 'Run tests with: python scripts/run_tests.py';
        tail -f /dev/null
      "
EOF
    
    print_status "Development override created: docker-compose.override.yml"
}

# Function to show usage help
show_help() {
    print_header "CodeLlama Fine-tuning Pipeline - Docker Build Script"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --build-only     Build image without testing"
    echo "  --test-only      Test existing image without building"
    echo "  --dev            Create development docker-compose override"
    echo "  --no-cache       Build without using Docker cache"
    echo "  --help           Show this help message"
    echo
    echo "Examples:"
    echo "  $0                    # Full build and test"
    echo "  $0 --build-only       # Build only"
    echo "  $0 --test-only        # Test existing image"
    echo "  $0 --dev              # Setup for development"
    echo "  $0 --no-cache         # Clean build"
}

# Parse command line arguments
BUILD_ONLY=false
TEST_ONLY=false
DEV_MODE=false
NO_CACHE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --test-only)
            TEST_ONLY=true
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
print_header "CodeLlama Fine-tuning Pipeline - Docker Build"
echo "=============================================="

# Check prerequisites
check_docker
check_nvidia_docker
check_gpu_memory

# Handle different modes
if [ "$DEV_MODE" = true ]; then
    create_dev_override
    exit 0
fi

if [ "$TEST_ONLY" = true ]; then
    test_image
    show_image_info
    exit 0
fi

# Build the image
if [ "$NO_CACHE" = true ]; then
    print_status "Building with --no-cache flag"
    docker build --no-cache -t codellama-finetune:latest .
else
    build_image
fi

# Test the image unless build-only is specified
if [ "$BUILD_ONLY" = false ]; then
    test_image
    show_image_info
fi

print_header "Build completed successfully!"
echo
print_status "Next steps:"
echo "1. Start the container: docker-compose up -d"
echo "2. Access the container: docker exec -it codellama-finetune bash"
echo "3. Create sample data: python train.py --create-sample-data"
echo "4. Start training: python train.py --model-config configs/model_configs/codellama_7b.yaml"
echo
print_status "Or use the helper script: ./scripts/docker_setup.sh"
