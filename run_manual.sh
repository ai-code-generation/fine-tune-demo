#!/bin/bash

# LLM Server Manual Run Script (No Docker)
# Quick debug and development script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
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

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Run LLM Server manually (without Docker) for quick debugging.

OPTIONS:
    --port PORT         Server port (default: 8000)
    --host HOST         Server host (default: 127.0.0.1)
    --device DEVICE     Device: cpu, cuda, auto (default: auto)
    --model-path PATH   Path to local model (default: ./models/my-finetuned-model)
    --log-level LEVEL   Log level: debug, info, warning, error (default: info)
    --reload            Enable auto-reload for development
    --install-deps      Install Python dependencies
    --check-deps        Check if dependencies are installed
    --download-model    Download a test model
    -h, --help          Show this help

EXAMPLES:
    $0                                    # Run with default settings
    $0 --port 8001 --device cpu          # Run on port 8001 with CPU only
    $0 --reload --log-level debug        # Development mode with debug logs
    $0 --install-deps                    # Install dependencies and exit
    $0 --download-model                  # Download test model and exit
    $0 --check-deps                      # Check dependencies and exit

ENVIRONMENT VARIABLES:
    HF_MODEL_LOCAL_PATH    Path to local model
    HF_MODEL_NAME          HuggingFace Hub model name (fallback)
    HF_TOKEN               HuggingFace token
    DEVICE                 Device to use (cpu, cuda, auto)
    PORT                   Server port
    HOST                   Server host
    LOG_LEVEL              Log level

EOF
}

# Default values
PORT=8000
HOST="127.0.0.1"
DEVICE="auto"
MODEL_PATH="./models/CodeLlama-7b-Instruct-hf"
LOG_LEVEL="info"
RELOAD=false
INSTALL_DEPS=false
CHECK_DEPS=false
DOWNLOAD_MODEL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --reload)
            RELOAD=true
            shift
            ;;
        --install-deps)
            INSTALL_DEPS=true
            shift
            ;;
        --check-deps)
            CHECK_DEPS=true
            shift
            ;;
        --download-model)
            DOWNLOAD_MODEL=true
            shift
            ;;
        -h|--help)
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

# Function to check if Python package is installed
check_package() {
    local package=$1
    if py -c "import $package" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to check dependencies
check_dependencies() {
    print_info "Checking Python dependencies..."
    
    local missing_packages=()
    local packages=("fastapi" "uvicorn" "transformers" "torch" "numpy")
    
    for package in "${packages[@]}"; do
        if check_package "$package"; then
            print_success "✓ $package installed"
        else
            print_error "✗ $package missing"
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        print_error "Missing packages: ${missing_packages[*]}"
        print_info "Run: $0 --install-deps"
        return 1
    else
        print_success "All dependencies are installed"
        return 0
    fi
}

# Function to install dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."
    
    # Check if pip is available
    if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
        print_error "pip not found. Please install pip first."
        exit 1
    fi
    
    local pip_cmd="pip3"
    if ! command -v pip3 &> /dev/null; then
        pip_cmd="pip"
    fi
    
    # Install from requirements.txt if it exists
    if [ -f "requirements.txt" ]; then
        print_info "Installing from requirements.txt..."
        $pip_cmd install -r requirements.txt
    else
        print_info "Installing essential packages..."
        $pip_cmd install fastapi uvicorn transformers torch numpy pydantic requests python-dotenv
    fi
    
    print_success "Dependencies installed successfully!"
}

# Function to download test model
download_test_model() {
    print_info "Downloading test model..."
    
    if ! check_package "transformers"; then
        print_error "transformers package not installed. Run --install-deps first."
        exit 1
    fi
    
    # Create models directory
    mkdir -p "$MODEL_PATH"
    
    py << EOF
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "$MODEL_PATH"
print(f"Downloading model to: {model_path}")

try:
    # Download a small model for testing
    model_name = "microsoft/DialoGPT-medium"
    print(f"Downloading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)
    
    print(f"✅ Model downloaded successfully to {model_path}")
    
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        print_success "Test model downloaded successfully!"
        print_info "Model location: $MODEL_PATH"
    else
        print_error "Failed to download test model"
        exit 1
    fi
}

# Function to check model files
check_model() {
    local model_path="$1"
    
    print_info "Checking model at: $model_path"
    
    if [ ! -d "$model_path" ]; then
        print_warning "Model directory not found: $model_path"
        return 1
    fi
    
    local required_files=("config.json")
    local model_files=("config.json" "pytorch_model-00001-of-00003.bin" "pytorch_model.bin" "model.safetensors" "pytorch_model.safetensors")
    
    # Check config.json
    if [ ! -f "$model_path/config.json" ]; then
        print_error "Missing required file: config.json"
        return 1
    else
        print_success "✓ config.json found"
    fi
    
    # Check for model files
    local model_file_found=false
    for file in "${model_files[@]}"; do
        if [ -f "$model_path/$file" ]; then
            print_success "✓ Model file found: $file"
            model_file_found=true
            break
        fi
    done
    
    if [ "$model_file_found" = false ]; then
        print_error "No model files found. Expected one of: ${model_files[*]}"
        return 1
    fi
    
    # Check tokenizer files
    if [ -f "$model_path/tokenizer_config.json" ] || [ -f "$model_path/tokenizer.json" ]; then
        print_success "✓ Tokenizer files found"
    else
        print_warning "No tokenizer files found"
    fi
    
    return 0
}

# Function to check GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            print_success "✓ NVIDIA GPU available"
            return 0
        else
            print_warning "nvidia-smi found but GPU not accessible"
        fi
    else
        print_info "No NVIDIA GPU detected"
    fi
    return 1
}

# Function to set environment variables
setup_environment() {
    print_info "Setting up environment..."
    
    # Load .env file if it exists
    if [ -f ".env" ]; then
        print_info "Loading .env file..."
        set -a
        source .env
        set +a
    fi
    
    # Override with command line arguments
    export HF_MODEL_LOCAL_PATH="${MODEL_PATH}"
    export DEVICE="${DEVICE}"
    export HOST="${HOST}"
    export PORT="${PORT}"
    export LOG_LEVEL="${LOG_LEVEL^^}"  # Convert to uppercase
    
    # Set fallback values
    export HF_MODEL_NAME="${HF_MODEL_NAME:-microsoft/DialoGPT-medium}"
    export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
    export TORCH_DTYPE="${TORCH_DTYPE:-float16}"
    export LOW_CPU_MEM_USAGE="${LOW_CPU_MEM_USAGE:-true}"
    
    print_info "Environment configuration:"
    echo "  - Model Path: $HF_MODEL_LOCAL_PATH"
    echo "  - Device: $DEVICE"
    echo "  - Host: $HOST"
    echo "  - Port: $PORT"
    echo "  - Log Level: $LOG_LEVEL"
}

# Function to run the server
run_server() {
    print_info "Starting LLM Server..."
    
    # Check if app.py exists
    if [ ! -f "app.py" ]; then
        print_error "app.py not found in current directory"
        exit 1
    fi
    
    # Prepare uvicorn command
    local uvicorn_cmd="py -m uvicorn app:app"
    uvicorn_cmd="$uvicorn_cmd --host $HOST --port $PORT"
    uvicorn_cmd="$uvicorn_cmd --log-level ${LOG_LEVEL,,}"  # Convert to lowercase
    
    if [ "$RELOAD" = true ]; then
        uvicorn_cmd="$uvicorn_cmd --reload"
        print_info "Development mode: Auto-reload enabled"
    fi
    
    print_info "Starting server with command:"
    echo "  $uvicorn_cmd"
    echo ""
    print_success "🚀 Server starting..."
    print_info "Access the API at: http://$HOST:$PORT"
    print_info "Health check: http://$HOST:$PORT/v1/health/ready"
    print_info "API docs: http://$HOST:$PORT/docs"
    echo ""
    print_info "Press Ctrl+C to stop the server"
    echo ""
    
    # Run the server
    exec $uvicorn_cmd
}

# Main execution
main() {
    print_info "LLM Server Manual Runner"
    echo ""
    
    # Handle special flags first
    if [ "$INSTALL_DEPS" = true ]; then
        install_dependencies
        exit 0
    fi
    
    if [ "$CHECK_DEPS" = true ]; then
        check_dependencies
        exit $?
    fi
    
    if [ "$DOWNLOAD_MODEL" = true ]; then
        download_test_model
        exit 0
    fi
    
    # Check dependencies
    if ! check_dependencies; then
        print_error "Please install dependencies first: $0 --install-deps"
        exit 1
    fi
    
    # Setup environment
    setup_environment
    
    # Check No model files found. Expected one of
    if ! check_model "$MODEL_PATH"; then
        print_warning "Model validation failed"
        print_info "To download a test model: $0 --download-model"
        print_info "Or set correct model path: --model-path /path/to/your/model"
        exit 1
    fi
    
    # Check GPU if requested
    if [ "$DEVICE" = "cuda" ]; then
        if ! check_gpu; then
            print_warning "CUDA requested but GPU not available, falling back to CPU"
            export DEVICE="cpu"
        fi
    elif [ "$DEVICE" = "auto" ]; then
        if check_gpu; then
            print_info "Auto-detected CUDA, using GPU acceleration"
            export DEVICE="cuda"
        else
            print_info "Auto-detected CPU-only mode"
            export DEVICE="cpu"
        fi
    fi
    
    # Run the server
    run_server
}

# Run main function
main "$@"
