#!/bin/bash

# Quick Start Script for NeMo Fine-tuning Pipeline
# This script helps you get started with the fine-tuning pipeline

set -e

echo "🚀 NeMo Fine-tuning Pipeline Quick Start"
echo "========================================"

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

# Check if running in NeMo container
check_environment() {
    print_step "Checking environment..."

    if [ -f "/opt/NeMo/README.rst" ] || [ -f "/opt/NeMo/README.md" ] || command -v nemo_run &> /dev/null; then
        print_status "Running in NeMo container ✓"
    else
        print_warning "Not running in NeMo container. Make sure NeMo Framework is installed."
        print_warning "You can start the container with: ./docker_start.sh"
    fi
    
    # Check for GPU
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected ✓"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    else
        print_error "No NVIDIA GPU detected. GPU is required for training."
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    print_step "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_status "Dependencies installed ✓"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Check available models
show_available_models() {
    print_step "Available models:"
    python -c "
from configs.model_configs import list_available_models, get_hardware_requirements
models = list_available_models()
for model in models:
    req = get_hardware_requirements(model)
    print(f'  - {model} (Min: {req[\"min_gpus\"]} GPUs, {req[\"min_gpu_memory_gb\"]}GB each)')
"
}

# Interactive model selection
select_model() {
    echo
    print_step "Model Selection"
    echo "Choose a model for fine-tuning:"
    echo "1) codellama-7b   (Recommended for testing)"
    echo "2) codellama-13b  (Good balance of performance/resources)"
    echo "3) codellama-34b  (High performance, requires more resources)"
    echo "4) llama3-8b      (General purpose, good for testing)"
    echo "5) llama3-70b     (Highest performance, requires significant resources)"
    
    read -p "Enter your choice (1-5): " choice
    
    case $choice in
        1) MODEL="codellama-7b" ;;
        2) MODEL="codellama-13b" ;;
        3) MODEL="codellama-34b" ;;
        4) MODEL="llama3-8b" ;;
        5) MODEL="llama3-70b" ;;
        *) print_error "Invalid choice"; exit 1 ;;
    esac
    
    print_status "Selected model: $MODEL"
    
    # Show hardware requirements
    python -c "
from configs.model_configs import get_hardware_requirements
req = get_hardware_requirements('$MODEL')
print(f'Hardware requirements:')
print(f'  Minimum GPUs: {req[\"min_gpus\"]}')
print(f'  Minimum GPU memory: {req[\"min_gpu_memory_gb\"]}GB per GPU')
print(f'  Recommended GPUs: {req[\"recommended_gpus\"]}')
print(f'  Recommended GPU memory: {req[\"recommended_gpu_memory_gb\"]}GB per GPU')
"
}

# Check for training data
check_training_data() {
    print_step "Checking training data..."
    
    if [ -f "example_training_data.yaml" ]; then
        print_status "Example training data found ✓"
        TRAINING_DATA="example_training_data.yaml"
    else
        print_warning "No training data found."
        echo "Please provide the path to your YAML training data file:"
        read -p "Training data path: " TRAINING_DATA
        
        if [ ! -f "$TRAINING_DATA" ]; then
            print_error "Training data file not found: $TRAINING_DATA"
            exit 1
        fi
    fi
}

# Get Hugging Face token
get_hf_token() {
    print_step "Hugging Face Token"
    
    if [ -n "$HF_TOKEN" ]; then
        print_status "Hugging Face token found in environment ✓"
    else
        echo "Please enter your Hugging Face token (required for model download):"
        echo "You can get a token from: https://huggingface.co/settings/tokens"
        read -s -p "HF Token: " HF_TOKEN
        echo
        
        if [ -z "$HF_TOKEN" ]; then
            print_error "Hugging Face token is required"
            exit 1
        fi
    fi
}

# Run preprocessing
run_preprocessing() {
    print_step "Preprocessing training data..."
    
    python data_preprocessing.py \
        --input "$TRAINING_DATA" \
        --output "processed_training_data" \
        --validation-split 0.1
    
    print_status "Data preprocessing completed ✓"
}

# Run training
run_training() {
    print_step "Starting fine-tuning..."
    
    echo "Training configuration:"
    echo "  Model: $MODEL"
    echo "  Training data: $TRAINING_DATA"
    echo "  Max steps: 50 (for quick demo)"
    echo
    
    read -p "Do you want to proceed with training? (y/N): " confirm
    
    if [[ $confirm =~ ^[Yy]$ ]]; then
        python finetune_pipeline.py \
            --model "$MODEL" \
            --data "$TRAINING_DATA" \
            --output-dir "./outputs" \
            --max-steps 50 \
            --hf-token "$HF_TOKEN"
        
        print_status "Training completed ✓"
        echo "Final model saved to: ./outputs/models/${MODEL}_merged.nemo"
    else
        print_status "Training skipped. You can run it manually later."
    fi
}

# Show next steps
show_next_steps() {
    echo
    print_step "Next Steps"
    echo "1. Evaluate your model:"
    echo "   python evaluate_model.py \\"
    echo "       --model-path ./outputs/models/${MODEL}_merged.nemo \\"
    echo "       --model-name $MODEL \\"
    echo "       --test-data processed_training_data.val.jsonl"
    echo
    echo "2. Use the Jupyter notebook for interactive exploration:"
    echo "   jupyter lab example_usage.ipynb"
    echo
    echo "3. For production training:"
    echo "   - Use more training data"
    echo "   - Increase max_steps (500-2000+)"
    echo "   - Monitor validation loss"
    echo "   - Use appropriate hardware"
    echo
    print_status "Quick start completed! 🎉"
}

# Main execution
main() {
    check_environment
    install_dependencies
    show_available_models
    select_model
    check_training_data
    get_hf_token
    run_preprocessing
    run_training
    show_next_steps
}

# Run main function
main "$@"
