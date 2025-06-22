#!/bin/bash

# NeMo 24.07 Quick Start Script
# Optimized for CodeLlama-13B and Llama3 models (8B-70B)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

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

print_status() {
    echo -e "${GREEN}[✓] $1${NC}"
}

# Check if running in Docker
check_environment() {
    print_step "Checking environment..."
    
    if [ -f /.dockerenv ]; then
        print_status "Running inside Docker container"
        DOCKER_ENV=true
    else
        print_warning "Not running in Docker container"
        print_info "For best results, use: docker-compose up nemo-finetuning"
        DOCKER_ENV=false
    fi
    
    # Check if we're in the right directory
    if [ ! -f "finetune_pipeline.py" ]; then
        print_error "finetune_pipeline.py not found"
        print_error "Make sure you're in the project directory"
        exit 1
    fi
    
    print_status "Environment check completed"
}

# Get Hugging Face token
get_hf_token() {
    print_step "Hugging Face Token Setup"
    
    # Check if token is in environment
    if [ -n "$HF_TOKEN" ]; then
        print_status "Hugging Face token found in environment ✓"
        return 0
    fi
    
    # Check if token is in .env file
    if [ -f ".env" ] && grep -q "HF_TOKEN=" .env; then
        HF_TOKEN=$(grep "HF_TOKEN=" .env | cut -d'=' -f2)
        if [ -n "$HF_TOKEN" ]; then
            print_status "Hugging Face token found in .env file ✓"
            export HF_TOKEN
            return 0
        fi
    fi
    
    # If no token found, prompt user
    print_warning "No Hugging Face token found"
    echo "You need a HF token to download gated models (CodeLlama, Llama3)"
    echo "1. Go to: https://huggingface.co/settings/tokens"
    echo "2. Create a token with 'read' permissions"
    echo "3. For gated models, request access first"
    echo ""
    read -p "Do you want to enter your token now? (y/N): " manual_token
    
    if [[ $manual_token =~ ^[Yy]$ ]]; then
        read -s -p "HF Token: " HF_TOKEN
        echo
        
        if [ -z "$HF_TOKEN" ]; then
            print_error "Hugging Face token is required"
            exit 1
        fi
        
        # Save to .env file for future use
        echo "HF_TOKEN=$HF_TOKEN" >> .env
        print_status "Token saved to .env file"
    else
        print_error "Hugging Face token is required for gated models"
        print_error "Please run: python setup_hf_auth.py"
        exit 1
    fi
}

# Select model
select_model() {
    print_step "Model Selection"
    
    echo "Available models for NeMo 24.07:"
    echo "1. codellama-13b    - CodeLlama 13B (requires HF access)"
    echo "2. llama3-8b        - Llama3 8B (requires HF access)"
    echo "3. llama3-70b       - Llama3 70B (requires HF access, high GPU requirements)"
    echo ""
    
    read -p "Select model (1-3): " model_choice
    
    case $model_choice in
        1)
            MODEL="codellama-13b"
            print_info "Selected: CodeLlama-13B"
            ;;
        2)
            MODEL="llama3-8b"
            print_info "Selected: Llama3-8B"
            ;;
        3)
            MODEL="llama3-70b"
            print_info "Selected: Llama3-70B"
            print_warning "This model requires significant GPU resources (16+ GPUs)"
            ;;
        *)
            print_error "Invalid selection"
            exit 1
            ;;
    esac
}

# Check hardware requirements
check_hardware() {
    print_step "Checking hardware requirements for $MODEL..."
    
    python finetune_pipeline.py --model "$MODEL" --check-hardware
    
    echo ""
    read -p "Do your hardware specs meet the requirements? (y/N): " hw_confirm
    
    if [[ ! $hw_confirm =~ ^[Yy]$ ]]; then
        print_warning "Consider using a smaller model or upgrading hardware"
        echo "You can still proceed, but training may be slow or fail"
        read -p "Continue anyway? (y/N): " force_continue
        if [[ ! $force_continue =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Select training data
select_training_data() {
    print_step "Training Data Selection"
    
    if [ -f "example_training_data.yaml" ]; then
        echo "Found example_training_data.yaml"
        read -p "Use example data? (Y/n): " use_example
        
        if [[ $use_example =~ ^[Nn]$ ]]; then
            read -p "Enter path to your YAML training data: " TRAINING_DATA
        else
            TRAINING_DATA="example_training_data.yaml"
        fi
    else
        read -p "Enter path to your YAML training data: " TRAINING_DATA
    fi
    
    if [ ! -f "$TRAINING_DATA" ]; then
        print_error "Training data file not found: $TRAINING_DATA"
        exit 1
    fi
    
    print_status "Training data: $TRAINING_DATA"
}

# Set training parameters
set_training_params() {
    print_step "Training Parameters"
    
    echo "Training configuration:"
    echo "  Model: $MODEL"
    echo "  Data: $TRAINING_DATA"
    echo "  Output: ./outputs"
    
    read -p "Max training steps (default: 100): " MAX_STEPS
    MAX_STEPS=${MAX_STEPS:-100}
    
    read -p "Validation split (default: 0.1): " VAL_SPLIT
    VAL_SPLIT=${VAL_SPLIT:-0.1}
    
    print_info "Training will run for $MAX_STEPS steps with $VAL_SPLIT validation split"
}

# Run preprocessing
run_preprocessing() {
    print_step "Preprocessing training data..."
    
    python data_preprocessing.py \
        --input "$TRAINING_DATA" \
        --output "processed_training_data" \
        --validation-split "$VAL_SPLIT"
    
    if [ $? -eq 0 ]; then
        print_status "Data preprocessing completed ✓"
    else
        print_error "Data preprocessing failed"
        exit 1
    fi
}

# Run training
run_training() {
    print_step "Starting fine-tuning..."
    
    echo "Training configuration:"
    echo "  Model: $MODEL"
    echo "  Training data: $TRAINING_DATA"
    echo "  Max steps: $MAX_STEPS (for quick demo)"
    echo ""
    
    read -p "Do you want to proceed with training? (y/N): " confirm
    
    if [[ $confirm =~ ^[Yy]$ ]]; then
        python finetune_pipeline.py \
            --model "$MODEL" \
            --data "$TRAINING_DATA" \
            --output-dir "./outputs" \
            --max-steps "$MAX_STEPS" \
            --validation-split "$VAL_SPLIT" \
            --hf-token "$HF_TOKEN"
        
        if [ $? -eq 0 ]; then
            print_status "Fine-tuning completed successfully! ✓"
            echo ""
            echo "Your fine-tuned model is saved in: ./outputs/models/"
            echo "Training logs are in: ./outputs/logs/"
            echo ""
            echo "Next steps:"
            echo "1. Evaluate your model: python evaluate_model.py"
            echo "2. Test inference with your fine-tuned model"
            echo "3. Deploy for production use"
        else
            print_error "Fine-tuning failed"
            exit 1
        fi
    else
        print_info "Training cancelled by user"
        exit 0
    fi
}

# Main execution
main() {
    print_header "NeMo 24.07 Fine-tuning Quick Start"
    print_info "Optimized for CodeLlama-13B and Llama3 models (8B-70B)"
    print_info "Host mount: /static-data/team_08/simple-nemo/fine-tune-demo"
    echo ""
    
    check_environment
    get_hf_token
    select_model
    check_hardware
    select_training_data
    set_training_params
    run_preprocessing
    run_training
    
    print_header "Quick Start Completed Successfully!"
}

# Run main function
main "$@"
