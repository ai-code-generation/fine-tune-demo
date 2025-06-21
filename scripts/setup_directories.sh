#!/bin/bash

# Setup script to create necessary directories for the fine-tuning pipeline
# Run this before starting Docker containers

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

print_header "Setting up directories for CodeLlama fine-tuning pipeline"
echo "============================================================"

# Create necessary directories
print_status "Creating directories..."

mkdir -p cache
mkdir -p logs  
mkdir -p output
mkdir -p data
mkdir -p configs/model_configs
mkdir -p configs/lora_configs

# Set proper permissions
chmod 755 cache logs output data configs

print_status "Directory structure created:"
echo "  📁 ./cache     - Model downloads and caching"
echo "  📁 ./logs      - Training logs and monitoring"
echo "  📁 ./output    - Trained models and checkpoints"
echo "  📁 ./data      - Training and validation data"
echo "  📁 ./configs   - Model and LoRA configurations"

# Check if directories exist and show sizes
print_status "Directory status:"
for dir in cache logs output data configs; do
    if [ -d "$dir" ]; then
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "  ✅ $dir ($size)"
    else
        echo "  ❌ $dir (not created)"
    fi
done

print_status "Setup complete! You can now start the Docker containers."
echo ""
echo "Next steps:"
echo "  1. ./scripts/build_docker.sh"
echo "  2. ./scripts/docker_setup.sh start"
echo "  3. docker exec -it codellama-finetune python train.py --create-sample-data"
