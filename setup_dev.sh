#!/bin/bash

# Quick Development Setup Script
# Sets up everything needed for manual LLM server development

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
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

print_info "🚀 LLM Server Development Setup"
echo

# Step 1: Check Python
print_info "Step 1: Checking Python..."
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_success "✓ Python 3 found: $python_version"
elif command -v python &> /dev/null; then
    python_version=$(python --version 2>&1 | cut -d' ' -f2)
    if [[ $python_version == 3.* ]]; then
        print_success "✓ Python 3 found: $python_version"
        alias python3=python
    else
        print_error "❌ Python 3 required, found: $python_version"
        exit 1
    fi
else
    print_error "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

# Step 2: Check pip
print_info "Step 2: Checking pip..."
if command -v pip3 &> /dev/null || command -v pip &> /dev/null; then
    print_success "✓ pip found"
else
    print_error "❌ pip not found. Please install pip"
    exit 1
fi

# Step 3: Install dependencies
print_info "Step 3: Installing dependencies..."
if ./run_manual.sh --install-deps; then
    print_success "✓ Dependencies installed"
else
    print_error "❌ Failed to install dependencies"
    exit 1
fi

# Step 4: Setup .env
print_info "Step 4: Setting up .env file..."
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        print_success "✓ Created .env from .env.example"
    else
        cat > .env << EOF
# LLM Server Configuration for Manual Development

# Model Configuration
HF_MODEL_LOCAL_PATH=./models/my-finetuned-model
HF_MODEL_NAME=microsoft/DialoGPT-medium

# Server Configuration
HOST=127.0.0.1
PORT=8000
LOG_LEVEL=INFO

# Performance Settings
DEVICE=auto
TORCH_DTYPE=float16
LOW_CPU_MEM_USAGE=true

# Development Settings
MAX_TOKENS=512
TEMPERATURE=0.7
EOF
        print_success "✓ Created default .env file"
    fi
else
    print_success "✓ .env file already exists"
fi

# Step 5: Download test model
print_info "Step 5: Checking/downloading test model..."
if [ -d "./models/my-finetuned-model" ] && [ -f "./models/my-finetuned-model/config.json" ]; then
    print_success "✓ Test model already exists"
else
    print_info "Downloading test model (this may take a few minutes)..."
    if ./run_manual.sh --download-model; then
        print_success "✓ Test model downloaded"
    else
        print_error "❌ Failed to download test model"
        exit 1
    fi
fi

# Step 6: Make scripts executable
print_info "Step 6: Making scripts executable..."
chmod +x run_manual.sh test_manual.sh
print_success "✓ Scripts are executable"

# Step 7: Quick validation
print_info "Step 7: Running validation check..."
if ./run_manual.sh --check-deps; then
    print_success "✓ All dependencies validated"
else
    print_error "❌ Dependency validation failed"
    exit 1
fi

echo
print_success "🎉 Development environment setup complete!"
echo
print_info "Quick start commands:"
echo "  ./run_manual.sh                 # Start server (default: http://127.0.0.1:8000)"
echo "  ./run_manual.sh --port 8001     # Start on different port"
echo "  ./run_manual.sh --reload        # Development mode with auto-reload"
echo "  ./run_manual.sh --device cpu    # Force CPU-only mode"
echo "  ./test_manual.sh                # Test the running server"
echo
print_info "Development workflow:"
echo "  1. Start server: ./run_manual.sh --reload --log-level debug"
echo "  2. In another terminal: ./test_manual.sh"
echo "  3. Test API: curl http://127.0.0.1:8000/v1/health/ready"
echo "  4. View docs: http://127.0.0.1:8000/docs"
echo
print_info "Configuration files:"
echo "  - .env                   # Environment variables"
echo "  - models/               # Local models directory"
echo "  - app.py                # Main server code"
echo
print_warning "Note: This setup is for development only. Use Docker for production."
