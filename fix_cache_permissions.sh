#!/bin/bash

# Fix Cache Permissions Script
# Resolves "Permission denied: '/.cache'" errors

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

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

print_step "Fixing cache permissions for NeMo 24.07 fine-tuning"
echo ""

# Check if we're in the right directory
if [ ! -f "finetune_pipeline.py" ]; then
    print_error "finetune_pipeline.py not found"
    print_error "Please run this script from the project directory"
    exit 1
fi

# Create cache directories
print_step "1. Creating cache directories..."
mkdir -p .cache/{huggingface,transformers,datasets,torch}
print_info "✓ Cache directories created"

# Set permissions
print_step "2. Setting permissions..."
chmod -R 755 .cache 2>/dev/null || {
    print_warning "Could not set permissions with chmod, trying alternative method..."
    # Try to create with specific permissions
    find .cache -type d -exec chmod 755 {} \; 2>/dev/null || true
    find .cache -type f -exec chmod 644 {} \; 2>/dev/null || true
}
print_info "✓ Permissions set"

# Set environment variables for current session
print_step "3. Setting environment variables..."
export TRANSFORMERS_CACHE="$(pwd)/.cache/transformers"
export HF_HOME="$(pwd)/.cache/huggingface"
export HF_DATASETS_CACHE="$(pwd)/.cache/datasets"
export TORCH_HOME="$(pwd)/.cache/torch"
export XDG_CACHE_HOME="$(pwd)/.cache"

print_info "✓ Environment variables set for current session"

# Create a script to set environment variables
print_step "4. Creating environment setup script..."
cat > set_cache_env.sh << 'EOF'
#!/bin/bash
# Set cache environment variables
export TRANSFORMERS_CACHE="$(pwd)/.cache/transformers"
export HF_HOME="$(pwd)/.cache/huggingface"
export HF_DATASETS_CACHE="$(pwd)/.cache/datasets"
export TORCH_HOME="$(pwd)/.cache/torch"
export XDG_CACHE_HOME="$(pwd)/.cache"

echo "Cache environment variables set:"
echo "  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "  HF_HOME=$HF_HOME"
echo "  XDG_CACHE_HOME=$XDG_CACHE_HOME"
EOF

chmod +x set_cache_env.sh
print_info "✓ Created set_cache_env.sh script"

# Test cache directory access
print_step "5. Testing cache directory access..."
if [ -w ".cache" ]; then
    print_info "✓ Cache directory is writable"
    
    # Test creating a file
    test_file=".cache/test_write_$(date +%s)"
    if echo "test" > "$test_file" 2>/dev/null; then
        rm -f "$test_file"
        print_info "✓ Cache directory write test passed"
    else
        print_warning "Cache directory write test failed"
    fi
else
    print_warning "Cache directory is not writable"
fi

# Show current cache configuration
print_step "6. Current cache configuration:"
echo "  Cache directory: $(pwd)/.cache"
echo "  Directory permissions: $(ls -ld .cache 2>/dev/null | awk '{print $1}' || echo 'unknown')"
echo "  Directory owner: $(ls -ld .cache 2>/dev/null | awk '{print $3":"$4}' || echo 'unknown')"
echo "  Current user: $(id -u):$(id -g)"

echo ""
print_step "Cache permissions fix completed!"
echo ""
print_info "To apply environment variables in your current session:"
print_info "  source ./set_cache_env.sh"
echo ""
print_info "To run training with fixed cache:"
print_info "  source ./set_cache_env.sh && python finetune_pipeline.py --model codellama-13b --data example_training_data.yaml"
echo ""

# Check if we're in Docker
if [ -f /.dockerenv ]; then
    print_info "Running inside Docker container ✓"
    print_info "Cache fix should resolve permission issues"
else
    print_warning "Not running in Docker container"
    print_info "For Docker usage, make sure to use --user flag:"
    print_info "  docker run --user \$(id -u):\$(id -g) ..."
fi
