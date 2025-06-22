#!/bin/bash

# Fix permissions script for NeMo fine-tuning pipeline
# This script helps fix common permission issues with Docker containers

echo "🔧 Fixing Permissions for NeMo Pipeline"
echo "======================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "finetune_pipeline.py" ]; then
    print_error "finetune_pipeline.py not found in current directory."
    print_error "Please run this script from the project root directory."
    exit 1
fi

print_status "Found project files in current directory"

# Create necessary directories with proper permissions
print_status "Creating necessary directories..."

DIRS=("outputs" "data" "models" "outputs/models" "outputs/data" "outputs/experiments")

for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    else
        print_status "Directory already exists: $dir"
    fi
done

# Set proper permissions
print_status "Setting proper permissions..."

# Make scripts executable
SCRIPTS=("quick_start.sh" "docker_start.sh" "debug_docker.sh" "fix_permissions.sh")

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        chmod +x "$script"
        print_status "Made executable: $script"
    fi
done

# Set directory permissions
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        chmod 755 "$dir"
        print_status "Set permissions for: $dir"
    fi
done

# Export environment variables for docker-compose
print_status "Setting up environment variables for Docker Compose..."

# Create .env file for docker-compose
cat > .env << EOF
# User and Group IDs for proper permissions
UID=$(id -u)
GID=$(id -g)
EOF

print_status "Created .env file with UID=$(id -u) and GID=$(id -g)"

echo ""
print_status "✅ Permission fixes completed!"
echo ""
echo "Now you can run the container with proper permissions:"
echo ""
echo "Option 1 - Docker Compose:"
echo "  docker-compose up nemo-finetuning"
echo ""
echo "Option 2 - Direct Docker:"
echo "  docker run --gpus all --shm-size=8g \\"
echo "    --ulimit memlock=-1 --rm -it \\"
echo "    --user \$(id -u):\$(id -g) \\"
echo "    -v \"\$(pwd):/workspace\" -w /workspace \\"
echo "    -p 8888:8888 \\"
echo "    nvcr.io/nvidia/nemo:25.04.01.llama_nemotron_nano_vl \\"
echo "    bash -c 'mkdir -p outputs data models && bash'"
echo ""
echo "Option 3 - Use the startup script:"
echo "  ./docker_start.sh"
echo ""
print_status "If you still get permission errors, try running with sudo or check Docker permissions."
