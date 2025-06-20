#!/bin/bash

# Setup script for Docker environment
# This script prepares the host system for running the NeMo fine-tuning pipeline in Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
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

# Function to check if running as root
check_root() {
    if [ "$EUID" -eq 0 ]; then
        print_error "Please do not run this script as root"
        exit 1
    fi
}

# Function to check OS compatibility
check_os() {
    print_status "Checking OS compatibility..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_success "Linux detected"
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_success "macOS detected"
        OS="macos"
        print_warning "GPU support not available on macOS"
    else
        print_error "Unsupported OS: $OSTYPE"
        exit 1
    fi
}

# Function to install Docker
install_docker() {
    print_status "Installing Docker..."
    
    if command -v docker &> /dev/null; then
        print_success "Docker is already installed"
        docker --version
        return 0
    fi
    
    if [ "$OS" == "linux" ]; then
        # Install Docker on Linux
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        
        # Add user to docker group
        sudo usermod -aG docker $USER
        
        # Start Docker service
        sudo systemctl enable docker
        sudo systemctl start docker
        
        rm get-docker.sh
        print_success "Docker installed successfully"
        print_warning "Please log out and log back in for group changes to take effect"
        
    elif [ "$OS" == "macos" ]; then
        print_error "Please install Docker Desktop for Mac manually from https://docker.com"
        exit 1
    fi
}

# Function to install Docker Compose
install_docker_compose() {
    print_status "Installing Docker Compose..."
    
    if command -v docker-compose &> /dev/null; then
        print_success "Docker Compose is already installed"
        docker-compose --version
        return 0
    fi
    
    if [ "$OS" == "linux" ]; then
        # Install Docker Compose on Linux
        DOCKER_COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d\" -f4)
        sudo curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
        
        print_success "Docker Compose installed successfully"
        docker-compose --version
        
    elif [ "$OS" == "macos" ]; then
        print_success "Docker Compose comes with Docker Desktop for Mac"
    fi
}

# Function to install NVIDIA Docker (Linux only)
install_nvidia_docker() {
    if [ "$OS" != "linux" ]; then
        print_warning "NVIDIA Docker is only available on Linux"
        return 0
    fi
    
    print_status "Installing NVIDIA Docker runtime..."
    
    # Check if NVIDIA drivers are installed
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "NVIDIA drivers not found. Please install NVIDIA drivers first."
        print_error "Visit: https://www.nvidia.com/drivers"
        exit 1
    fi
    
    # Check if nvidia-docker2 is already installed
    if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        print_success "NVIDIA Docker runtime is already working"
        return 0
    fi
    
    # Install NVIDIA Docker
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    
    # Restart Docker
    sudo systemctl restart docker
    
    # Test NVIDIA Docker
    if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        print_success "NVIDIA Docker runtime installed successfully"
    else
        print_error "NVIDIA Docker installation failed"
        exit 1
    fi
}

# Function to create output directories
create_directories() {
    print_status "Creating output directories..."
    
    # Create output directories with proper permissions
    mkdir -p outputs/{checkpoints,logs,deploy,cache}
    mkdir -p notebooks
    mkdir -p docker/monitoring
    
    # Set permissions
    chmod -R 755 outputs/
    chmod -R 755 notebooks/
    
    print_success "Output directories created"
    ls -la outputs/
}

# Function to create monitoring configuration
create_monitoring_config() {
    print_status "Creating monitoring configuration..."
    
    # Create Prometheus configuration
    cat > docker/monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'docker'
    static_configs:
      - targets: ['localhost:9323']
  
  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['localhost:9445']
EOF
    
    # Create Fluent Bit configuration
    cat > docker/monitoring/fluent-bit.conf << EOF
[SERVICE]
    Flush         1
    Log_Level     info
    Daemon        off
    Parsers_File  parsers.conf

[INPUT]
    Name              tail
    Path              /var/lib/docker/containers/*/*-json.log
    Parser            docker
    Tag               docker.*
    Refresh_Interval  5

[OUTPUT]
    Name  stdout
    Match *
EOF
    
    print_success "Monitoring configuration created"
}

# Function to test Docker installation
test_docker() {
    print_status "Testing Docker installation..."
    
    # Test basic Docker functionality
    if docker run --rm hello-world &> /dev/null; then
        print_success "Docker is working correctly"
    else
        print_error "Docker test failed"
        exit 1
    fi
    
    # Test GPU support if available
    if [ "$OS" == "linux" ] && command -v nvidia-smi &> /dev/null; then
        if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
            print_success "GPU support is working"
        else
            print_warning "GPU support test failed"
        fi
    fi
}

# Function to create example environment file
create_env_file() {
    print_status "Creating environment configuration..."
    
    cat > .env.example << EOF
# NeMo Fine-Tuning Pipeline Environment Configuration

# Model Configuration
MODEL_TYPE=llama2
MODEL_SIZE=7b
MAX_EPOCHS=3
BATCH_SIZE=2
LEARNING_RATE=2e-4

# Hardware Configuration
GPUS=1
PRECISION=bf16

# Paths
TRAIN_FILE=data/train.yaml
VAL_FILE=data/val.yaml
TEST_FILE=data/test.yaml

# Output Configuration
OUTPUT_DIR=outputs
CHECKPOINT_DIR=outputs/checkpoints
LOG_DIR=outputs/logs
DEPLOY_DIR=outputs/deploy

# Monitoring
WANDB_PROJECT=nemo-s32k144-finetune
WANDB_ENTITY=your-wandb-entity
TENSORBOARD_LOG_DIR=outputs/logs/tensorboard

# Docker Configuration
CONTAINER_NAME=nemo-finetune-s32k144
IMAGE_NAME=nemo-finetune-pipeline:latest

# Cache Directories
NEMO_CACHE_DIR=outputs/cache/nemo
HF_CACHE_DIR=outputs/cache/huggingface
WANDB_CACHE_DIR=outputs/cache/wandb
EOF
    
    print_success "Environment configuration created (.env.example)"
    print_status "Copy .env.example to .env and customize as needed"
}

# Function to show next steps
show_next_steps() {
    print_success "Docker environment setup completed!"
    echo
    print_status "Next steps:"
    echo "1. Log out and log back in (if Docker was just installed)"
    echo "2. Copy .env.example to .env and customize settings"
    echo "3. Ensure your training data is in the data/ directory"
    echo "4. Run the fine-tuning pipeline:"
    echo "   ./scripts/docker_run.sh run"
    echo
    print_status "Available commands:"
    echo "   ./scripts/docker_run.sh build     # Build Docker image"
    echo "   ./scripts/docker_run.sh run       # Run fine-tuning"
    echo "   ./scripts/docker_run.sh shell     # Interactive shell"
    echo "   ./scripts/docker_run.sh jupyter   # Start Jupyter Lab"
    echo "   ./scripts/docker_run.sh logs      # View training logs"
    echo "   ./scripts/docker_run.sh stop      # Stop containers"
    echo "   ./scripts/docker_run.sh clean     # Clean up"
    echo
    print_status "For detailed documentation, see docker/README.md"
}

# Main execution
main() {
    echo "========================================"
    echo "NeMo Fine-Tuning Pipeline Docker Setup"
    echo "========================================"
    echo
    
    check_root
    check_os
    install_docker
    install_docker_compose
    
    if [ "$OS" == "linux" ]; then
        install_nvidia_docker
    fi
    
    create_directories
    create_monitoring_config
    create_env_file
    test_docker
    show_next_steps
}

# Run main function
main "$@"
