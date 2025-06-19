#!/bin/bash

# LLM Server Deploy Script - HuggingFace Backend Only

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Deploy LLM Server with HuggingFace backend"
    echo ""
    echo "OPTIONS:"
    echo "  -d, --down   Stop and remove containers"
    echo "  -h, --help   Show this help"
    echo ""
    echo "Examples:"
    echo "  $0           # Deploy HuggingFace LLM server"
    echo "  $0 --down    # Stop deployment"
}

# Default values
DOWN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--down)
            DOWN=true
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

# Check if .env exists
if [ ! -f .env ]; then
    print_warning ".env file not found, copying from .env.example"
    cp .env.example .env
    print_info "Please edit .env file with your HuggingFace configurations"
fi

# Create network if it doesn't exist
docker network create rag-network 2>/dev/null || true

if [ "$DOWN" = true ]; then
    print_info "Stopping LLM server..."
    docker-compose down -v
    
    print_info "LLM server stopped"
    exit 0
fi

print_info "Starting LLM server with HuggingFace backend..."

# Try to start with error handling
if ! docker-compose up -d --build; then
    print_error "Failed to start LLM server"
    print_info "Troubleshooting steps:"
    print_info "1. Check Docker logs: docker-compose logs"
    print_info "2. Verify .env configuration"
    print_info "3. Ensure sufficient memory available"
    exit 1
fi

# Wait for service to be ready
print_info "Waiting for service to be ready..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s -f http://localhost:8000/v1/health/ready > /dev/null 2>&1; then
        break
    fi
    
    attempt=$((attempt + 1))
    if [ $attempt -eq $max_attempts ]; then
        print_warning "Service health check timed out"
        print_info "Service may still be starting. Check logs: docker logs llm-server"
        break
    fi
    sleep 2
done

print_info "LLM server started successfully!"
echo ""
print_info "Access points:"
echo "  - LLM API: http://localhost:8000"
echo "  - Health: http://localhost:8000/v1/health/ready"
echo "  - Models: http://localhost:8000/v1/models"
echo "  - Chat Completions: http://localhost:8000/v1/chat/completions"

print_info "Backend: HuggingFace Transformers"

echo ""
print_info "To view logs:"
echo "  docker logs -f llm-server"
echo ""
print_info "To stop:"
echo "  ./deploy.sh --down"

print_info "Use '$0 --down' to stop the deployment"
