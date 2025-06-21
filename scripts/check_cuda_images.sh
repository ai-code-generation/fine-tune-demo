#!/bin/bash

# Script to check available NVIDIA CUDA Docker images
# This helps troubleshoot Docker image availability issues

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

echo "========================================"
echo "NVIDIA CUDA Docker Image Checker"
echo "========================================"
echo

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

print_status "Checking Docker connectivity..."
if ! docker info &> /dev/null; then
    print_error "Cannot connect to Docker daemon. Make sure Docker is running."
    exit 1
fi
print_success "Docker is running"

echo
print_status "Testing NVIDIA CUDA base images..."

# List of NVIDIA CUDA images to test (in order of preference)
CUDA_IMAGES=(
    "nvidia/cuda:11.8-cudnn8-devel-ubuntu22.04"
    "nvidia/cuda:11.8-devel-ubuntu22.04"
    "nvidia/cuda:11.8-cudnn8-devel-ubuntu20.04"
    "nvidia/cuda:11.8-devel-ubuntu20.04"
    "nvidia/cuda:12.0-cudnn8-devel-ubuntu22.04"
    "nvidia/cuda:12.0-devel-ubuntu22.04"
    "nvidia/cuda:11.7-cudnn8-devel-ubuntu22.04"
    "nvidia/cuda:11.7-devel-ubuntu22.04"
)

WORKING_IMAGES=()
FAILED_IMAGES=()

for image in "${CUDA_IMAGES[@]}"; do
    print_status "Testing: $image"
    
    # Try to pull the image metadata (without downloading the full image)
    if docker manifest inspect "$image" &> /dev/null; then
        print_success "✓ Available: $image"
        WORKING_IMAGES+=("$image")
    else
        print_warning "✗ Not available: $image"
        FAILED_IMAGES+=("$image")
    fi
done

echo
echo "========================================"
echo "RESULTS"
echo "========================================"

if [ ${#WORKING_IMAGES[@]} -gt 0 ]; then
    print_success "Available NVIDIA CUDA images:"
    for image in "${WORKING_IMAGES[@]}"; do
        echo "  ✓ $image"
    done
    
    echo
    print_status "Recommended image: ${WORKING_IMAGES[0]}"
    
    echo
    print_status "To update your Dockerfile, use:"
    echo "FROM ${WORKING_IMAGES[0]} as base"
    
else
    print_error "No NVIDIA CUDA images are available!"
    print_error "This might indicate:"
    echo "  1. Network connectivity issues"
    echo "  2. Docker Hub access problems"
    echo "  3. NVIDIA Docker repository issues"
    
    echo
    print_status "Troubleshooting steps:"
    echo "1. Check internet connectivity:"
    echo "   ping docker.io"
    echo
    echo "2. Try pulling a basic image:"
    echo "   docker pull ubuntu:22.04"
    echo
    echo "3. Check Docker Hub status:"
    echo "   https://status.docker.com/"
    echo
    echo "4. Try using a different registry:"
    echo "   docker pull nvcr.io/nvidia/cuda:11.8-devel-ubuntu22.04"
fi

if [ ${#FAILED_IMAGES[@]} -gt 0 ]; then
    echo
    print_warning "Unavailable images:"
    for image in "${FAILED_IMAGES[@]}"; do
        echo "  ✗ $image"
    done
fi

echo
print_status "Alternative solutions:"
echo

echo "1. Use NVIDIA Container Registry (nvcr.io):"
echo "   FROM nvcr.io/nvidia/cuda:11.8-devel-ubuntu22.04"
echo

echo "2. Use a different CUDA version:"
echo "   FROM nvidia/cuda:12.0-devel-ubuntu22.04"
echo

echo "3. Use Ubuntu base + CUDA installation:"
echo "   FROM ubuntu:22.04"
echo "   # Then install CUDA manually"
echo

echo "4. Use PyTorch official image:"
echo "   FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel"
echo

# Test NVIDIA Container Registry as alternative
echo
print_status "Testing NVIDIA Container Registry (nvcr.io) as alternative..."

NVCR_IMAGES=(
    "nvcr.io/nvidia/cuda:11.8-devel-ubuntu22.04"
    "nvcr.io/nvidia/pytorch:23.08-py3"
    "nvcr.io/nvidia/tensorflow:23.08-tf2-py3"
)

for image in "${NVCR_IMAGES[@]}"; do
    print_status "Testing: $image"
    if docker manifest inspect "$image" &> /dev/null; then
        print_success "✓ Available: $image"
    else
        print_warning "✗ Not available: $image"
    fi
done

echo
print_status "If you're still having issues, try building without CUDA first:"
echo "docker build --target base-no-cuda -t test-build ."
