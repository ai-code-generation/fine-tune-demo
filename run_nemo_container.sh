#!/bin/bash

# Simple NeMo 24.07 Container Runner
# Based on official NeMo documentation

echo "🐳 Starting NeMo 24.07 Container..."
echo "📁 Workspace: $(pwd)"
echo "📁 Results will be saved to: $(pwd)/results"

# Create results directory if it doesn't exist
mkdir -p results

# Run NeMo container with proper settings
docker run --gpus all \
  --shm-size=2g \
  --net=host \
  --ulimit memlock=-1 \
  --rm -it \
  -v ${PWD}:/workspace \
  -w /workspace \
  -v ${PWD}/results:/results \
  --user $(id -u):$(id -g) \
  nvcr.io/nvidia/nemo:24.07 \
  bash

echo "🏁 Container exited"
