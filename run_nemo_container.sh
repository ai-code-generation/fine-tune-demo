#!/bin/bash

# Simple NeMo 24.07 Container Runner
# Based on official NeMo documentation

echo "🐳 Starting NeMo 24.07 Container..."
echo "📁 Workspace: /static-data/team_08/simple-nemo/fine-tune-demo"
echo "📁 Results will be saved to: /static-data/team_08/simple-nemo/fine-tune-demo/results"

# Create necessary directories
mkdir -p results
mkdir -p cache
mkdir -p models

# Run NeMo container with proper settings and cache directory
docker run --gpus all \
  --shm-size=2g \
  --net=host \
  --ulimit memlock=-1 \
  --rm -it \
  -v /static-data/team_08/simple-nemo/fine-tune-demo:/workspace \
  -w /workspace \
  -v /static-data/team_08/simple-nemo/fine-tune-demo/results:/results \
  -v /static-data/team_08/simple-nemo/fine-tune-demo/cache:/root/.cache \
  -e HF_HOME=/workspace/cache/huggingface \
  -e TRANSFORMERS_CACHE=/workspace/cache/transformers \
  -e HF_DATASETS_CACHE=/workspace/cache/datasets \
  nvcr.io/nvidia/nemo:24.07 \
  bash

echo "🏁 Container exited"
