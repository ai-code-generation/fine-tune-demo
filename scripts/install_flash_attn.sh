#!/bin/bash

# Script to install flash-attn inside the Docker container
# This is separated from the main Dockerfile due to compilation complexity

echo "Installing flash-attn..."
echo "This may take several minutes and requires significant memory."
echo "Make sure you have at least 8GB of RAM available."

# Set environment variables for compilation
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install flash-attn with specific options
pip install flash-attn>=2.0.0 --no-build-isolation --verbose

if [ $? -eq 0 ]; then
    echo "flash-attn installed successfully!"
    python -c "import flash_attn; print(f'Flash Attention version: {flash_attn.__version__}')"
else
    echo "flash-attn installation failed. This is optional and the pipeline will work without it."
    echo "Flash attention provides memory-efficient attention computation but is not required."
fi
