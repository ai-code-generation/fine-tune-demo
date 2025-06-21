#!/bin/bash

# Script to install NVIDIA Apex inside the Docker container
# This is separated from the main Dockerfile due to compilation complexity

echo "Installing NVIDIA Apex..."
echo "This may take several minutes and requires CUDA support."

# Set environment variables for compilation
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Method 1: Try installing from PyPI first
echo "Attempting to install apex from PyPI..."
pip install enscons
pip install apex --no-build-isolation

if [ $? -eq 0 ]; then
    echo "Apex installed successfully from PyPI!"
    python -c "import apex; print('Apex imported successfully')"
    exit 0
fi

# Method 2: Install from source if PyPI fails
echo "PyPI installation failed. Installing from source..."
cd /tmp
git clone https://github.com/NVIDIA/apex.git
cd apex

# Install with C++ and CUDA extensions
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--global-option=--cpp_ext" \
    --config-settings "--global-option=--cuda_ext" ./

if [ $? -eq 0 ]; then
    echo "Apex installed successfully from source!"
    python -c "import apex; print('Apex imported successfully')"
    cd /workspace
    rm -rf /tmp/apex
else
    echo "Apex installation failed. This is optional and the pipeline will work without it."
    echo "Apex provides mixed precision training optimizations but is not required."
    cd /workspace
    rm -rf /tmp/apex
    exit 1
fi
