# Dockerfile for NeMo Fine-Tuning Pipeline
# Robust build with enhanced dependency management
FROM ubuntu:22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    gcc \
    g++ \
    make \
    ninja-build \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    sox \
    libsox-fmt-all \
    software-properties-common \
    gnupg2 \
    pkg-config \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA manually
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get -y install cuda-toolkit-11-8 && \
    rm cuda-keyring_1.0-1_all.deb && \
    rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip and install build tools
RUN python -m pip install --upgrade pip setuptools wheel

# Install build dependencies required for NeMo compilation
RUN pip install \
    Cython \
    pybind11 \
    numpy \
    packaging

# Install PyTorch with CUDA support (updated to 2.1+ for better compatibility)
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install NeMo and dependencies
RUN pip install nemo_toolkit[all]==1.20.0

# Install additional ML dependencies (compatible with NeMo requirements)
RUN pip install \
    datasets \
    evaluate \
    rouge-score \
    sacrebleu \
    nltk \
    matplotlib \
    seaborn \
    jupyter \
    ipywidgets

# Install additional dependencies for the pipeline
RUN pip install \
    pyyaml \
    pathlib \
    typing-extensions \
    tqdm \
    psutil \
    gpustat

# Note: apex and flash-attn are excluded from Docker build due to compilation issues
# They can be installed manually if needed using the provided scripts

# Set working directory
WORKDIR /workspace

# Copy the entire pipeline
COPY . .

# Install the pipeline package dependencies using Docker-compatible requirements
RUN pip install -r requirements-docker.txt

# Install the pipeline package without dependencies (since we installed them separately)
RUN pip install -e . --no-deps

# Make scripts executable
RUN chmod +x scripts/*.sh scripts/*.py

# Create necessary directories with proper permissions
RUN mkdir -p /workspace/data \
    /workspace/logs \
    /workspace/checkpoints \
    /workspace/deploy \
    /workspace/outputs \
    && chmod -R 777 /workspace

# Create non-root user for security
RUN useradd -m -u 1000 nemo_user && \
    chown -R nemo_user:nemo_user /workspace

# Switch to non-root user
USER nemo_user

# Set environment variables for the pipeline
ENV PYTHONPATH=/workspace/src:$PYTHONPATH
ENV NEMO_CACHE_DIR=/workspace/.cache/nemo
ENV HF_HOME=/workspace/.cache/huggingface
ENV WANDB_CACHE_DIR=/workspace/.cache/wandb

# Create cache directories
RUN mkdir -p /workspace/.cache/nemo \
    /workspace/.cache/huggingface \
    /workspace/.cache/wandb

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available()); exit(0 if torch.cuda.is_available() else 1)"

# Default command
CMD ["/bin/bash"]
