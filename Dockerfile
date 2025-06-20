# Multi-stage Dockerfile for NeMo Fine-Tuning Pipeline
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libsndfile1 \
    ffmpeg \
    sox \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install NeMo and dependencies
RUN pip install nemo_toolkit[all]==1.20.0

# Install additional ML dependencies
RUN pip install \
    transformers>=4.30.0 \
    pytorch-lightning>=2.0.0 \
    omegaconf>=2.3.0 \
    hydra-core>=1.3.0 \
    wandb \
    tensorboard \
    datasets \
    evaluate \
    rouge-score \
    sacrebleu \
    nltk \
    scikit-learn \
    pandas \
    numpy \
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

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the entire pipeline
COPY . .

# Install the pipeline package
RUN pip install -e .

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
