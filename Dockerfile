# CodeLlama Fine-tuning Pipeline Dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies and CUDA toolkit
RUN apt-get update && apt-get install -y \
    # Python and development tools
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    # Build tools
    build-essential \
    cmake \
    ninja-build \
    # Version control and utilities
    git \
    wget \
    curl \
    unzip \
    # System monitoring and editing
    vim \
    nano \
    htop \
    tmux \
    tree \
    # Additional libraries for ML
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    # CUDA development tools
    cuda-toolkit-12-1 \
    cuda-nvcc-12-1 \
    cuda-libraries-dev-12-1 \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip and install essential Python tools
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA support first (most important dependency)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install other ML/AI dependencies
RUN pip install --no-cache-dir \
    # Hugging Face ecosystem
    transformers==4.35.2 \
    tokenizers==0.14.1 \
    datasets==2.14.6 \
    accelerate==0.24.1 \
    peft==0.6.2 \
    # Quantization and optimization
    bitsandbytes==0.41.3 \
    # Scientific computing
    numpy==1.24.4 \
    scipy==1.11.4 \
    pandas==2.1.3 \
    scikit-learn==1.3.2

# Install remaining dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Verify CUDA installation
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Copy the entire project
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /workspace/output /workspace/logs /workspace/cache /workspace/data /workspace/configs

# Set permissions for scripts
RUN chmod +x train.py && \
    chmod +x setup.py && \
    chmod +x scripts/manage_models.py && \
    chmod +x scripts/run_tests.py && \
    chmod +x scripts/docker_setup.sh

# Create a non-root user for security
RUN useradd -m -u 1000 trainer && \
    chown -R trainer:trainer /workspace

# Switch to non-root user
USER trainer

# Set environment variables for Hugging Face and caching
ENV HF_HOME=/workspace/cache
ENV TRANSFORMERS_CACHE=/workspace/cache
ENV HF_DATASETS_CACHE=/workspace/cache
ENV TORCH_HOME=/workspace/cache
ENV CUDA_LAUNCH_BLOCKING=1

# Verify installation by running a quick test
RUN python -c "from src.data_handler import ConversationDataHandler; print('✅ Data handler import successful')" && \
    python -c "from src.model_setup import ModelSetup; print('✅ Model setup import successful')" && \
    python -c "from src.training import CodeLlamaTrainer; print('✅ Training module import successful')" && \
    python -c "from src.utils import validate_model_output; print('✅ Utils import successful')"

# Expose ports
EXPOSE 6006  # TensorBoard
EXPOSE 8888  # Jupyter (if used)

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command with helpful information
CMD ["bash", "-c", "echo '🚀 CodeLlama Fine-tuning Pipeline Container Ready!'; echo ''; echo 'Available commands:'; echo '  python train.py --help'; echo '  python scripts/manage_models.py --help'; echo '  python setup.py --help'; echo ''; echo 'Quick start:'; echo '  python train.py --create-sample-data'; echo '  python train.py --model-config configs/model_configs/codellama_7b.yaml --train-data data/train.yaml'; echo ''; echo 'GPU Status:'; python -c \"import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None\"; echo ''; echo 'Container is ready. Use docker exec to run commands or start training.'; tail -f /dev/null"]
