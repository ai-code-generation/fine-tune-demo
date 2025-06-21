# CodeLlama Fine-tuning Pipeline

A comprehensive pipeline for fine-tuning CodeLlama models (7B and 13B) using Hugging Face Transformers and PEFT (LoRA).

## Features

- Support for CodeLlama 7B and 13B models
- YAML-based training data format with conversation structure
- LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning
- Docker containerization for easy deployment
- Automated model output management
- Instruction-tuning format conversion

## Project Structure

```
fine-tune-pipeline/
├── data/                    # Training data directory
│   ├── train.yaml          # Training conversations
│   └── validation.yaml     # Validation conversations
├── configs/                 # Configuration files
│   ├── model_configs/      # Model-specific configurations
│   └── lora_configs/       # LoRA configurations
├── src/                    # Source code
│   ├── data_handler.py     # YAML data processing
│   ├── model_setup.py      # Model and tokenizer setup
│   ├── training.py         # Training logic
│   └── utils.py           # Utility functions
├── output/                 # Fine-tuned model outputs
├── scripts/               # Training and utility scripts
├── tests/                 # Test files
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker compose setup
├── requirements.txt       # Python dependencies
└── train.py              # Main training script
```

## Quick Start

### Option 1: Docker (Recommended)

**Prerequisites:**
- Docker with GPU support (nvidia-docker2 or Docker 19.03+)
- NVIDIA drivers installed on host system

1. **Build the Docker image with all dependencies:**
   ```bash
   ./scripts/build_docker.sh
   ```

2. **Setup the environment:**
   ```bash
   ./scripts/docker_setup.sh setup
   ```

3. **Verify installation:**
   ```bash
   docker exec -it codellama-finetune python scripts/verify_installation.py
   ```

4. **Create sample training data:**
   ```bash
   ./scripts/docker_setup.sh sample-data
   ```

5. **Start training:**
   ```bash
   ./scripts/docker_setup.sh train
   ```

### Option 2: Local Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create sample data:**
   ```bash
   python train.py --create-sample-data
   ```

3. **Run training:**
   ```bash
   python train.py --model-config configs/model_configs/codellama_7b.yaml \
                   --train-data data/train.yaml \
                   --eval-data data/validation.yaml
   ```

## Training Data Format

```yaml
messages:
  - role: system
    content: "You are an expert test automation assistant using SWTBot."
  - role: user
    content: "Task: Create and build an S32K144 project..."
  - role: assistant
    content: |
      ```java
      bot.menu("File")...
      ```
---
messages:
  - role: system
    content: "You are a helpful coding assistant."
  - role: user
    content: "How do I implement a binary search?"
  - role: assistant
    content: |
      ```python
      def binary_search(arr, target):
          left, right = 0, len(arr) - 1
          ...
      ```
```

## Detailed Usage

### Training Configuration

The pipeline supports two model sizes:

- **CodeLlama 7B**: `configs/model_configs/codellama_7b.yaml`
- **CodeLlama 13B**: `configs/model_configs/codellama_13b.yaml`

LoRA configuration can be customized in `configs/lora_configs/lora_default.yaml`:

```yaml
lora:
  r: 16                    # LoRA rank (higher = more parameters)
  lora_alpha: 32          # LoRA scaling factor
  lora_dropout: 0.1       # Dropout rate
  target_modules:         # Modules to apply LoRA to
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
```

### Training Commands

**Basic training:**
```bash
python train.py --model-config configs/model_configs/codellama_7b.yaml \
                --train-data data/train.yaml \
                --eval-data data/validation.yaml
```

**Training with validation:**
```bash
python train.py --model-config configs/model_configs/codellama_7b.yaml \
                --train-data data/train.yaml \
                --eval-data data/validation.yaml \
                --validate-data
```

**Resume from checkpoint:**
```bash
python train.py --model-config configs/model_configs/codellama_7b.yaml \
                --train-data data/train.yaml \
                --resume-from-checkpoint output/checkpoint-500
```

### Model Management

**List available models:**
```bash
python scripts/manage_models.py list --output-dir output
```

**Validate a model:**
```bash
python scripts/manage_models.py validate --model-dir output/model_name
```

**Test model inference:**
```bash
python scripts/manage_models.py test --model-dir output/model_name \
                                     --prompt "Write a Python function to sort a list"
```

**Copy model to final location:**
```bash
python scripts/manage_models.py copy --source output/checkpoint-final \
                                     --output /path/to/final/models \
                                     --name my-codellama-model
```

### Docker Usage

**Build image with all dependencies:**
```bash
./scripts/build_docker.sh              # Full build and test
./scripts/build_docker.sh --no-cache   # Clean build
./scripts/build_docker.sh --build-only # Build without testing
```

**Complete setup:**
```bash
./scripts/docker_setup.sh setup
```

**Start container:**
```bash
./scripts/docker_setup.sh start
```

**Verify installation:**
```bash
docker exec -it codellama-finetune python scripts/verify_installation.py
```

**Access container shell:**
```bash
./scripts/docker_setup.sh shell
```

**Monitor training with TensorBoard:**
```bash
./scripts/docker_setup.sh tensorboard
# Access at http://localhost:6006
```

**Start Jupyter for experimentation:**
```bash
./scripts/docker_setup.sh jupyter
# Access at http://localhost:8888
```

**Stop and cleanup:**
```bash
./scripts/docker_setup.sh cleanup
```

### Docker Image Features

The Docker image includes:
- ✅ **CUDA 12.1** with full development toolkit
- ✅ **PyTorch 2.1.0** with CUDA support
- ✅ **All ML dependencies** (Transformers, PEFT, Datasets, etc.)
- ✅ **System utilities** (vim, htop, tmux, etc.)
- ✅ **Automatic verification** of installation
- ✅ **GPU memory optimization** settings
- ✅ **Health checks** for CUDA availability

### Testing

**Run all tests:**
```bash
python scripts/run_tests.py --all
```

**Quick tests only:**
```bash
python scripts/run_tests.py --quick
```

**Integration tests:**
```bash
python scripts/run_tests.py --integration
```
