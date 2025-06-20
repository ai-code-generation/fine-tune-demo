# NeMo Fine-Tuning Pipeline

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![NeMo](https://img.shields.io/badge/NeMo-1.20%2B-green.svg)](https://github.com/NVIDIA/NeMo)

A comprehensive, production-ready pipeline for fine-tuning LLaMA2, LLaMA3, and CodeLlama models using NVIDIA NeMo with optimized LoRA configurations.

## 🚀 Features

- **Multi-Model Support**: LLaMA2, LLaMA3, and CodeLlama with model-specific optimizations
- **YAML Training Data**: Simple, human-readable training data format
- **Optimized LoRA**: Model-specific LoRA configurations for maximum efficiency
- **Instruction Tuning**: Automatic format adaptation for each model type
- **Automated Pipeline**: Complete training-to-deployment workflow
- **Comprehensive Evaluation**: Multiple metrics including code-specific evaluations
- **Easy Configuration**: Template-based configuration management
- **Production Ready**: Deployment, conversion, and inference utilities

## 📁 Project Structure

```
fine-tune-pipeline/
├── 📂 src/                    # Core pipeline modules
│   ├── 📂 data/              # Data processing utilities
│   ├── 📂 models/            # Model configuration and factory
│   ├── 📂 training/          # Training and LoRA configuration
│   ├── 📂 evaluation/        # Evaluation metrics and evaluator
│   ├── 📂 deployment/        # Model deployment and conversion
│   └── 📂 config/            # Configuration management
├── 📂 configs/               # Configuration templates
│   ├── 📂 models/            # Model-specific configs (LLaMA2/3, CodeLlama)
│   ├── 📂 training/          # Training configurations
│   └── 📂 lora/             # LoRA configurations
├── 📂 scripts/              # Main pipeline scripts
├── 📂 examples/             # Usage examples and tutorials
├── 📂 tests/                # Unit tests
├── 📂 docs/                 # Documentation
├── 📂 data/                 # Training data
├── 📂 logs/                 # Training logs
├── 📂 checkpoints/          # Model checkpoints
└── 📂 deploy/              # Deployed models
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/ai-code-generation/fine-tune-demo.git
cd fine-tune-demo
git checkout pipeline-fine-tune

# Setup Docker environment (one-time setup)
chmod +x scripts/setup_docker_environment.sh
./scripts/setup_docker_environment.sh

# Run fine-tuning in Docker with persistent outputs
chmod +x scripts/docker_run.sh
./scripts/docker_run.sh run
```

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/ai-code-generation/fine-tune-demo.git
cd fine-tune-demo
git checkout pipeline-fine-tune

# Install dependencies
pip install -r requirements.txt

# Validate setup
python scripts/validate_setup.py
```

### 2. Prepare Training Data

Create your training data in YAML format:

```yaml
---
messages:
  - role: system
    content: "You are an expert test automation assistant using SWTBot."
  - role: user
    content: "Task: Create and build an S32K144 project in Eclipse IDE."
  - role: assistant
    content: |
      ```java
      bot.menu("File").menu("New").menu("Project...").click();
      bot.shell("New Project").activate();
      bot.tree().expandNode("C/C++").select("C Project");
      bot.button("Next >").click();
      ```
---
messages:
  # Additional training examples...
```

### 3. Run the Pipeline

#### Docker (Recommended)
```bash
# Basic S32K144 fine-tuning (outputs saved to ./outputs/)
./scripts/docker_run.sh run

# Advanced training with custom parameters
./scripts/docker_run.sh --model-type llama3 --model-size 8b --epochs 5 --gpus 2 run

# Monitor training progress
./scripts/docker_run.sh logs

# Access Jupyter Lab for analysis
./scripts/docker_run.sh jupyter  # http://localhost:8888

# View TensorBoard
./scripts/docker_run.sh tensorboard  # http://localhost:6006
```

#### Local Installation
```bash
# Basic training
python scripts/run_pipeline.py \
  --model-type llama2 \
  --model-size 7b \
  --train-file data/train.yaml \
  --val-file data/val.yaml \
  --test-file data/test.yaml

# Advanced training with custom configuration
python scripts/run_pipeline.py \
  --model-type llama3 \
  --model-size 8b \
  --config configs/training/llama3_training.yaml \
  --base-model meta-llama/Meta-Llama-3-8B \
  --max-epochs 5 \
  --gpus 2
```

## 🎯 Supported Models

| Model | Sizes | Context Length | Specialization |
|-------|-------|----------------|----------------|
| **LLaMA2** | 7B, 13B, 70B | 4K tokens | General instruction-following |
| **LLaMA3** | 8B, 70B | 8K tokens | Enhanced reasoning, GQA |
| **CodeLlama** | 7B, 13B, 34B | 16K tokens | Code generation and understanding |

## ⚙️ Configuration

### Model-Specific Optimizations

Each model type includes optimized configurations:

- **LLaMA2**: Balanced LoRA (rank=32, α=64) for general tasks
- **LLaMA3**: Higher capacity LoRA (rank=64, α=128) for complex reasoning
- **CodeLlama**: Focused LoRA (rank=16, α=32) targeting later layers for code

### Hardware Optimization

The pipeline automatically optimizes for your hardware:

```python
# Automatic optimization based on GPU memory
config_manager.optimize_for_hardware(gpu_memory_gb=24, num_gpus=2)
```

## 📊 Evaluation Metrics

- **General**: Perplexity, BLEU, ROUGE, Exact Match
- **Code-Specific**: Code BLEU, Syntax Validity, Function Accuracy
- **Model-Specific**: Optimized evaluation for each model type

## 🚀 Deployment

### Docker Deployment (Persistent Outputs)

With Docker, all outputs are automatically saved to your host machine:

```bash
# After training, find your models in:
ls outputs/checkpoints/    # Model checkpoints (.nemo files)
ls outputs/deploy/         # Deployed models with inference scripts
ls outputs/logs/           # Training logs and TensorBoard data

# Models include:
# - model.nemo (trained model)
# - tokenizer/ (tokenizer files)
# - config.yaml (model configuration)
# - inference.py (ready-to-use inference script)
# - README.md (usage instructions)
```

### Local Deployment

```bash
# Models are automatically deployed to deploy/ folder
# Includes: model files, tokenizer, configs, README, inference scripts
```

### Supported Formats
- NeMo (.nemo)
- HuggingFace (PyTorch)
- ONNX (optimized inference)
- Quantized models (INT8, FP16)

## 🐳 Docker Support

The pipeline includes comprehensive Docker support for easy deployment and reproducible training:

### Features
- **GPU Support**: NVIDIA Docker runtime with CUDA 11.8
- **Persistent Outputs**: All models and logs saved to host machine
- **Multi-Service**: Integrated Jupyter Lab and TensorBoard
- **Production Ready**: Optimized for both development and production use

### Quick Commands
```bash
# Setup (one-time)
./scripts/setup_docker_environment.sh

# Build and run
./scripts/docker_run.sh run

# Monitor training
./scripts/docker_run.sh logs

# Development tools
./scripts/docker_run.sh jupyter     # Jupyter Lab
./scripts/docker_run.sh tensorboard # TensorBoard
./scripts/docker_run.sh shell       # Interactive shell

# Management
./scripts/docker_run.sh stop        # Stop containers
./scripts/docker_run.sh clean       # Clean up
```

### Output Structure
```
outputs/
├── checkpoints/    # Model checkpoints (.nemo files)
├── logs/          # Training logs and TensorBoard data
├── deploy/        # Deployed models with inference scripts
└── cache/         # Model cache (HuggingFace, NeMo, etc.)
```

For detailed Docker documentation, see [`docker/README.md`](docker/README.md).

## 📚 Examples

- **Basic Training**: `examples/basic_training.py`
- **Advanced Training**: `examples/advanced_training.py`
- **CodeLlama Specialization**: `examples/codellama_example.py`

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python tests/test_data_processing.py
python tests/test_config.py

# Validate setup
python scripts/validate_setup.py
```

## 📖 Documentation

- **[Usage Guide](docs/USAGE.md)**: Comprehensive usage instructions
- **[API Reference](docs/API.md)**: Complete API documentation
- **[Configuration Guide](configs/README.md)**: Configuration management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) for the training framework
- [Meta AI](https://ai.meta.com/) for the LLaMA model family
- [Hugging Face](https://huggingface.co/) for model hosting and tokenizers

## 📞 Support

- 📖 Check the [documentation](docs/)
- 🐛 Report issues on [GitHub Issues](https://github.com/ai-code-generation/fine-tune-demo/issues)
- 💬 Join discussions in [GitHub Discussions](https://github.com/ai-code-generation/fine-tune-demo/discussions)

---

**Ready to fine-tune your models?** 🚀 Start with `python scripts/validate_setup.py`
