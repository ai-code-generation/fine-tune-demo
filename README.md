# NeMo Fine-tuning Pipeline for CodeLlama and Llama3

A comprehensive fine-tuning pipeline for CodeLlama and Llama3 models using NVIDIA's NeMo Framework with LoRA (Low-Rank Adaptation) technique.

## Features

- **Multi-model Support**: Fine-tune CodeLlama (7B, 13B, 34B) and Llama3 (8B, 70B) models
- **YAML Data Format**: Support for conversational training data in YAML format
- **LoRA Fine-tuning**: Parameter-efficient fine-tuning using Low-Rank Adaptation
- **Automated Pipeline**: Complete pipeline from data preprocessing to model evaluation
- **Hardware Optimization**: Configurations optimized for different GPU setups
- **Comprehensive Evaluation**: Multiple evaluation metrics including perplexity and code quality

## Requirements

### Hardware Requirements

| Model | Min GPUs | Min GPU Memory | Recommended GPUs | Recommended GPU Memory |
|-------|----------|----------------|------------------|----------------------|
| CodeLlama-7B | 2 | 16GB | 4 | 24GB |
| CodeLlama-13B | 4 | 24GB | 8 | 40GB |
| CodeLlama-34B | 8 | 40GB | 8 | 80GB |
| Llama3-8B | 2 | 16GB | 4 | 24GB |
| Llama3-70B | 16 | 80GB | 16 | 80GB |

### Software Requirements

- NVIDIA GPU with CUDA support
- Docker (recommended) or local NeMo Framework installation
- Python 3.8+
- NVIDIA NeMo Framework

## Quick Start

### 1. Setup Environment

#### Option A: Using Docker Compose (Recommended)

```bash
# Easy startup with the provided script
./docker_start.sh

# Or manually with docker-compose
docker-compose up nemo-finetuning

# Access Jupyter Lab at http://localhost:8888
# Or connect to the container to run scripts
docker-compose exec nemo-finetuning bash
```

#### Option B: Using NeMo Container Directly

```bash
# Pull the NeMo container
docker pull nvcr.io/nvidia/nemo:25.04.01.llama_nemotron_nano_vl

# Run the container
docker run --gpus all --shm-size=8g --net=host --ulimit memlock=-1 \
    --rm -it -v ${PWD}:/workspace -w /workspace \
    nvcr.io/nvidia/nemo:25.04.01.llama_nemotron_nano_vl
```

#### Option C: Local Installation

```bash
# Clone this repository
git clone <repository-url>
cd simple_nemo

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### 2. Prepare Your Data

Create a YAML file with your training conversations:

```yaml
messages:
  - role: system
    content: You are an expert test automation assistant using SWTBot.
  - role: user
    content: "Task: Create and build an S32K144 project..."
  - role: assistant
    content: |
      ```java
      bot.menu("File").click();
      bot.menu("New").click();
      // ... more code
      ```
---
messages:
  - role: user
    content: "How do I create a new Eclipse project?"
  - role: assistant
    content: |
      ```java
      // Create new project
      bot.menu("File").menu("New").menu("Project...").click();
      ```
```

### 3. Run Fine-tuning

```bash
# Check hardware requirements
python finetune_pipeline.py --model codellama-13b --check-hardware

# Run the complete pipeline
python finetune_pipeline.py \
    --model codellama-13b \
    --data your_training_data.yaml \
    --output-dir ./outputs \
    --max-steps 100 \
    --hf-token your_huggingface_token
```

### 4. Evaluate the Model

```bash
# Run comprehensive evaluation
python evaluate_model.py \
    --model-path ./outputs/models/codellama-13b_merged.nemo \
    --model-name codellama-13b \
    --test-data ./outputs/data/codellama-13b_training_data.val.jsonl \
    --eval-type comprehensive
```

## Usage Examples

### Data Preprocessing Only

```bash
python data_preprocessing.py \
    --input your_data.yaml \
    --output processed_data \
    --validation-split 0.1
```

### Custom Configuration

You can modify the configuration files in the `configs/` directory to customize:
- Model architecture parameters
- Training hyperparameters
- LoRA adapter settings
- Hardware allocation

### Supported Models

- `codellama-7b`: CodeLlama 7B model
- `codellama-13b`: CodeLlama 13B model  
- `codellama-34b`: CodeLlama 34B model
- `llama3-8b`: Llama3 8B model
- `llama3-70b`: Llama3 70B model

## Configuration

### Model Configurations

Each model has optimized configurations in the `configs/` directory:
- `codellama_13b_lora_config.yaml`: Configuration for CodeLlama-13B
- `llama3_8b_lora_config.yaml`: Configuration for Llama3-8B
- `llama3_70b_lora_config.yaml`: Configuration for Llama3-70B

### Key Parameters

- **LoRA Adapter Dimension**: Controls the rank of adaptation (16-64)
- **Learning Rate**: Optimized per model size (1e-4 to 5e-5)
- **Batch Size**: Adjusted based on GPU memory
- **Sequence Length**: Maximum input sequence length (2048-8192)

## Pipeline Steps

1. **Model Download**: Downloads the base model from Hugging Face
2. **Format Conversion**: Converts HuggingFace model to NeMo format
3. **Data Preprocessing**: Converts YAML conversations to JSONL format
4. **Configuration Generation**: Creates optimized training configuration
5. **LoRA Training**: Runs parameter-efficient fine-tuning
6. **Weight Merging**: Merges LoRA weights with base model
7. **Evaluation**: Comprehensive model evaluation

## Evaluation Metrics

- **Perplexity**: Language modeling performance
- **Code Quality**: Syntax validity, structure analysis
- **Generation Quality**: Length, patterns, completeness

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Model Download Fails**: Check Hugging Face token and internet connection
3. **Training Crashes**: Verify hardware requirements and reduce model parallelism

### Performance Optimization

- Use mixed precision training (bf16)
- Enable gradient checkpointing for large models
- Optimize tensor and pipeline parallelism settings
- Use Flash Attention for memory efficiency

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- NVIDIA NeMo Framework team
- Hugging Face for model hosting
- Meta AI for CodeLlama and Llama3 models
