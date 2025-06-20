# Usage Guide

This guide provides detailed instructions on how to use the NeMo Fine-Tuning Pipeline.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd fine-tune-pipeline

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 2. Prepare Your Data

Create your training data in YAML format:

```yaml
---
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
  # Additional training examples...
```

### 3. Run the Pipeline

```bash
# Basic usage
python scripts/run_pipeline.py \
  --model-type llama2 \
  --model-size 7b \
  --train-file data/train.yaml \
  --val-file data/val.yaml \
  --test-file data/test.yaml

# With custom configuration
python scripts/run_pipeline.py \
  --model-type llama3 \
  --model-size 8b \
  --config configs/training/custom_config.yaml \
  --train-file data/train.yaml \
  --base-model meta-llama/Meta-Llama-3-8B \
  --max-epochs 5 \
  --gpus 2
```

## Detailed Usage

### Data Preparation

#### YAML Format Requirements

Your training data must follow this structure:

```yaml
messages:
  - role: system
    content: "System prompt here"
  - role: user  
    content: "User input here"
  - role: assistant
    content: "Expected response here"
```

#### Data Processing

```python
from src.data import DataProcessor, InstructionFormatter

# Process YAML data
processor = DataProcessor("llama2")
conversations = processor.load_yaml_data("data/train.yaml")
filtered_conversations = processor.filter_conversations(conversations)

# Format for training
formatter = InstructionFormatter("llama2")
formatted_data = formatter.format_for_training(filtered_conversations)
```

### Configuration Management

#### Using Configuration Templates

```python
from src.config import ConfigManager

# Load default configuration for model type
config_manager = ConfigManager("llama2")
config = config_manager.load_config()

# Create custom configuration
config_manager.create_config_from_template(
    "configs/my_custom_config.yaml",
    overrides={
        "training.max_epochs": 5,
        "lora.rank": 64,
        "data.batch_size": 2
    }
)
```

#### Configuration Validation

```python
from src.config import ConfigValidator

validator = ConfigValidator()
is_valid = validator.validate_config(config)

if not is_valid:
    report = validator.get_validation_report()
    print("Validation errors:", report["errors"])
```

### Model Training

#### Basic Training

```python
from src.training import NeMoTrainer

# Initialize trainer
trainer = NeMoTrainer("llama2", "7b")

# Setup components
trainer.setup_model(base_model_path="meta-llama/Llama-2-7b-hf")
trainer.setup_data("data/train.yaml", "data/val.yaml")
trainer.setup_trainer(max_epochs=3, gpus=1)

# Start training
trainer.train()

# Save model
trainer.save_model("checkpoints/my_model.nemo")
```

#### Advanced Training Options

```python
# Custom LoRA configuration
from src.training import LoRAConfig

lora_config = LoRAConfig("llama2")
lora_config.load_config()
lora_config.update_config({
    "lora.rank": 64,
    "lora.alpha": 128,
    "lora.dropout": 0.05
})

# Training with custom settings
trainer.setup_trainer(
    output_dir="checkpoints",
    log_dir="logs",
    max_epochs=5,
    gpus=2,
    precision="bf16",
    accumulate_grad_batches=8
)
```

### Model Evaluation

```python
from src.evaluation import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator("llama2", "checkpoints/my_model.nemo")

# Load model and tokenizer
evaluator.load_model()
evaluator.load_tokenizer()

# Run evaluation
results = evaluator.evaluate_on_file(
    "data/test.yaml",
    output_file="logs/evaluation_results.json",
    max_examples=100
)

print("Evaluation results:", results)
```

### Model Deployment

```python
from src.deployment import ModelDeployer

# Initialize deployer
deployer = ModelDeployer("llama2", "7b")

# Deploy model
deployment_path = deployer.deploy_model(
    checkpoint_path="checkpoints/my_model.nemo",
    model_name="my_finetuned_model",
    copy_tokenizer=True,
    create_config=True,
    create_readme=True
)

print(f"Model deployed to: {deployment_path}")
```

### Model Format Conversion

```python
from src.deployment import ModelConverter

converter = ModelConverter("llama2")

# Convert to HuggingFace format
converter.convert_nemo_to_hf(
    "deploy/my_model/model.nemo",
    "deploy/my_model_hf"
)

# Create inference script
converter.create_inference_script(
    "deploy/my_model",
    "deploy/my_model/inference.py"
)
```

## Command Line Interface

### Main Pipeline Script

```bash
python scripts/run_pipeline.py [OPTIONS]
```

#### Required Arguments

- `--model-type`: Model type (llama2, llama3, codellama)
- `--model-size`: Model size (7b, 8b, 13b, 34b, 70b)
- `--train-file`: Path to training data YAML file

#### Optional Arguments

- `--config`: Path to custom configuration file
- `--val-file`: Path to validation data YAML file
- `--test-file`: Path to test data YAML file
- `--base-model`: Base model path or HuggingFace model name
- `--checkpoint`: Path to existing checkpoint to resume from
- `--max-epochs`: Maximum number of training epochs (default: 3)
- `--gpus`: Number of GPUs to use (default: 1)
- `--output-dir`: Directory to save checkpoints (default: checkpoints)
- `--skip-evaluation`: Skip evaluation phase
- `--skip-deployment`: Skip deployment phase
- `--training-only`: Run only training phase
- `--max-eval-examples`: Maximum examples for evaluation

### Examples

#### Train LLaMA2 7B

```bash
python scripts/run_pipeline.py \
  --model-type llama2 \
  --model-size 7b \
  --train-file data/train.yaml \
  --val-file data/val.yaml \
  --test-file data/test.yaml \
  --base-model meta-llama/Llama-2-7b-hf \
  --max-epochs 3 \
  --gpus 1
```

#### Train LLaMA3 8B with Custom Config

```bash
python scripts/run_pipeline.py \
  --model-type llama3 \
  --model-size 8b \
  --config configs/training/llama3_custom.yaml \
  --train-file data/train.yaml \
  --base-model meta-llama/Meta-Llama-3-8B \
  --max-epochs 5 \
  --gpus 2
```

#### Train CodeLlama for Code Generation

```bash
python scripts/run_pipeline.py \
  --model-type codellama \
  --model-size 7b \
  --train-file data/code_train.yaml \
  --val-file data/code_val.yaml \
  --test-file data/code_test.yaml \
  --base-model codellama/CodeLlama-7b-hf \
  --max-epochs 5 \
  --gpus 1
```

#### Training Only (Skip Evaluation and Deployment)

```bash
python scripts/run_pipeline.py \
  --model-type llama2 \
  --model-size 7b \
  --train-file data/train.yaml \
  --training-only \
  --max-epochs 3
```

## Configuration Files

### Model-Specific Configurations

The pipeline includes optimized configurations for each model type:

- `configs/models/llama2_7b.yaml`: LLaMA2 7B model configuration
- `configs/models/llama3_8b.yaml`: LLaMA3 8B model configuration  
- `configs/models/codellama_7b.yaml`: CodeLlama 7B model configuration

### Training Configurations

- `configs/training/llama2_training.yaml`: LLaMA2 training settings
- `configs/training/llama3_training.yaml`: LLaMA3 training settings
- `configs/training/codellama_training.yaml`: CodeLlama training settings

### LoRA Configurations

- `configs/lora/llama2_lora.yaml`: LLaMA2 LoRA settings
- `configs/lora/llama3_lora.yaml`: LLaMA3 LoRA settings
- `configs/lora/codellama_lora.yaml`: CodeLlama LoRA settings

## Best Practices

### Data Preparation

1. **Quality over Quantity**: Focus on high-quality, diverse examples
2. **Consistent Format**: Ensure all conversations follow the same format
3. **Balanced Dataset**: Include varied instruction types and lengths
4. **Validation Split**: Reserve 10-20% of data for validation

### Training Configuration

1. **Start Small**: Begin with smaller models and shorter training
2. **Monitor Metrics**: Watch for overfitting and adjust accordingly
3. **LoRA Tuning**: Start with rank 16-32, adjust based on results
4. **Learning Rate**: Use conservative learning rates (1e-4 to 2e-4)

### Hardware Optimization

1. **GPU Memory**: Adjust batch size based on available memory
2. **Multi-GPU**: Use DDP strategy for multiple GPUs
3. **Precision**: Use bf16 or fp16 for memory efficiency
4. **Gradient Accumulation**: Increase for effective larger batch sizes

### Evaluation

1. **Multiple Metrics**: Use diverse evaluation metrics
2. **Test Set**: Keep test set separate and representative
3. **Human Evaluation**: Supplement automated metrics with human review
4. **Comparison**: Compare against baseline models

## Troubleshooting

### Common Issues

#### Out of Memory Errors

```bash
# Reduce batch size
--config configs/low_memory_config.yaml

# Or modify existing config
python -c "
from src.config import ConfigManager
cm = ConfigManager('llama2')
config = cm.load_config()
cm.set_training_params(batch_size=1)
cm.save_config('configs/low_memory.yaml')
"
```

#### Slow Training

```bash
# Increase batch size and use gradient accumulation
python scripts/run_pipeline.py \
  --model-type llama2 \
  --model-size 7b \
  --train-file data/train.yaml \
  --config configs/fast_training.yaml
```

#### Poor Model Performance

1. Check data quality and format
2. Increase training epochs
3. Adjust LoRA rank and alpha
4. Modify learning rate
5. Add more diverse training data

### Getting Help

1. Check the logs in `logs/` directory
2. Validate your configuration with `ConfigValidator`
3. Review the model info with `ModelFactory.get_model_info()`
4. Check GPU memory usage and adjust batch size accordingly
