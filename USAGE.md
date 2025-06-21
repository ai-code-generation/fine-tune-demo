# Advanced Usage Guide

This guide covers advanced usage scenarios for the CodeLlama fine-tuning pipeline.

## Table of Contents

1. [Custom Training Data](#custom-training-data)
2. [Model Configuration](#model-configuration)
3. [LoRA Tuning](#lora-tuning)
4. [Training Strategies](#training-strategies)
5. [Model Evaluation](#model-evaluation)
6. [Production Deployment](#production-deployment)
7. [Troubleshooting](#troubleshooting)

## Custom Training Data

### Data Format Requirements

Your YAML training data must follow this exact format:

```yaml
messages:
  - role: system
    content: "System prompt here"
  - role: user
    content: "User question or task"
  - role: assistant
    content: "Assistant response with code or explanation"
---
messages:
  - role: system
    content: "Another system prompt"
  - role: user
    content: "Another user question"
  - role: assistant
    content: "Another assistant response"
```

### Best Practices for Training Data

1. **Consistent System Prompts**: Use consistent system prompts for similar tasks
2. **Code Quality**: Ensure all code examples are syntactically correct
3. **Diverse Examples**: Include various programming languages and scenarios
4. **Balanced Dataset**: Mix simple and complex examples
5. **Clear Instructions**: Make user requests clear and specific

### Data Validation

Always validate your data before training:

```bash
python train.py --train-data your_data.yaml --validate-data
```

## Model Configuration

### CodeLlama 7B vs 13B

**CodeLlama 7B** (Recommended for most use cases):
- Faster training and inference
- Lower memory requirements
- Good performance for most coding tasks
- Suitable for single GPU training

**CodeLlama 13B** (For advanced use cases):
- Better performance on complex tasks
- Higher memory requirements
- Slower training and inference
- May require multiple GPUs or gradient accumulation

### Memory Optimization

For limited GPU memory, adjust these settings in your model config:

```yaml
training:
  batch_size: 1                    # Reduce batch size
  gradient_accumulation_steps: 8   # Increase accumulation
  
quantization:
  load_in_4bit: true              # Enable 4-bit quantization
  
memory_optimization:
  gradient_checkpointing: true     # Enable gradient checkpointing
  fp16: true                      # Use mixed precision
```

## LoRA Tuning

### LoRA Parameters Explained

- **r (rank)**: Controls the number of trainable parameters
  - Lower (4-8): Faster, less memory, may underfit
  - Medium (16-32): Good balance
  - Higher (64+): More parameters, may overfit

- **lora_alpha**: Scaling factor for LoRA weights
  - Usually 2x the rank value
  - Higher values = stronger adaptation

- **target_modules**: Which model layers to adapt
  - More modules = more parameters but better adaptation
  - Start with attention layers: `["q_proj", "v_proj"]`

### LoRA Configuration Examples

**Minimal (for testing):**
```yaml
lora:
  r: 4
  lora_alpha: 8
  target_modules: ["q_proj", "v_proj"]
```

**Balanced (recommended):**
```yaml
lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

**Aggressive (for complex tasks):**
```yaml
lora:
  r: 64
  lora_alpha: 128
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

## Training Strategies

### Learning Rate Scheduling

Different learning rate strategies for different scenarios:

**Conservative (stable training):**
```yaml
training:
  learning_rate: 1e-5
  lr_scheduler_type: "linear"
  warmup_steps: 100
```

**Aggressive (faster convergence):**
```yaml
training:
  learning_rate: 2e-4
  lr_scheduler_type: "cosine"
  warmup_steps: 50
```

### Multi-GPU Training

For multiple GPUs, use these settings:

```yaml
training:
  batch_size: 2                    # Per device
  gradient_accumulation_steps: 4   # Effective batch size = 2 * num_gpus * 4
  dataloader_num_workers: 4        # Increase for faster data loading
```

Run with:
```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py --model-config your_config.yaml
```

### Resuming Training

To resume from a checkpoint:

```bash
python train.py --model-config configs/model_configs/codellama_7b.yaml \
                --train-data data/train.yaml \
                --resume-from-checkpoint output/checkpoint-500
```

## Model Evaluation

### Automatic Evaluation

The pipeline automatically evaluates on validation data during training. Monitor:

- **Training Loss**: Should decrease steadily
- **Validation Loss**: Should decrease without overfitting
- **Learning Rate**: Should follow the schedule

### Manual Testing

Test your model interactively:

```bash
python scripts/manage_models.py test --model-dir output/your_model \
                                     --prompt "Write a Python function to reverse a string"
```

### Custom Evaluation Scripts

Create custom evaluation scripts for your specific use case:

```python
from src.utils import load_finetuned_model, generate_text

# Load model
model, tokenizer = load_finetuned_model("output/your_model")

# Test prompts
test_prompts = [
    "Write a function to sort a list",
    "Create a class for a binary tree",
    "Implement bubble sort algorithm"
]

for prompt in test_prompts:
    response = generate_text(model, tokenizer, prompt, max_length=512)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print("-" * 50)
```

## Production Deployment

### Model Export

Copy your trained model to a production directory:

```bash
python scripts/manage_models.py copy --source output/checkpoint-final \
                                     --output /production/models \
                                     --name codellama-v1.0
```

### Inference Server

Create a simple inference server:

```python
from flask import Flask, request, jsonify
from src.utils import load_finetuned_model, generate_text

app = Flask(__name__)

# Load model once at startup
model, tokenizer = load_finetuned_model("/production/models/codellama-v1.0")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 512)
    
    response = generate_text(model, tokenizer, prompt, max_length=max_length)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Docker Deployment

Build a production Docker image:

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Copy your trained model
COPY /production/models/codellama-v1.0 /app/model

# Copy inference code
COPY inference_server.py /app/

# Install dependencies
RUN pip install torch transformers peft flask

# Run server
CMD ["python", "/app/inference_server.py"]
```

## Troubleshooting

### Common Issues

**Out of Memory Errors:**
- Reduce batch size
- Enable gradient checkpointing
- Use 4-bit quantization
- Reduce sequence length

**Slow Training:**
- Increase batch size (if memory allows)
- Use multiple GPUs
- Reduce logging frequency
- Use faster storage (SSD)

**Poor Model Performance:**
- Increase LoRA rank
- Add more target modules
- Improve training data quality
- Train for more epochs

**Model Not Learning:**
- Check learning rate (may be too low/high)
- Verify data format
- Check for data leakage
- Increase model capacity

### Debug Mode

Run training with debug logging:

```bash
python train.py --log-level DEBUG --model-config your_config.yaml
```

### Memory Profiling

Monitor GPU memory usage:

```bash
watch -n 1 nvidia-smi
```

### Performance Monitoring

Use TensorBoard to monitor training:

```bash
tensorboard --logdir output --host 0.0.0.0 --port 6006
```

## Advanced Features

### Custom Loss Functions

Modify the training script to use custom loss functions for specific tasks.

### Multi-Task Learning

Train on multiple tasks simultaneously by mixing different types of training data.

### Continual Learning

Fine-tune an already fine-tuned model for additional tasks without forgetting previous knowledge.

### Model Merging

Combine multiple LoRA adapters for different capabilities.

For more advanced customizations, refer to the source code in the `src/` directory.
