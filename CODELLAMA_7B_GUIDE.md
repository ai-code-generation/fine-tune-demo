# 🎯 Simple NeMo 24.07 Fine-tuning Guide for CodeLlama-7B

This guide provides the simplest workflow for fine-tuning **CodeLlama-7B** with **reduced resource requirements** compared to the 13B model.

## 📋 Prerequisites

- Docker with GPU support
- HuggingFace token (for CodeLlama access)
- **At least 2 GPUs with 16GB+ VRAM each** (less than 13B requirements)

## 🚀 Quick Start (5 Simple Steps)

### Step 1: Get HuggingFace Token
```bash
# Get token from: https://huggingface.co/settings/tokens
# Request access to: https://huggingface.co/meta-llama/CodeLlama-7b-hf
export HF_TOKEN="your_token_here"
```

### Step 2: Start NeMo Container
```bash
cd /static-data/team_08/simple-nemo/fine-tune-demo

# Create cache directories
mkdir -p cache results models

# Run container with GPU access
docker run --gpus all \
  --shm-size=2g \
  --net=host \
  --ulimit memlock=-1 \
  --rm -it \
  -v ${PWD}:/workspace \
  -w /workspace \
  -v ${PWD}/results:/results \
  -v ${PWD}/cache:/root/.cache \
  -e HF_HOME=/workspace/cache/huggingface \
  -e TRANSFORMERS_CACHE=/workspace/cache/transformers \
  -e HF_DATASETS_CACHE=/workspace/cache/datasets \
  nvcr.io/nvidia/nemo:24.07 \
  bash
```

### Step 3: Check GPU Access (inside container)
```bash
# Verify GPU access
nvidia-smi

# Should show your GPUs with memory info
```

### Step 4: Run Fine-tuning for CodeLlama-7B (inside container)
```bash
# Set your HF token inside container
export HF_TOKEN="your_token_here"

# Run the CodeLlama-7B fine-tuning script
python simple_nemo_finetune_7b.py \
  --data example_training_data.yaml \
  --max-steps 50 \
  --hf-token $HF_TOKEN
```

### Step 5: Check Results (inside container)
```bash
# List results
ls -la /results/

# Check training logs
ls -la /results/checkpoints/

# View final model
ls -la /results/checkpoints/megatron_gpt_peft_lora_tuning.nemo
```

### Step 6: Test the Model (inside container)
```bash
# Run inference test
python /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_generate.py \
  model.restore_from_path=/results/checkpoints/megatron_gpt_peft_lora_tuning.nemo \
  trainer.devices=2 \
  model.tensor_model_parallel_size=1 \
  model.pipeline_model_parallel_size=1 \
  model.data.test_ds.file_names="[/workspace/validation.jsonl]" \
  model.data.test_ds.names="['codellama_7b_test']" \
  model.data.test_ds.global_batch_size=4 \
  model.data.test_ds.micro_batch_size=2 \
  model.data.test_ds.tokens_to_generate=50 \
  inference.greedy=True \
  model.data.test_ds.output_file_path_prefix=/results/codellama_7b_results \
  model.data.test_ds.write_predictions_to_file=True
```

## 🔧 CodeLlama-7B Optimizations

### **Reduced Resource Requirements**
- **GPUs**: 2 GPUs instead of 4 (for 13B)
- **Memory**: 16GB+ VRAM per GPU instead of 24GB+
- **Parallelism**: No tensor parallelism needed
- **Batch Size**: Larger batches possible due to smaller model

### **Optimized Settings**
```bash
# CodeLlama-7B specific settings:
trainer.devices=2                    # 2 GPUs instead of 4
model.micro_batch_size=2             # Larger micro batch
model.global_batch_size=16           # Larger global batch  
model.tensor_model_parallel_size=1   # No TP needed
model.optim.lr=2e-5                  # Slightly higher LR
torchrun --nproc_per_node=2          # 2 processes
```

## 📊 Expected Performance

### **Training Speed**
- **Faster per step** due to smaller model
- **Less memory usage** per GPU
- **Higher throughput** with larger batches

### **Model Quality**
- **Good for code generation** tasks
- **Faster inference** than 13B
- **Lower resource requirements** for deployment

## 🆚 Comparison: 7B vs 13B

| **Aspect** | **CodeLlama-7B** | **CodeLlama-13B** |
|------------|------------------|-------------------|
| **GPUs Required** | 2 x 16GB+ | 4 x 24GB+ |
| **Training Speed** | ⚡ Faster | 🐌 Slower |
| **Memory Usage** | 💚 Lower | 🔴 Higher |
| **Model Quality** | ✅ Good | ⭐ Better |
| **Inference Speed** | ⚡ Fast | 🐌 Slower |
| **Deployment** | 💚 Easier | 🔴 Harder |

## 🎯 Expected Output

```
✅ GPU access confirmed
📊 Detected 2 GPU(s)
🎯 Simple NeMo 24.07 Fine-tuning Pipeline for CodeLlama-7B
============================================================
📥 Downloading codellama-7b...
✅ Downloaded to ./codellama-7b-hf
🔄 Converting ./codellama-7b-hf to ./codellama-7b.nemo...
✅ Converted to ./codellama-7b.nemo
📝 Preparing data from example_training_data.yaml...
✅ Created X training and Y validation examples
🚀 Starting fine-tuning for 50 steps...
✅ Fine-tuning completed successfully!
🎉 Fine-tuning pipeline completed!
📁 Trained model: /results/checkpoints/megatron_gpt_peft_lora_tuning.nemo
```

## 🐛 Troubleshooting

### Memory Issues (Less Common with 7B)
```bash
# If still getting OOM, reduce batch size further:
# Edit simple_nemo_finetune_7b.py:
# model.global_batch_size=8   # instead of 16
# model.micro_batch_size=1    # instead of 2
```

### GPU Requirements
```bash
# Minimum requirements for CodeLlama-7B:
# - 2 GPUs with 16GB VRAM each
# - Can potentially run on 1 GPU with 24GB+ VRAM
```

## 🎯 When to Use CodeLlama-7B vs 13B

### **Use CodeLlama-7B when:**
- ✅ Limited GPU resources (2 GPUs available)
- ✅ Need faster training/inference
- ✅ Prototyping and experimentation
- ✅ Good enough quality for your use case

### **Use CodeLlama-13B when:**
- ⭐ Maximum code generation quality needed
- ⭐ Have sufficient GPU resources (4+ GPUs)
- ⭐ Production deployment with quality priority
- ⭐ Complex code generation tasks

## 📚 Based on Official Documentation

- [NeMo 24.07 Getting Started](https://docs.nvidia.com/nemo-framework/user-guide/24.07/getting-started.html)
- [CodeLlama-7B Model Card](https://huggingface.co/meta-llama/CodeLlama-7b-hf)
- [NeMo GitHub Repository](https://github.com/NVIDIA/NeMo)
