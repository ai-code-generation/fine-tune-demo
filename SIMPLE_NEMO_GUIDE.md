# 🎯 Simple NeMo 24.07 Fine-tuning Guide for CodeLlama-13B

This guide follows the **official NeMo documentation** and provides the simplest possible workflow for fine-tuning CodeLlama-13B.

## 📋 Prerequisites

- Docker with GPU support
- HuggingFace token (for CodeLlama access)
- At least 4 GPUs with 24GB+ VRAM each

## 🚀 Quick Start (5 Simple Steps)

### Step 1: Get HuggingFace Token
```bash
# Get token from: https://huggingface.co/settings/tokens
# Request access to: https://huggingface.co/meta-llama/CodeLlama-13b-hf
export HF_TOKEN="your_token_here"
```

### Step 2: Start NeMo Container
```bash
cd /static-data/team_08/simple-nemo/fine-tune-demo
./run_nemo_container.sh
```

### Step 3: Run Fine-tuning (inside container)
```bash
# Set your HF token inside container
export HF_TOKEN="your_token_here"

# Run the simple fine-tuning script
python simple_nemo_finetune.py \
  --data example_training_data.yaml \
  --max-steps 50 \
  --hf-token $HF_TOKEN
```

### Step 4: Check Results (inside container)
```bash
# List results
ls -la /results/

# Check training logs
ls -la /results/checkpoints/

# View final model
ls -la /results/checkpoints/megatron_gpt_peft_lora_tuning.nemo
```

### Step 5: Test the Model (inside container)
```bash
# Run inference test
python /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_generate.py \
  model.restore_from_path=/results/checkpoints/megatron_gpt_peft_lora_tuning.nemo \
  trainer.devices=4 \
  model.tensor_model_parallel_size=2 \
  model.pipeline_model_parallel_size=1 \
  model.data.test_ds.file_names="[/workspace/validation.jsonl]" \
  model.data.test_ds.names="['codellama_test']" \
  model.data.test_ds.global_batch_size=4 \
  model.data.test_ds.micro_batch_size=1 \
  model.data.test_ds.tokens_to_generate=50 \
  inference.greedy=True \
  model.data.test_ds.output_file_path_prefix=/results/codellama_results \
  model.data.test_ds.write_predictions_to_file=True
```

## 📁 What the Script Does

1. **Downloads CodeLlama-13B** from HuggingFace
2. **Converts to .nemo format** using official NeMo converter
3. **Prepares training data** from YAML to JSONL format
4. **Runs LoRA fine-tuning** using official NeMo script
5. **Saves results** to `/results` directory

## 🔧 Configuration

The script uses these **official NeMo settings**:
- **Model**: CodeLlama-13B with LoRA fine-tuning
- **Precision**: bf16 (recommended for A100)
- **Parallelism**: TP=2, PP=1 (for 4 GPUs)
- **Batch Size**: Global=8, Micro=1
- **Learning Rate**: 1e-5
- **Sequence Length**: 2048 tokens

## 📊 Expected Output

```
🎯 Simple NeMo 24.07 Fine-tuning Pipeline
==================================================
📥 Downloading codellama-13b...
✅ Downloaded to ./codellama-13b-hf
🔄 Converting ./codellama-13b-hf to ./codellama-13b.nemo...
✅ Converted to ./codellama-13b.nemo
📝 Preparing data from example_training_data.yaml...
✅ Created 45 training and 5 validation examples
🚀 Starting fine-tuning for 50 steps...
✅ Fine-tuning completed successfully!
🎉 Fine-tuning pipeline completed!
📁 Trained model: /results/checkpoints/megatron_gpt_peft_lora_tuning.nemo
📁 Results directory: /results
```

## 🐛 Troubleshooting

### Container Issues
```bash
# If container fails to start
docker system prune -f
nvidia-docker run --rm nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

### Memory Issues
```bash
# Reduce batch size in script if OOM
# Edit simple_nemo_finetune.py:
# model.global_batch_size=4  # instead of 8
# model.micro_batch_size=1   # keep as 1
```

### Token Issues
```bash
# Test HF token
python -c "from huggingface_hub import whoami; print(whoami())"
```

## 📚 Based on Official Documentation

- [NeMo 24.07 Getting Started](https://docs.nvidia.com/nemo-framework/user-guide/24.07/getting-started.html)
- [NeMo Llama2 SFT Playbook](https://docs.nvidia.com/nemo-framework/user-guide/24.07/playbooks/llama2sft.html)
- [NeMo GitHub Repository](https://github.com/NVIDIA/NeMo)

## 🎯 Key Differences from Complex Pipeline

- ✅ **Follows official NeMo workflow** exactly
- ✅ **Uses official NeMo scripts** directly
- ✅ **No custom config generation** - uses NeMo defaults
- ✅ **Proper .nemo conversion** - mandatory step
- ✅ **Simple command-line interface** - no complex abstractions
- ✅ **Clear error messages** - based on official examples
