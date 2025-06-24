# 🎯 Simple NeMo Fine-tuning for CodeLlama

A streamlined, production-ready pipeline for fine-tuning CodeLlama models using NVIDIA NeMo 24.07 framework.

## 🚀 **Two Optimized Options**

### **🔥 CodeLlama-7B (Resource Efficient)**
- **Requirements**: 2 GPUs with 16GB+ VRAM each
- **Use case**: Limited resources, experimentation, faster training
- **Quality**: Good for most code generation tasks
- **Script**: `simple_nemo_finetune_7b.py`

### **⭐ CodeLlama-13B (Higher Quality)**
- **Requirements**: 4 GPUs with 24GB+ VRAM each
- **Use case**: Production deployment, maximum quality
- **Quality**: Best for complex code generation tasks
- **Script**: `simple_nemo_finetune.py`

## 📋 **Prerequisites**

- Docker with GPU support
- NVIDIA drivers and nvidia-docker2
- HuggingFace account with access to CodeLlama models
- GPU requirements (see options above)

## 📁 **Project Structure**

```
simple-nemo-finetune/
├── 📄 README.md                           # This comprehensive guide
├── 📄 simple_nemo_finetune_7b.py         # CodeLlama-7B script (2 GPUs)
├── 📄 simple_nemo_finetune.py            # CodeLlama-13B script (4 GPUs)
├── 📄 debug_jsonl_files.py               # JSONL debugging tool
├── 📄 simple_docker_run.sh               # Container runner
├── 📄 requirements.txt                   # Python dependencies
├── 📄 example_training_data.yaml         # Sample training data
└── 📄 LICENSE                           # License file
```

## 🎯 **Quick Start Guide**

### **Step 1: Get HuggingFace Token**
```bash
# Get token from: https://huggingface.co/settings/tokens
# Request access to: https://huggingface.co/meta-llama/CodeLlama-7b-hf
# Request access to: https://huggingface.co/meta-llama/CodeLlama-13b-hf
export HF_TOKEN="your_token_here"
```

### **Step 2: Fix File Permissions (if needed)**
```bash
# If you get permission denied errors after cloning:
./fix_permissions.sh

# Or manually:
chmod 755 *.py *.sh
```

### **Step 3: Start NeMo Container**
```bash
# Simple method using provided script
./simple_docker_run.sh

# Manual method (if needed)
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

### **Step 4: Choose Your Model (inside container)**

#### **For CodeLlama-7B (Resource Efficient)**
```bash
export HF_TOKEN="your_token_here"
python simple_nemo_finetune_7b.py \
  --data example_training_data.yaml \
  --max-steps 50 \
  --hf-token $HF_TOKEN
```

#### **For CodeLlama-13B (Higher Quality)**
```bash
export HF_TOKEN="your_token_here"
python simple_nemo_finetune.py \
  --data example_training_data.yaml \
  --max-steps 50 \
  --hf-token $HF_TOKEN
```

### **Step 5: Check Results (inside container)**
```bash
# List results
ls -la /results/

# Check training logs
ls -la /results/checkpoints/

# View final model
ls -la /results/checkpoints/megatron_gpt_peft_lora_tuning.nemo
```

## 📊 **Model Comparison**

| **Aspect** | **CodeLlama-7B** | **CodeLlama-13B** |
|------------|------------------|-------------------|
| **GPUs Required** | 2 x 16GB+ | 4 x 24GB+ |
| **Training Speed** | ⚡ Faster | 🐌 Slower |
| **Memory Usage** | 💚 Lower | 🔴 Higher |
| **Model Quality** | ✅ Good | ⭐ Better |
| **Inference Speed** | ⚡ Fast | 🐌 Slower |
| **Deployment** | 💚 Easier | 🔴 Harder |

## 🎯 **When to Use Each Model**

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

## 🔧 **Technical Configuration**

### **CodeLlama-7B Settings**
```bash
# Optimized for smaller model, less resources
trainer.devices=2                    # 2 GPUs instead of 4
model.micro_batch_size=2             # Larger micro batch
model.global_batch_size=16           # Larger global batch
model.tensor_model_parallel_size=1   # No TP needed
model.optim.lr=2e-5                  # Slightly higher LR
torchrun --nproc_per_node=2          # 2 processes
```

### **CodeLlama-13B Settings**
```bash
# Optimized for higher quality, more resources
trainer.devices=4                    # 4 GPUs for better performance
model.micro_batch_size=1             # Smaller micro batch
model.global_batch_size=8            # Smaller global batch
model.tensor_model_parallel_size=2   # 2-way tensor parallelism
model.optim.lr=1e-5                  # Lower LR for stability
torchrun --nproc_per_node=4          # 4 processes
```

## 📝 **Training Data Format**

Your training data should be in YAML format with multiple documents:

```yaml
---
messages:
  - role: system
    content: "You are an expert assistant."
  - role: user
    content: "Create a function to calculate factorial."
  - role: assistant
    content: |
      ```python
      def factorial(n):
          if n <= 1:
              return 1
          return n * factorial(n - 1)
      ```
---
messages:
  - role: user
    content: "How do I handle exceptions in Python?"
  - role: assistant
    content: |
      ```python
      try:
          result = risky_operation()
      except ValueError as e:
          print(f"Error: {e}")
      except Exception as e:
          print(f"Unexpected error: {e}")
      finally:
          cleanup()
      ```
```

## 🐛 **Troubleshooting**

### **Common Issues:**

#### **1. CUDA/GPU Issues**
```bash
# Check GPU access
nvidia-smi

# If "CUDA error: no CUDA-capable device is detected":
# - Make sure you're using --gpus all in docker run
# - Check host GPU access: nvidia-smi (outside container)
# - Restart Docker daemon: sudo systemctl restart docker
```

#### **2. JSON Parsing Errors**
```bash
# Debug JSONL files
python debug_jsonl_files.py

# This will analyze and fix:
# - Line ending issues (CRLF vs LF)
# - Invalid JSON formatting
# - Character encoding problems
```

#### **3. Memory Issues**
```bash
# For CodeLlama-7B, reduce batch size:
# Edit simple_nemo_finetune_7b.py:
# model.global_batch_size=8   # instead of 16
# model.micro_batch_size=1    # instead of 2

# For CodeLlama-13B, reduce batch size:
# Edit simple_nemo_finetune.py:
# model.global_batch_size=4   # instead of 8
# model.micro_batch_size=1    # keep as 1
```

#### **4. Permission Issues**
```bash
# If you get "Permission denied" when running scripts:
./fix_permissions.sh

# Or manually fix permissions:
chmod 755 *.py *.sh
chmod 644 *.md *.yaml *.txt LICENSE

# Check current permissions:
ls -la *.py *.sh
```

#### **5. Container Issues**
```bash
# Clean Docker system
docker system prune -f

# Test GPU access in container
nvidia-docker run --rm nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

#### **6. Token Issues**
```bash
# Test HF token
python -c "from huggingface_hub import whoami; print(whoami())"
```

### **Quick Fixes:**
```bash
# Clean start (remove cache/index files)
rm -rf cache/ *.jsonl.idx.*

# Reset environment
unset HF_TOKEN
export HF_TOKEN="your_new_token"

# Check container access
docker run --gpus all --rm nvcr.io/nvidia/nemo:24.07 nvidia-smi
```

## 📊 **Expected Output**

### **Successful Training Output:**
```
🎯 Simple NeMo 24.07 Fine-tuning Pipeline for CodeLlama-7B
============================================================
✅ GPU access confirmed
📊 Detected 2 GPU(s)
📥 Downloading codellama-7b...
✅ Downloaded to ./codellama-7b-hf
🔄 Converting ./codellama-7b-hf to ./codellama-7b.nemo...
✅ Converted to ./codellama-7b.nemo
🧹 Cleaning up existing NeMo index files...
✅ Cleaned up 0 index files
📝 Preparing data from example_training_data.yaml...
🔍 Validating generated JSONL files...
✅ Created 45 training and 5 validation examples
✅ JSONL files validated successfully
🚀 Starting fine-tuning for 50 steps...
[NeMo I] Building index files...
[NeMo I] Loading /workspace/train.jsonl
[NeMo I] Loading /workspace/validation.jsonl
✅ Fine-tuning completed successfully!
🎉 Fine-tuning pipeline completed!
📁 Trained model: /results/checkpoints/megatron_gpt_peft_lora_tuning.nemo
📁 Results directory: /results
```

## 🧪 **Testing Your Model**

### **Run Inference Test (inside container)**
```bash
# For CodeLlama-7B
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

# For CodeLlama-13B
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

## 🔧 **Advanced Features**

### **Debug JSONL Files**
```bash
# Analyze and fix JSONL file issues
python debug_jsonl_files.py

# This tool will:
# ✅ Check line endings (CRLF vs LF)
# ✅ Validate JSON formatting
# ✅ Detect character encoding issues
# ✅ Fix common problems automatically
# ✅ Create backups before fixing
```

### **Custom Training Parameters**
```bash
# Adjust training steps
python simple_nemo_finetune_7b.py --max-steps 100

# Use different data file
python simple_nemo_finetune_7b.py --data my_custom_data.yaml

# Specify HF token directly
python simple_nemo_finetune_7b.py --hf-token hf_your_token_here
```

## 🎉 **Success Stories**

This pipeline has been successfully tested with:
- ✅ **S32K14 automotive microcontroller** code generation
- ✅ **Java test automation frameworks** using SWTBot
- ✅ **Embedded systems programming** for automotive applications
- ✅ **Multi-language code generation** tasks
- ✅ **Eclipse IDE automation** scripts

## 📚 **Based on Official Documentation**

- [NeMo 24.07 Getting Started](https://docs.nvidia.com/nemo-framework/user-guide/24.07/getting-started.html)
- [NeMo Llama2 SFT Playbook](https://docs.nvidia.com/nemo-framework/user-guide/24.07/playbooks/llama2sft.html)
- [NeMo GitHub Repository](https://github.com/NVIDIA/NeMo)
- [CodeLlama Model Cards](https://huggingface.co/meta-llama)

## 🔑 **Key Features**

- ✅ **Simplified Scripts**: Two optimized scripts for different resource levels
- ✅ **Robust Error Handling**: Comprehensive CUDA, JSON, and memmap error fixes
- ✅ **Official NeMo Workflow**: Follows NeMo 24.07 documentation exactly
- ✅ **Resource Optimization**: Configurations tuned for 7B vs 13B models
- ✅ **Debugging Tools**: JSONL validation and fixing utilities
- ✅ **Production Ready**: Tested, working implementations
- ✅ **Docker Integration**: Simple container setup with proper GPU access
- ✅ **Automatic Cleanup**: Handles corrupted index files and cache issues

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Test your changes with both 7B and 13B models
4. Submit a pull request

## 🙏 **Acknowledgments**

- NVIDIA NeMo team for the excellent framework
- Meta for the CodeLlama models
- Community contributors and testers

---

**Ready to start fine-tuning? Choose your model size and run the appropriate script!** 🚀
│   ├── model_configs.py           # Model configurations
│   ├── codellama_13b_config.yaml  # CodeLlama-13B config
│   ├── llama3_8b_config.yaml      # Llama3-8B config
│   └── llama3_70b_config.yaml     # Llama3-70B config
├── docker-compose.yml             # Docker Compose with --user support
├── docker_start.sh                # Docker startup script
├── simple_docker_run.sh           # Simple Docker run with --user
├── quick_start.sh                 # Interactive setup
├── DOCKER_GUIDE.md               # Step-by-step Docker guide
├── data_preprocessing.py           # YAML data processing
├── setup_hf_auth.py               # HuggingFace authentication
├── example_training_data.yaml     # Example training data
└── outputs/                       # Training outputs
    ├── models/                    # Downloaded and trained models
    ├── data/                      # Processed training data
    ├── experiments/               # Training configurations
    └── logs/                      # Training logs
```

## 🚀 **Quick Start**

### **1. Set Up Environment**

```bash
# Clone and navigate to project
cd /static-data/team_08/simple-nemo/fine-tune-demo

# Set up HuggingFace token
echo "HF_TOKEN=your_token_here" > .env

# Start Docker container
./docker_start.sh
```

### **2. Run Interactive Setup**

```bash
# Inside the container
./quick_start.sh
```

### **3. Direct Pipeline Usage**

```bash
# Check hardware requirements
python finetune_pipeline.py --model codellama-13b --check-hardware

# Run fine-tuning
python finetune_pipeline.py \
    --model codellama-13b \
    --data your_training_data.yaml \
    --output-dir ./outputs \
    --max-steps 100 \
    --hf-token your_token_here
```

## 📊 **Supported Models**

### **CodeLlama Models**
- **`codellama-13b`**: CodeLlama 13B - Specialized for code generation

### **Llama3 Models**  
- **`llama3-8b`**: Llama3 8B - Balanced performance and efficiency
- **`llama3-70b`**: Llama3 70B - Maximum performance (high GPU requirements)

> **Note**: All models require HuggingFace access approval. Visit the model pages to request access.

## 📝 **Training Data Format**

Your YAML training data should follow this format:

```yaml
---
messages:
  - role: system
    content: |
      You are an expert test automation assistant using SWTBot.
  - role: user
    content: |
      Task: Create and build an S32K144 project...
  - role: assistant
    content: |
      ```java
      bot.menu("File")...
      ```
---
messages:
  - role: system
    content: |
      You are an expert test automation assistant using SWTBot.
  - role: user
    content: |
      Another task description...
  - role: assistant
    content: |
      ```java
      // Another code example
      ```
```

## 🐳 **Docker Usage**

### **Option 1: Docker Compose (Recommended)**

```bash
# Set up environment
echo "HF_TOKEN=your_token" > .env

# Start container
docker-compose up nemo-finetuning

# In another terminal, connect to container
docker exec -it nemo-finetuning-24.07 /bin/bash
```

### **Option 2: Simple Docker Run (with --user flag)**

```bash
# Use the simple script (includes --user flag automatically)
./simple_docker_run.sh

# Or use the advanced script
./docker_start.sh

# Or manual docker run with --user flag
docker run -it --rm \
  --user $(id -u):$(id -g) \
  --gpus all --shm-size=16g \
  -v /static-data/team_08/simple-nemo/fine-tune-demo:/workspace \
  -e HF_TOKEN=your_token \
  nvcr.io/nvidia/nemo:24.07 /bin/bash
```

## ⚙️ **Configuration**

### **Configuration Files**

The pipeline uses these optimized configuration files:

- **`configs/codellama_13b_config.yaml`** - CodeLlama-13B optimized settings
- **`configs/llama3_8b_config.yaml`** - Llama3-8B optimized settings
- **`configs/llama3_70b_config.yaml`** - Llama3-70B optimized settings (FP8, sequence parallelism)
- **`configs/model_configs.py`** - Model definitions and hardware requirements

### **Model-Specific Optimizations**

Each model has optimized configurations:

- **CodeLlama-13B**: Selective activation checkpointing, Flash Attention
- **Llama3-8B**: Standard optimizations for balanced performance
- **Llama3-70B**: FP8 training, sequence parallelism, aggressive optimizations

### **Hardware Scaling**

The pipeline automatically adjusts:
- Tensor parallelism based on model size
- Batch sizes for available GPU memory
- Gradient accumulation for effective batch sizes

## 🔧 **Advanced Usage**

### **Custom Training Parameters**

```bash
python finetune_pipeline.py \
    --model llama3-8b \
    --data training_data.yaml \
    --output-dir ./outputs \
    --max-steps 500 \
    --validation-split 0.15 \
    --hf-token your_token
```

### **Hardware Requirements Check**

```bash
# Check requirements for specific model
python finetune_pipeline.py --model llama3-70b --check-hardware
```

### **Multi-Node Training**

For Llama3-70B, configure multi-node setup:

```yaml
# In config file
trainer:
  devices: 16
  num_nodes: 2
  
model:
  tensor_model_parallel_size: 8
  pipeline_model_parallel_size: 2
```

## 🎯 **NeMo 24.07 Optimizations**

### **Performance Features**
- ✅ **Transformer Engine**: Hardware-accelerated attention
- ✅ **Flash Attention**: Memory-efficient attention computation
- ✅ **FP8 Training**: Reduced precision for large models
- ✅ **Sequence Parallelism**: Efficient long sequence handling
- ✅ **Gradient Fusion**: Optimized gradient operations

### **Memory Optimizations**
- ✅ **Selective Activation Checkpointing**: Balanced memory/compute trade-off
- ✅ **Gradient Accumulation Fusion**: Reduced memory overhead
- ✅ **Mixed Precision**: BF16 training with stability

## 🔍 **Troubleshooting**

### **Common Issues**

1. **HuggingFace Access Denied**:
   ```bash
   # Request access to gated models
   # Visit: https://huggingface.co/meta-llama/CodeLlama-13b-hf
   # Click "Request Access" and wait for approval
   ```

2. **GPU Memory Issues**:
   ```bash
   # Reduce batch size in config
   micro_batch_size: 1
   global_batch_size: 4
   ```

3. **Docker Mount Issues**:
   ```bash
   # Ensure host path exists and has proper permissions
   sudo chown -R $USER:$USER /static-data/team_08/simple-nemo/fine-tune-demo
   ```

### **Performance Tuning**

- **For faster training**: Increase `micro_batch_size` if GPU memory allows
- **For memory constraints**: Enable `activations_checkpoint_granularity: selective`
- **For large models**: Use `sequence_parallel: true` and `fp8_training: true`

## 📈 **Expected Performance**

### **Training Speed (approximate)**
- **CodeLlama-13B**: ~2-3 minutes per 100 steps (4x A100)
- **Llama3-8B**: ~1-2 minutes per 100 steps (4x A100)  
- **Llama3-70B**: ~5-8 minutes per 100 steps (16x A100)

### **Memory Usage**
- **CodeLlama-13B**: ~20GB per GPU (with optimizations)
- **Llama3-8B**: ~15GB per GPU (with optimizations)
- **Llama3-70B**: ~75GB per GPU (with optimizations)

## 🚀 **Quick Reference**

### **Simple 3-Step Setup**

```bash
# 1. Navigate and setup
cd /static-data/team_08/simple-nemo/fine-tune-demo
echo "HF_TOKEN=your_token_here" > .env

# 2. Start container (with --user flag)
./simple_docker_run.sh

# 3. Inside container - run training
./quick_start.sh
```

### **Direct Training Commands**

```bash
# Check hardware requirements
python finetune_pipeline.py --model codellama-13b --check-hardware

# Train CodeLlama-13B
python finetune_pipeline.py --model codellama-13b --data example_training_data.yaml --max-steps 100

# Train Llama3-8B
python finetune_pipeline.py --model llama3-8b --data example_training_data.yaml --max-steps 100
```

## 🎉 **Getting Started**

1. **Set up your environment**: Ensure Docker and NVIDIA support
2. **Get HuggingFace access**: Request access to gated models
3. **Prepare your data**: Format in YAML conversation structure
4. **Run the pipeline**: Use `simple_docker_run.sh` or `quick_start.sh` for guided setup
5. **Monitor training**: Check logs in `./outputs/logs/`
6. **Evaluate results**: Test your fine-tuned model

## 📚 **Additional Resources**

- [NeMo 24.07 Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/)
- [HuggingFace Model Hub](https://huggingface.co/models)
- [Docker GPU Support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

---

**Ready to start fine-tuning with NeMo 24.07!** 🚀
