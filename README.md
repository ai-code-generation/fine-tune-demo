# NeMo 24.07 Fine-tuning Pipeline

**Refactored project optimized for NeMo 24.07 with focus on CodeLlama-13B and Llama3 models (8B-70B)**

## 🚀 **Key Features**

- ✅ **NeMo 24.07 Optimized**: Latest NeMo framework with performance improvements
- ✅ **CodeLlama & Llama3 Support**: Specialized for CodeLlama-13B and Llama3-8B/70B models
- ✅ **Docker Integration**: Seamless container deployment with proper volume mounting
- ✅ **YAML Training Data**: Support for multi-document YAML conversation format
- ✅ **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning
- ✅ **Hardware Optimization**: Configurations tuned for different GPU setups
- ✅ **Automated Pipeline**: End-to-end training with minimal manual intervention

## 📋 **Requirements**

### **Hardware**
- **CodeLlama-13B**: 4+ GPUs, 24GB+ VRAM per GPU
- **Llama3-8B**: 4+ GPUs, 16GB+ VRAM per GPU  
- **Llama3-70B**: 16+ GPUs, 80GB+ VRAM per GPU

### **Software**
- Docker with NVIDIA GPU support
- Host path: `/static-data/team_08/simple-nemo/fine-tune-demo`
- HuggingFace account with access to gated models

## 🏗️ **Project Structure**

```
fine-tune-demo/
├── finetune_pipeline.py           # Main NeMo 24.07 pipeline
├── configs/
│   ├── model_configs_nemo24.py    # Model configurations
│   ├── codellama_13b_nemo24_config.yaml
│   ├── llama3_8b_nemo24_config.yaml
│   └── llama3_70b_nemo24_config.yaml
├── docker-compose.yml             # Updated for NeMo 24.07
├── docker_start.sh                # Docker startup script
├── quick_start.sh                 # Interactive setup
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

### **Option 2: Direct Docker Run**

```bash
# Use the provided script
./docker_start.sh

# Or manual docker run
docker run -it --gpus all --shm-size=16g \
  -v /static-data/team_08/simple-nemo/fine-tune-demo:/workspace \
  -e HF_TOKEN=your_token \
  nvcr.io/nvidia/nemo:24.07 /bin/bash
```

## ⚙️ **Configuration**

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

## 🎉 **Getting Started**

1. **Set up your environment**: Ensure Docker and NVIDIA support
2. **Get HuggingFace access**: Request access to gated models
3. **Prepare your data**: Format in YAML conversation structure
4. **Run the pipeline**: Use `quick_start_nemo24.sh` for guided setup
5. **Monitor training**: Check logs in `./outputs/logs/`
6. **Evaluate results**: Test your fine-tuned model

## 📚 **Additional Resources**

- [NeMo 24.07 Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/)
- [HuggingFace Model Hub](https://huggingface.co/models)
- [Docker GPU Support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

---

**Ready to start fine-tuning with NeMo 24.07!** 🚀
