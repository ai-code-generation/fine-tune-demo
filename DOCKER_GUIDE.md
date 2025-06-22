# Simple Docker Build, Run & Training Guide

**Step-by-step instructions for NeMo 24.07 fine-tuning with Docker**

## 📋 **Prerequisites**

- Docker with NVIDIA GPU support installed
- Host directory: `/static-data/team_08/simple-nemo/fine-tune-demo`
- HuggingFace token for gated models

## 🚀 **Step 1: Build (Optional - Using Pre-built Image)**

We use the official NeMo 24.07 image, so no building required:

```bash
# Pull the latest NeMo 24.07 image
docker pull nvcr.io/nvidia/nemo:24.07
```

## 🏃 **Step 2: Run Docker Container**

### **Option A: Simple Run Command (with --user flag)**

```bash
# Navigate to your project directory
cd /static-data/team_08/simple-nemo/fine-tune-demo

# Set up HuggingFace token
echo "HF_TOKEN=your_huggingface_token_here" > .env

# Run container with --user flag
docker run -it --rm \
  --user $(id -u):$(id -g) \
  --gpus all \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /static-data/team_08/simple-nemo/fine-tune-demo:/workspace \
  -v /static-data/team_08/simple-nemo/fine-tune-demo/outputs:/workspace/outputs \
  -v /static-data/team_08/simple-nemo/fine-tune-demo/.cache:/workspace/.cache \
  -e HF_TOKEN=$(cat .env | grep HF_TOKEN | cut -d'=' -f2) \
  -e CUDA_VISIBLE_DEVICES=all \
  -e PYTHONPATH=/workspace \
  -w /workspace \
  nvcr.io/nvidia/nemo:24.07 \
  bash
```

### **Option B: Using Docker Compose (Recommended)**

```bash
# Navigate to project directory
cd /static-data/team_08/simple-nemo/fine-tune-demo

# Set up environment
echo "HF_TOKEN=your_huggingface_token_here" > .env
echo "USER_ID=$(id -u)" >> .env
echo "GROUP_ID=$(id -g)" >> .env

# Start with docker-compose
docker-compose up -d nemo-finetuning

# Connect to running container
docker exec -it --user $(id -u):$(id -g) nemo-finetuning bash
```

### **Option C: Using Provided Script**

```bash
# Use the provided script (automatically adds --user)
./docker_start.sh
```

## 🎯 **Step 3: Training**

Once inside the container, you have several options:

### **Option A: Interactive Quick Start**

```bash
# Run interactive setup
./quick_start.sh
```

### **Option B: Direct Training Command**

```bash
# Check available models
python finetune_pipeline.py --help

# Check hardware requirements
python finetune_pipeline.py --model codellama-13b --check-hardware

# Run training with your data
python finetune_pipeline.py \
  --model codellama-13b \
  --data example_training_data.yaml \
  --output-dir ./outputs \
  --max-steps 100 \
  --validation-split 0.1
```

### **Option C: Step-by-Step Manual Training**

```bash
# 1. Check your training data format
head -20 example_training_data.yaml

# 2. Preprocess data (if needed)
python data_preprocessing.py \
  --input example_training_data.yaml \
  --output processed_data \
  --validation-split 0.1

# 3. Run training
python finetune_pipeline.py \
  --model llama3-8b \
  --data example_training_data.yaml \
  --max-steps 100
```

## 📊 **Available Models**

| Model | GPUs Required | VRAM per GPU | Command |
|-------|---------------|--------------|---------|
| **CodeLlama-13B** | 4+ | 24GB+ | `--model codellama-13b` |
| **Llama3-8B** | 4+ | 16GB+ | `--model llama3-8b` |
| **Llama3-70B** | 16+ | 80GB+ | `--model llama3-70b` |

## 🔧 **Training Data Format**

Your YAML file should follow this structure:

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
      bot.menu("File").click();
      bot.menu("New").click();
      ```
---
messages:
  - role: system
    content: |
      You are an expert test automation assistant using SWTBot.
  - role: user
    content: |
      Another task...
  - role: assistant
    content: |
      ```java
      // Another code example
      ```
```

## 🎛️ **Configuration Files**

The pipeline uses these optimized configurations:

- **`configs/codellama_13b_config.yaml`** - CodeLlama-13B settings
- **`configs/llama3_8b_config.yaml`** - Llama3-8B settings  
- **`configs/llama3_70b_config.yaml`** - Llama3-70B settings

## 📁 **Output Structure**

After training, your outputs will be in:

```
outputs/
├── models/           # Downloaded and trained models
│   ├── codellama-13b/
│   └── codellama-13b_merged.nemo
├── data/            # Processed training data
├── experiments/     # Training configurations
└── logs/           # Training logs
```

## 🔍 **Monitoring Training**

```bash
# Check training logs
tail -f outputs/logs/training.log

# Monitor GPU usage
nvidia-smi

# Check training progress
ls -la outputs/experiments/
```

## 🚨 **Troubleshooting**

### **Permission Issues**
```bash
# Fix permissions if needed
sudo chown -R $(id -u):$(id -g) /static-data/team_08/simple-nemo/fine-tune-demo
```

### **GPU Memory Issues**
```bash
# Check available GPU memory
nvidia-smi

# Use smaller batch size in config files
# Edit: micro_batch_size: 1, global_batch_size: 4
```

### **HuggingFace Access Issues**
```bash
# Set up HF authentication
python setup_hf_auth.py

# Or manually set token
export HF_TOKEN=your_token_here
```

## ✅ **Complete Example Workflow**

```bash
# 1. Navigate to project
cd /static-data/team_08/simple-nemo/fine-tune-demo

# 2. Set up environment
echo "HF_TOKEN=hf_your_token_here" > .env

# 3. Start container with --user flag
docker run -it --rm \
  --user $(id -u):$(id -g) \
  --gpus all \
  --shm-size=16g \
  -v /static-data/team_08/simple-nemo/fine-tune-demo:/workspace \
  -e HF_TOKEN=hf_your_token_here \
  -w /workspace \
  nvcr.io/nvidia/nemo:24.07 \
  bash

# 4. Inside container - run training
./quick_start.sh

# Or direct command:
python finetune_pipeline.py \
  --model codellama-13b \
  --data example_training_data.yaml \
  --max-steps 100
```

## 🎉 **Success!**

Your fine-tuned model will be saved in `outputs/models/` and ready for use!

---

**Note**: The `--user $(id -u):$(id -g)` flag ensures that files created inside the container have the correct ownership on the host system.
