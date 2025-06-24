# 🔄 NeMo to HuggingFace Model Conversion Guide

This guide helps you convert your trained NeMo LoRA model to HuggingFace format for easy deployment and inference.

## 📋 Your Training Results

Based on your output, you have successfully trained a CodeLlama model with these files:
```
./results/checkpoints/
├── megatron_gpt_peft_lora_tuning--validation_loss=0.690-step=50-consumed_samples=800.0-last.ckpt
├── megatron_gpt_peft_lora_tuning--validation_loss=0.690-step=50-consumed_samples=800.0.ckpt
└── megatron_gpt_peft_lora_tuning.nemo  # ← This is what we need to convert
```

## 🎯 Conversion Options

### **Option 1: Full LoRA Conversion (Recommended)**
Converts and merges LoRA weights into the base model:

```bash
# Inside NeMo container
python convert_lora_to_hf.py \
  --nemo-model /results/checkpoints/megatron_gpt_peft_lora_tuning.nemo \
  --base-model meta-llama/CodeLlama-7b-hf \
  --output-dir ./converted_codellama_7b \
  --merge-lora
```

### **Option 2: Official NeMo Converter**
Uses NeMo's built-in conversion tools:

```bash
# Inside NeMo container
python convert_nemo_to_hf.py \
  --nemo-model /results/checkpoints/megatron_gpt_peft_lora_tuning.nemo \
  --output-dir ./converted_codellama_7b \
  --base-model meta-llama/CodeLlama-7b-hf
```

### **Option 3: Simple Base Model Export**
Exports base model as starting point:

```bash
# Inside NeMo container
python convert_lora_to_hf.py \
  --nemo-model /results/checkpoints/megatron_gpt_peft_lora_tuning.nemo \
  --base-model meta-llama/CodeLlama-7b-hf \
  --output-dir ./converted_codellama_7b \
  --simple
```

## 🚀 Step-by-Step Conversion

### **Step 1: Ensure You're in NeMo Container**
```bash
# If not already in container, start it:
./simple_docker_run.sh

# Inside container, verify NeMo is available:
python -c "import nemo; print('NeMo version:', nemo.__version__)"
```

### **Step 2: Run Conversion**
```bash
# For CodeLlama-7B (most common)
python convert_lora_to_hf.py \
  --nemo-model /results/checkpoints/megatron_gpt_peft_lora_tuning.nemo \
  --base-model meta-llama/CodeLlama-7b-hf \
  --output-dir ./converted_model \
  --merge-lora

# For CodeLlama-13B (if you trained 13B model)
python convert_lora_to_hf.py \
  --nemo-model /results/checkpoints/megatron_gpt_peft_lora_tuning.nemo \
  --base-model meta-llama/CodeLlama-13b-hf \
  --output-dir ./converted_model \
  --merge-lora
```

### **Step 3: Test Converted Model**
```bash
# Basic test with predefined prompts
python test_converted_model.py --model-path ./converted_model

# Interactive testing
python test_converted_model.py --model-path ./converted_model --interactive

# Benchmark performance
python test_converted_model.py --model-path ./converted_model --benchmark
```

### **Step 4: Verify Conversion**
```bash
# Check converted files
ls -la ./converted_model/

# Expected files:
# ├── config.json
# ├── pytorch_model.bin (or model.safetensors)
# ├── tokenizer.json
# ├── tokenizer_config.json
# ├── special_tokens_map.json
# └── README.md
```

## 🧪 Testing Your Converted Model

### **Quick Test**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load your converted model
tokenizer = AutoTokenizer.from_pretrained("./converted_model")
model = AutoModelForCausalLM.from_pretrained("./converted_model", torch_dtype=torch.bfloat16)

# Test code generation
prompt = "def factorial(n):"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### **Advanced Testing**
```bash
# Test with custom prompts
python test_converted_model.py \
  --model-path ./converted_model \
  --max-length 300 \
  --temperature 0.8 \
  --interactive
```

## 📊 Expected Results

### **Successful Conversion Output:**
```
🎯 NeMo LoRA to HuggingFace Converter
========================================
📁 NeMo model: /results/checkpoints/megatron_gpt_peft_lora_tuning.nemo
📁 Base model: meta-llama/CodeLlama-7b-hf
📁 Output: ./converted_model
🔗 Merge LoRA: True
📦 Importing required libraries...
📥 Loading base model: meta-llama/CodeLlama-7b-hf
📥 Loading NeMo LoRA model: /results/checkpoints/megatron_gpt_peft_lora_tuning.nemo
🔄 Extracting LoRA configuration...
🔧 Creating PEFT model...
⚖️ Transferring LoRA weights...
🔗 Merging LoRA weights with base model...
💾 Saving model to: ./converted_model
✅ Conversion completed successfully!
```

### **Test Output Example:**
```
🧪 Testing model with code generation prompts...
============================================================

🔍 Test 1: def factorial(n):
----------------------------------------
📝 Input: def factorial(n):
🤖 Generated:
def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)

# Test the function
print(factorial(5))  # Output: 120
```

## 🐛 Troubleshooting

### **Common Issues:**

#### **1. Import Errors**
```bash
# If you get "No module named 'nemo'":
# Make sure you're in NeMo container
docker exec -it <container_name> bash

# If you get "No module named 'peft'":
pip install peft
```

#### **2. CUDA Memory Issues**
```bash
# Use CPU for conversion if GPU memory is limited
export CUDA_VISIBLE_DEVICES=""
python convert_lora_to_hf.py --simple
```

#### **3. Model Loading Errors**
```bash
# If NeMo model fails to load, check the path:
ls -la /results/checkpoints/*.nemo

# Try the official converter:
python convert_nemo_to_hf.py
```

#### **4. Weight Transfer Issues**
```bash
# If LoRA weight transfer fails, use simple mode:
python convert_lora_to_hf.py --simple
# This exports the base model, manual weight transfer needed
```

## 🚀 Using Converted Model

### **Local Inference**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./converted_model")
model = AutoModelForCausalLM.from_pretrained("./converted_model")

def generate_code(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
code = generate_code("def bubble_sort(arr):")
print(code)
```

### **Upload to HuggingFace Hub**
```python
from huggingface_hub import HfApi

# Upload your model (requires HF token)
api = HfApi()
api.upload_folder(
    folder_path="./converted_model",
    repo_id="your-username/codellama-7b-finetuned",
    repo_type="model"
)
```

## 📁 File Structure After Conversion

```
converted_model/
├── 📄 config.json                 # Model configuration
├── 📄 pytorch_model.bin           # Model weights
├── 📄 tokenizer.json             # Tokenizer
├── 📄 tokenizer_config.json      # Tokenizer config
├── 📄 special_tokens_map.json    # Special tokens
├── 📄 README.md                  # Model card
└── 📄 training_info.txt          # Training details
```

## 🎉 Success Indicators

✅ **Conversion Successful If:**
- No error messages during conversion
- All expected files are created in output directory
- Test script generates reasonable code
- Model loads without errors in transformers

✅ **Model Quality Check:**
- Generated code is syntactically correct
- Code follows the patterns from your training data
- Model responds appropriately to different prompts
- Performance is better than base model on your specific tasks

Your model with **validation_loss=0.690** after 50 steps shows good training progress and should convert successfully! 🚀
