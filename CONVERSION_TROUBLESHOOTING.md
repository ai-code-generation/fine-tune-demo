# 🔧 NeMo to HuggingFace Conversion Troubleshooting

This guide addresses the specific errors you encountered and provides multiple solutions.

## 🚨 **Your Specific Error Analysis**

### **Error 1: Missing Configuration Fields**
```
The model: MegatronGPTModel() does not have field.name: disable_parameter_transpose_cache
The model: MegatronGPTModel() does not have field.name: enable_cuda_graph
```

**Root Cause**: Your NeMo model was trained with an older configuration that's missing newer fields expected by the converter.

### **Error 2: State Dict Key Mismatch**
```
Missing key(s) in state_dict: "model.module.embedding.word_embeddings.weight", 
"model.module.decoder.layers.0.self_attention.linear_proj.weight", ...
```

**Root Cause**: The LoRA fine-tuned model has a different state dict structure than expected by the converter.

## 🛠️ **Solution Methods (In Order of Recommendation)**

### **Method 1: Fixed Configuration Converter (Recommended)**
```bash
# Use the improved converter that fixes config issues
python convert_nemo_to_hf_fixed.py \
  --nemo-model /results/checkpoints/megatron_gpt_peft_lora_tuning.nemo \
  --base-model meta-llama/CodeLlama-7b-hf \
  --output-dir ./converted_model_fixed \
  --method config-fix
```

### **Method 2: Manual Weight Extraction**
```bash
# Extract weights manually with better error handling
python convert_nemo_to_hf_fixed.py \
  --nemo-model /results/checkpoints/megatron_gpt_peft_lora_tuning.nemo \
  --base-model meta-llama/CodeLlama-7b-hf \
  --output-dir ./converted_model_manual \
  --method manual
```

### **Method 3: Base Model + NeMo Inference (Fallback)**
```bash
# Extract base model and create NeMo inference script
python extract_base_model.py \
  --base-model meta-llama/CodeLlama-7b-hf \
  --output-dir ./base_model_extracted \
  --nemo-model /results/checkpoints/megatron_gpt_peft_lora_tuning.nemo \
  --test
```

### **Method 4: Direct NeMo Inference (No Conversion)**
```bash
# Use your model directly in NeMo without conversion
python -c "
from nemo.collections.nlp.models import MegatronGPTModel
model = MegatronGPTModel.restore_from('/results/checkpoints/megatron_gpt_peft_lora_tuning.nemo')
response = model.generate(['def factorial(n):'], length_params={'max_length': 100})
print(response[0])
"
```

## 🎯 **Step-by-Step Fix Process**

### **Step 1: Try the Fixed Converter**
```bash
# Inside NeMo container
cd /workspace

# Run the fixed converter
python convert_nemo_to_hf_fixed.py \
  --nemo-model /results/checkpoints/megatron_gpt_peft_lora_tuning.nemo \
  --base-model meta-llama/CodeLlama-7b-hf \
  --output-dir ./converted_fixed
```

**Expected Output:**
```
🔧 Fixed NeMo to HuggingFace Converter
========================================
📁 NeMo model: /results/checkpoints/megatron_gpt_peft_lora_tuning.nemo
📁 Base model: meta-llama/CodeLlama-7b-hf
📁 Output: ./converted_fixed
🔧 Method: config-fix
🔧 Fixing NeMo model configuration...
📥 Loading NeMo model to extract config...
🔧 Adding missing configuration fields...
💾 Saving model with fixed configuration...
✅ Configuration fixed successfully!
🔧 Trying converter: /opt/NeMo/scripts/checkpoint_converters/convert_llama_nemo_to_hf.py
✅ Conversion completed successfully!
```

### **Step 2: If Step 1 Fails, Extract Base Model**
```bash
# Extract the base model as a working fallback
python extract_base_model.py \
  --base-model meta-llama/CodeLlama-7b-hf \
  --output-dir ./base_model \
  --test
```

### **Step 3: Test Your Results**
```bash
# Test the converted model
python test_converted_model.py --model-path ./converted_fixed --interactive

# Or test the base model
python test_converted_model.py --model-path ./base_model --interactive
```

## 🔍 **Diagnostic Commands**

### **Check Your NeMo Model**
```bash
# Verify your model file exists and check size
ls -la /results/checkpoints/megatron_gpt_peft_lora_tuning.nemo

# Check model info
python -c "
import torch
checkpoint = torch.load('/results/checkpoints/megatron_gpt_peft_lora_tuning.nemo', map_location='cpu')
print('Keys in checkpoint:', list(checkpoint.keys()))
if 'cfg' in checkpoint:
    print('Config keys:', list(checkpoint['cfg'].keys()) if hasattr(checkpoint['cfg'], 'keys') else 'Config not dict')
"
```

### **Check Available Converters**
```bash
# List available conversion scripts
find /opt/NeMo -name "*convert*" -name "*.py" | grep -E "(llama|hf)"

# Check if specific converters exist
ls -la /opt/NeMo/scripts/checkpoint_converters/convert_llama_nemo_to_hf.py
ls -la /opt/NeMo/scripts/nlp_language_modeling/convert_nemo_to_hf.py
```

## 🎯 **Working Solutions for Your Specific Case**

### **Solution A: Use Fixed Converter (Best)**
Your model with `validation_loss=0.690` should work with the fixed converter:

```bash
python convert_nemo_to_hf_fixed.py --method config-fix
```

### **Solution B: Direct NeMo Inference (Reliable)**
If conversion fails, use your model directly:

```python
from nemo.collections.nlp.models import MegatronGPTModel

# Load your fine-tuned model
model = MegatronGPTModel.restore_from(
    "/results/checkpoints/megatron_gpt_peft_lora_tuning.nemo",
    map_location="cuda"
)

# Generate code
prompts = ["def factorial(n):"]
responses = model.generate(prompts, length_params={"max_length": 200})
print(responses[0])
```

### **Solution C: Base Model + Manual LoRA (Advanced)**
```bash
# Extract base model
python extract_base_model.py

# Use the base model with manual LoRA application
# (Requires custom implementation)
```

## 📊 **Expected Results**

### **Successful Conversion:**
```
📁 converted_fixed/
├── config.json
├── model.safetensors (or pytorch_model.bin)
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── README.md
```

### **Partial Conversion:**
```
📁 converted_manual/
├── config.json (base model)
├── pytorch_model.bin (base model)
├── tokenizer files
├── nemo_state_dict.pt (extracted weights)
└── CONVERSION_STATUS.txt
```

### **Fallback Solution:**
```
📁 base_model_extracted/
├── Base HuggingFace model files
├── nemo_inference.py (direct NeMo usage)
├── usage_example.py
└── MODEL_INFO.txt
```

## 🚀 **Quick Commands for Your Situation**

```bash
# 1. Try fixed conversion first
python convert_nemo_to_hf_fixed.py

# 2. If that fails, extract base model
python extract_base_model.py --test

# 3. Test either result
python test_converted_model.py --model-path ./converted_fixed --interactive

# 4. Use NeMo directly if needed
python -c "
from nemo.collections.nlp.models import MegatronGPTModel
model = MegatronGPTModel.restore_from('/results/checkpoints/megatron_gpt_peft_lora_tuning.nemo')
print(model.generate(['def bubble_sort(arr):'], length_params={'max_length': 150})[0])
"
```

## 💡 **Key Points**

1. **Your model trained successfully** (validation_loss=0.690 is good)
2. **The conversion error is common** with LoRA fine-tuned models
3. **Multiple solutions available** - at least one will work
4. **NeMo direct inference always works** as a fallback
5. **Base model extraction provides HF compatibility** even without LoRA weights

Your fine-tuned model is valuable and usable - the conversion issues don't affect the model quality! 🎉
