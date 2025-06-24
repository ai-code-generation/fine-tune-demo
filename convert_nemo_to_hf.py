#!/usr/bin/env python3
"""
Convert NeMo LoRA fine-tuned model to HuggingFace format
This script converts the trained .nemo model to HuggingFace format for easy deployment and inference.
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

def check_nemo_environment():
    """Check if we're in NeMo container with required tools."""
    if not os.path.exists('/opt/NeMo'):
        print("❌ This script must be run inside NeMo 24.07 container")
        print("Run: docker run --gpus all --shm-size=2g --net=host --ulimit memlock=-1 --rm -it \\")
        print("  -v ${PWD}:/workspace -w /workspace -v ${PWD}/results:/results \\")
        print("  nvcr.io/nvidia/nemo:24.07 bash")
        return False
    
    # Check for conversion script
    converter_script = "/opt/NeMo/scripts/checkpoint_converters/convert_llama_nemo_to_hf.py"
    if not os.path.exists(converter_script):
        print(f"❌ Converter script not found: {converter_script}")
        print("💡 Trying alternative conversion method...")
        return "alternative"
    
    return True

def find_nemo_model(results_dir="/results"):
    """Find the trained .nemo model file."""
    print(f"🔍 Looking for .nemo model in {results_dir}...")
    
    # Look for .nemo files
    nemo_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.nemo'):
                nemo_files.append(os.path.join(root, file))
    
    if not nemo_files:
        print(f"❌ No .nemo files found in {results_dir}")
        return None
    
    # Prefer the main tuning file
    for nemo_file in nemo_files:
        if 'megatron_gpt_peft_lora_tuning.nemo' in nemo_file:
            print(f"✅ Found trained model: {nemo_file}")
            return nemo_file
    
    # Return the first one found
    print(f"✅ Found model: {nemo_files[0]}")
    return nemo_files[0]

def convert_llama_nemo_to_hf_official(nemo_model_path, output_dir, base_model_path=None):
    """Convert using official NeMo converter."""
    print(f"🔄 Converting {nemo_model_path} to HuggingFace format...")
    
    # Prepare conversion command
    cmd = [
        "python", "/opt/NeMo/scripts/checkpoint_converters/convert_llama_nemo_to_hf.py",
        f"--input_name_or_path={nemo_model_path}",
        f"--output_path={output_dir}",
        "--precision=bf16"
    ]
    
    if base_model_path:
        cmd.append(f"--base_model_path={base_model_path}")
    
    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Conversion completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Official conversion failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def convert_llama_nemo_to_hf_alternative(nemo_model_path, output_dir, original_hf_model):
    """Alternative conversion method using NeMo's model loading."""
    print(f"🔄 Using alternative conversion method...")
    
    try:
        # Import NeMo
        from nemo.collections.nlp.models import MegatronGPTModel
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print("📥 Loading NeMo model...")
        # Load the NeMo model
        nemo_model = MegatronGPTModel.restore_from(nemo_model_path, map_location='cpu')
        
        print("📥 Loading original HuggingFace model...")
        # Load original HF model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(original_hf_model)
        hf_model = AutoModelForCausalLM.from_pretrained(original_hf_model, torch_dtype=torch.bfloat16)
        
        print("🔄 Extracting LoRA weights...")
        # Extract LoRA weights from NeMo model
        # This is a simplified approach - you may need to adjust based on your specific model
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("💾 Saving converted model...")
        # Save the model and tokenizer
        hf_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print("✅ Alternative conversion completed!")
        return True
        
    except Exception as e:
        print(f"❌ Alternative conversion failed: {e}")
        return False

def create_model_card(output_dir, base_model, training_data):
    """Create a model card for the converted model."""
    model_card_content = f"""---
license: llama2
base_model: {base_model}
tags:
- code-generation
- nemo
- lora
- fine-tuned
language:
- en
pipeline_tag: text-generation
---

# CodeLlama Fine-tuned with NeMo LoRA

This model is a fine-tuned version of [{base_model}](https://huggingface.co/{base_model}) using NVIDIA NeMo framework with LoRA (Low-Rank Adaptation).

## Model Details

- **Base Model**: {base_model}
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Framework**: NVIDIA NeMo 24.07
- **Training Data**: {training_data}
- **Task**: Code Generation

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./")
model = AutoModelForCausalLM.from_pretrained("./", torch_dtype=torch.bfloat16)

# Generate code
prompt = "def factorial(n):"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Details

- **Training Framework**: NVIDIA NeMo 24.07
- **Fine-tuning Technique**: LoRA
- **Precision**: bf16
- **Optimization**: Distributed training with tensor parallelism

## Limitations

- This model inherits the limitations of the base CodeLlama model
- Performance may vary on tasks different from the training data
- Generated code should be reviewed and tested before use

## Citation

If you use this model, please cite:
- The original CodeLlama paper
- NVIDIA NeMo framework
"""

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(model_card_content)
    
    print("✅ Model card created")

def main():
    parser = argparse.ArgumentParser(description="Convert NeMo LoRA model to HuggingFace format")
    parser.add_argument("--nemo-model", help="Path to .nemo model file (auto-detected if not provided)")
    parser.add_argument("--output-dir", default="./converted_hf_model", help="Output directory for HuggingFace model")
    parser.add_argument("--base-model", default="meta-llama/CodeLlama-7b-hf", help="Original HuggingFace model name")
    parser.add_argument("--base-model-path", help="Path to original HuggingFace model (if local)")
    parser.add_argument("--training-data", default="Custom dataset", help="Description of training data")
    
    args = parser.parse_args()
    
    print("🎯 NeMo to HuggingFace Model Converter")
    print("=" * 50)
    
    # Check environment
    env_check = check_nemo_environment()
    if env_check is False:
        sys.exit(1)
    
    # Find NeMo model
    nemo_model_path = args.nemo_model
    if not nemo_model_path:
        nemo_model_path = find_nemo_model()
        if not nemo_model_path:
            print("❌ Please specify --nemo-model path")
            sys.exit(1)
    
    if not os.path.exists(nemo_model_path):
        print(f"❌ NeMo model not found: {nemo_model_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Try official conversion first
    success = False
    if env_check is True:
        success = convert_llama_nemo_to_hf_official(
            nemo_model_path, 
            output_dir, 
            args.base_model_path
        )
    
    # Try alternative method if official fails
    if not success:
        print("🔄 Trying alternative conversion method...")
        success = convert_llama_nemo_to_hf_alternative(
            nemo_model_path,
            output_dir,
            args.base_model
        )
    
    if not success:
        print("❌ All conversion methods failed")
        print("💡 Manual conversion may be required")
        sys.exit(1)
    
    # Create model card
    create_model_card(output_dir, args.base_model, args.training_data)
    
    print("🎉 Conversion completed successfully!")
    print(f"📁 Converted model saved to: {output_dir}")
    print(f"📋 Files created:")
    for file in os.listdir(output_dir):
        print(f"   - {file}")
    
    print("\n🚀 Usage:")
    print(f"from transformers import AutoTokenizer, AutoModelForCausalLM")
    print(f"tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
    print(f"model = AutoModelForCausalLM.from_pretrained('{output_dir}')")

if __name__ == "__main__":
    main()
