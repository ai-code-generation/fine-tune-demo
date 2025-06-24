#!/usr/bin/env python3
"""
Simple script to extract the base model from NeMo and save as HuggingFace format.
This bypasses the complex conversion issues by working with the base model directly.
"""

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def extract_base_model(base_model_name, output_dir, add_training_info=True):
    """Extract and save the base model in HuggingFace format."""
    
    print(f"📥 Loading base model: {base_model_name}")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        print(f"📊 Model info:")
        print(f"   - Parameters: {model.num_parameters():,}")
        print(f"   - Model type: {type(model).__name__}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"💾 Saving model to: {output_dir}")
        
        # Save model and tokenizer
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        
        if add_training_info:
            # Create info file about the extraction
            with open(os.path.join(output_dir, "MODEL_INFO.txt"), "w") as f:
                f.write("BASE MODEL EXTRACTION\n")
                f.write("=" * 25 + "\n\n")
                f.write(f"Base model: {base_model_name}\n")
                f.write(f"Parameters: {model.num_parameters():,}\n")
                f.write(f"Model type: {type(model).__name__}\n")
                f.write(f"Precision: bfloat16\n\n")
                f.write("Status: Base model successfully extracted\n")
                f.write("Note: This is the original base model without fine-tuning\n")
                f.write("To use your fine-tuned weights, you'll need to:\n")
                f.write("1. Load this base model\n")
                f.write("2. Apply your LoRA weights manually\n")
                f.write("3. Or use the NeMo model directly for inference\n")
            
            # Create a simple usage example
            with open(os.path.join(output_dir, "usage_example.py"), "w") as f:
                f.write(f'''#!/usr/bin/env python3
"""
Usage example for the extracted base model.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model():
    """Load the extracted model."""
    tokenizer = AutoTokenizer.from_pretrained(".")
    model = AutoModelForCausalLM.from_pretrained(".", torch_dtype=torch.bfloat16)
    return tokenizer, model

def generate_code(prompt, max_length=200, temperature=0.7):
    """Generate code using the model."""
    tokenizer, model = load_model()
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Test the model
    prompt = "def factorial(n):"
    result = generate_code(prompt)
    print("Generated code:")
    print(result)
''')
        
        print("✅ Base model extracted successfully!")
        print(f"📁 Files created:")
        for file in os.listdir(output_dir):
            print(f"   - {file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to extract base model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_extracted_model(model_dir):
    """Test the extracted model with a simple prompt."""
    
    print(f"🧪 Testing extracted model in: {model_dir}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Load the model
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
        
        # Test with a simple prompt
        prompt = "def hello_world():"
        print(f"📝 Testing with prompt: {prompt}")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🤖 Generated:")
        print(generated)
        
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False

def create_inference_script(output_dir, nemo_model_path):
    """Create a script for inference using the original NeMo model."""
    
    script_content = f'''#!/usr/bin/env python3
"""
Inference script using the original NeMo model.
This bypasses conversion issues by using NeMo directly.
"""

import os
import sys

def nemo_inference(prompt, max_tokens=200):
    """Run inference using the NeMo model directly."""
    
    try:
        from nemo.collections.nlp.models import MegatronGPTModel
        
        print("📥 Loading NeMo model...")
        model = MegatronGPTModel.restore_from(
            "{nemo_model_path}",
            map_location="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        )
        
        print("🤖 Generating response...")
        response = model.generate([prompt], length_params={{"max_length": max_tokens}})
        
        return response[0] if response else "Generation failed"
        
    except Exception as e:
        print(f"❌ NeMo inference failed: {{e}}")
        return None

def main():
    """Main inference function."""
    
    # Test prompts
    prompts = [
        "def factorial(n):",
        "# Create a function to add two numbers\\ndef add(",
        "class Calculator:",
    ]
    
    print("🎯 NeMo Model Direct Inference")
    print("=" * 35)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\\n🔍 Test {{i}}: {{prompt}}")
        print("-" * 30)
        
        result = nemo_inference(prompt)
        if result:
            print(f"🤖 Generated:\\n{{result}}")
        else:
            print("❌ Generation failed")

if __name__ == "__main__":
    main()
'''
    
    script_path = os.path.join(output_dir, "nemo_inference.py")
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod(script_path, 0o755)
    
    print(f"✅ Created NeMo inference script: {script_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract base model to HuggingFace format")
    parser.add_argument("--base-model", 
                       default="meta-llama/CodeLlama-7b-hf",
                       help="Base model to extract")
    parser.add_argument("--output-dir", 
                       default="./base_model_extracted",
                       help="Output directory")
    parser.add_argument("--test", 
                       action="store_true",
                       help="Test the extracted model")
    parser.add_argument("--nemo-model",
                       default="/results/checkpoints/megatron_gpt_peft_lora_tuning.nemo",
                       help="Path to NeMo model (for creating inference script)")
    
    args = parser.parse_args()
    
    print("📦 Base Model Extractor")
    print("=" * 25)
    print(f"📁 Base model: {args.base_model}")
    print(f"📁 Output: {args.output_dir}")
    
    # Extract the base model
    success = extract_base_model(args.base_model, args.output_dir)
    
    if success:
        # Create NeMo inference script
        if os.path.exists(args.nemo_model):
            create_inference_script(args.output_dir, args.nemo_model)
        
        # Test if requested
        if args.test:
            test_extracted_model(args.output_dir)
        
        print("\n🎉 Extraction completed successfully!")
        print(f"📁 Base model saved to: {args.output_dir}")
        print("\n💡 Usage options:")
        print("1. Use the extracted HuggingFace model (base model only)")
        print("2. Use nemo_inference.py for inference with your fine-tuned model")
        print("3. Manually merge LoRA weights with the base model")
        
    else:
        print("\n❌ Extraction failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
