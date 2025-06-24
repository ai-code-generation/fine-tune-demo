#!/usr/bin/env python3
"""
Fixed NeMo to HuggingFace converter that handles configuration and state dict issues.
This script addresses the specific errors encountered during conversion.
"""

import os
import sys
import argparse
import subprocess
import torch
import tempfile
import shutil
from pathlib import Path

def fix_nemo_config(nemo_model_path, output_path):
    """Fix NeMo model configuration to add missing fields."""
    print("🔧 Fixing NeMo model configuration...")
    
    try:
        from nemo.collections.nlp.models import MegatronGPTModel
        from omegaconf import OmegaConf
        
        # Load the model to get its config
        print("📥 Loading NeMo model to extract config...")
        model = MegatronGPTModel.restore_from(nemo_model_path, map_location='cpu', strict=False)
        
        # Get the config
        cfg = model.cfg
        
        # Add missing configuration fields
        print("🔧 Adding missing configuration fields...")
        
        # Add missing fields that cause warnings/errors
        if not hasattr(cfg, 'disable_parameter_transpose_cache'):
            cfg.disable_parameter_transpose_cache = False
            
        if not hasattr(cfg, 'enable_cuda_graph'):
            cfg.enable_cuda_graph = False
            
        # Add other potentially missing fields
        if not hasattr(cfg.model, 'sequence_parallel'):
            cfg.model.sequence_parallel = False
            
        if not hasattr(cfg.model, 'gradient_accumulation_fusion'):
            cfg.model.gradient_accumulation_fusion = False
            
        if not hasattr(cfg.model, 'bias_activation_fusion'):
            cfg.model.bias_activation_fusion = False
            
        if not hasattr(cfg.model, 'bias_dropout_add_fusion'):
            cfg.model.bias_dropout_add_fusion = False
            
        if not hasattr(cfg.model, 'masked_softmax_fusion'):
            cfg.model.masked_softmax_fusion = True
            
        if not hasattr(cfg.model, 'seed'):
            cfg.model.seed = 1234
            
        if not hasattr(cfg.model, 'use_cpu_initialization'):
            cfg.model.use_cpu_initialization = False
            
        # Save the model with fixed config
        print("💾 Saving model with fixed configuration...")
        model.save_to(output_path)
        
        print("✅ Configuration fixed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to fix configuration: {e}")
        return False

def convert_with_fixed_config(nemo_model_path, output_dir, base_model_name="meta-llama/CodeLlama-7b-hf"):
    """Convert NeMo model to HuggingFace with configuration fixes."""
    print("🔄 Converting with configuration fixes...")
    
    # Create temporary file for fixed model
    with tempfile.NamedTemporaryFile(suffix='.nemo', delete=False) as tmp_file:
        fixed_model_path = tmp_file.name
    
    try:
        # Fix the configuration first
        if not fix_nemo_config(nemo_model_path, fixed_model_path):
            return False
        
        # Try conversion with the fixed model
        converter_scripts = [
            "/opt/NeMo/scripts/checkpoint_converters/convert_llama_nemo_to_hf.py",
            "/opt/NeMo/scripts/nlp_language_modeling/convert_nemo_to_hf.py"
        ]
        
        for script in converter_scripts:
            if not os.path.exists(script):
                continue
                
            print(f"🔧 Trying converter: {script}")
            
            cmd = [
                "python", script,
                f"--input_name_or_path={fixed_model_path}",
                f"--output_path={output_dir}",
                "--precision=bf16",
                "--cpu_only"
            ]
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print("✅ Conversion completed successfully!")
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ Converter {script} failed: {e}")
                continue
        
        return False
        
    finally:
        # Clean up temporary file
        if os.path.exists(fixed_model_path):
            os.unlink(fixed_model_path)

def extract_lora_weights_manual(nemo_model_path, base_model_name, output_dir):
    """Manually extract LoRA weights and merge with base model."""
    print("🔄 Manual LoRA weight extraction...")
    
    try:
        from nemo.collections.nlp.models import MegatronGPTModel
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print("📥 Loading base HuggingFace model...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )
        
        print("📥 Loading NeMo model (this may take a while)...")
        # Try loading with strict=False to handle missing keys
        nemo_model = MegatronGPTModel.restore_from(
            nemo_model_path, 
            map_location='cpu',
            strict=False,
            override_config_path=None
        )
        
        print("🔄 Extracting model state...")
        # Get the state dict from NeMo model
        nemo_state_dict = nemo_model.state_dict()
        
        print("💾 Saving base model (LoRA weights need manual merging)...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the base model
        base_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save NeMo state dict for manual inspection
        torch.save(nemo_state_dict, os.path.join(output_dir, "nemo_state_dict.pt"))
        
        # Create instruction file
        with open(os.path.join(output_dir, "CONVERSION_STATUS.txt"), "w") as f:
            f.write("PARTIAL CONVERSION COMPLETED\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Base model: {base_model_name}\n")
            f.write(f"NeMo model: {nemo_model_path}\n\n")
            f.write("Files created:\n")
            f.write("- Base HuggingFace model (config.json, pytorch_model.bin, tokenizer files)\n")
            f.write("- nemo_state_dict.pt (NeMo model weights for manual inspection)\n\n")
            f.write("Status: Base model ready, LoRA weights extracted but not merged\n")
            f.write("The base model can be used as-is, or LoRA weights can be manually merged.\n")
        
        print("✅ Partial conversion completed!")
        print("💡 Base model saved. LoRA weights extracted but need manual merging.")
        return True
        
    except Exception as e:
        print(f"❌ Manual extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def simple_base_model_export(base_model_name, output_dir):
    """Export just the base model as a fallback."""
    print("🔄 Exporting base model as fallback...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"📥 Loading base model: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16
        )
        
        print(f"💾 Saving to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Create status file
        with open(os.path.join(output_dir, "FALLBACK_CONVERSION.txt"), "w") as f:
            f.write("FALLBACK CONVERSION - BASE MODEL ONLY\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Base model: {base_model_name}\n")
            f.write("Status: Only base model exported\n")
            f.write("Fine-tuned weights were not transferred\n")
            f.write("This model will behave like the original base model\n")
        
        print("✅ Base model exported successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Base model export failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fixed NeMo to HuggingFace converter")
    parser.add_argument("--nemo-model", 
                       default="/results/checkpoints/megatron_gpt_peft_lora_tuning.nemo",
                       help="Path to NeMo .nemo model")
    parser.add_argument("--base-model", 
                       default="meta-llama/CodeLlama-7b-hf",
                       help="Base HuggingFace model name")
    parser.add_argument("--output-dir", 
                       default="./converted_model_fixed",
                       help="Output directory")
    parser.add_argument("--method", 
                       choices=["config-fix", "manual", "base-only"],
                       default="config-fix",
                       help="Conversion method to use")
    
    args = parser.parse_args()
    
    print("🔧 Fixed NeMo to HuggingFace Converter")
    print("=" * 40)
    print(f"📁 NeMo model: {args.nemo_model}")
    print(f"📁 Base model: {args.base_model}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🔧 Method: {args.method}")
    
    # Check if NeMo model exists
    if not os.path.exists(args.nemo_model):
        print(f"❌ NeMo model not found: {args.nemo_model}")
        print("💡 Available models:")
        results_dir = "/results/checkpoints"
        if os.path.exists(results_dir):
            for file in os.listdir(results_dir):
                if file.endswith('.nemo'):
                    print(f"   - {os.path.join(results_dir, file)}")
        sys.exit(1)
    
    success = False
    
    if args.method == "config-fix":
        print("🔧 Trying configuration fix method...")
        success = convert_with_fixed_config(args.nemo_model, args.output_dir, args.base_model)
    
    if not success and args.method in ["manual", "config-fix"]:
        print("🔧 Trying manual extraction method...")
        success = extract_lora_weights_manual(args.nemo_model, args.base_model, args.output_dir)
    
    if not success:
        print("🔧 Using fallback: base model only...")
        success = simple_base_model_export(args.base_model, args.output_dir)
    
    if success:
        print("\n🎉 Conversion process completed!")
        print(f"📁 Check output directory: {args.output_dir}")
        print("\n📋 Files created:")
        for file in os.listdir(args.output_dir):
            print(f"   - {file}")
    else:
        print("\n❌ All conversion methods failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
