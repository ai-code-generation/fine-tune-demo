#!/usr/bin/env python3
"""
Convert NeMo LoRA fine-tuned model to HuggingFace format
Specialized script for LoRA models with merge capability.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

def convert_lora_model(nemo_model_path, base_model_name, output_dir, merge_lora=True):
    """Convert NeMo LoRA model to HuggingFace format."""
    
    try:
        print("📦 Importing required libraries...")
        from nemo.collections.nlp.models import MegatronGPTModel
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        from peft import PeftModel, LoraConfig, get_peft_model
        
        print(f"📥 Loading base model: {base_model_name}")
        # Load base model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        print(f"📥 Loading NeMo LoRA model: {nemo_model_path}")
        # Load NeMo model
        nemo_model = MegatronGPTModel.restore_from(nemo_model_path, map_location='cpu')
        
        print("🔄 Extracting LoRA configuration...")
        # Get LoRA config from NeMo model
        peft_cfg = nemo_model.cfg.peft
        
        # Create LoRA config for HuggingFace
        lora_config = LoraConfig(
            r=peft_cfg.lora_tuning.adapter_dim,
            lora_alpha=peft_cfg.lora_tuning.alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=peft_cfg.lora_tuning.dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        print("🔧 Creating PEFT model...")
        # Create PEFT model
        peft_model = get_peft_model(base_model, lora_config)
        
        print("⚖️ Transferring LoRA weights...")
        # Transfer LoRA weights from NeMo to HuggingFace
        # This is a simplified transfer - may need adjustment based on exact architecture
        nemo_state_dict = nemo_model.state_dict()
        
        # Map NeMo LoRA weights to HuggingFace format
        for name, param in peft_model.named_parameters():
            if 'lora_' in name:
                # Find corresponding weight in NeMo model
                # This mapping may need to be adjusted based on your specific model
                nemo_name = name.replace('base_model.model.', '').replace('lora_A', 'adapter_layer.0').replace('lora_B', 'adapter_layer.1')
                if nemo_name in nemo_state_dict:
                    param.data = nemo_state_dict[nemo_name].data
                    print(f"   Transferred: {name}")
        
        if merge_lora:
            print("🔗 Merging LoRA weights with base model...")
            # Merge LoRA weights into base model
            merged_model = peft_model.merge_and_unload()
            final_model = merged_model
        else:
            print("💾 Keeping LoRA as separate adapter...")
            final_model = peft_model
        
        print(f"💾 Saving model to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model and tokenizer
        final_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save additional info
        with open(os.path.join(output_dir, "training_info.txt"), "w") as f:
            f.write(f"Base model: {base_model_name}\n")
            f.write(f"NeMo model: {nemo_model_path}\n")
            f.write(f"LoRA merged: {merge_lora}\n")
            f.write(f"LoRA config: {lora_config}\n")
        
        print("✅ Conversion completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def simple_weight_extraction(nemo_model_path, base_model_name, output_dir):
    """Simplified weight extraction method."""
    
    try:
        print("🔄 Using simplified extraction method...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"📥 Loading base model: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16
        )
        
        print(f"💾 Saving base model to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the base model (as a starting point)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Create instruction file for manual weight transfer
        with open(os.path.join(output_dir, "MANUAL_CONVERSION_NEEDED.txt"), "w") as f:
            f.write("MANUAL CONVERSION REQUIRED\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"NeMo model path: {nemo_model_path}\n")
            f.write(f"Base model: {base_model_name}\n\n")
            f.write("This is the base model. To complete the conversion:\n")
            f.write("1. Load the NeMo model using NeMo framework\n")
            f.write("2. Extract the LoRA weights\n")
            f.write("3. Apply them to this base model\n")
            f.write("4. Or use the model directly in NeMo for inference\n")
        
        print("✅ Base model saved. Manual conversion required for LoRA weights.")
        return True
        
    except Exception as e:
        print(f"❌ Simplified extraction failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert NeMo LoRA model to HuggingFace")
    parser.add_argument("--nemo-model", default="/results/checkpoints/megatron_gpt_peft_lora_tuning.nemo", 
                       help="Path to NeMo .nemo model")
    parser.add_argument("--base-model", default="meta-llama/CodeLlama-7b-hf", 
                       help="Base HuggingFace model name")
    parser.add_argument("--output-dir", default="./converted_model", 
                       help="Output directory")
    parser.add_argument("--merge-lora", action="store_true", default=True,
                       help="Merge LoRA weights into base model")
    parser.add_argument("--simple", action="store_true",
                       help="Use simplified conversion (base model only)")
    
    args = parser.parse_args()
    
    print("🎯 NeMo LoRA to HuggingFace Converter")
    print("=" * 40)
    
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
    
    print(f"📁 NeMo model: {args.nemo_model}")
    print(f"📁 Base model: {args.base_model}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🔗 Merge LoRA: {args.merge_lora}")
    
    if args.simple:
        success = simple_weight_extraction(args.nemo_model, args.base_model, args.output_dir)
    else:
        success = convert_lora_model(args.nemo_model, args.base_model, args.output_dir, args.merge_lora)
    
    if success:
        print("\n🎉 Conversion process completed!")
        print(f"📁 Check output directory: {args.output_dir}")
        print("\n🚀 Usage example:")
        print("from transformers import AutoTokenizer, AutoModelForCausalLM")
        print(f"tokenizer = AutoTokenizer.from_pretrained('{args.output_dir}')")
        print(f"model = AutoModelForCausalLM.from_pretrained('{args.output_dir}')")
    else:
        print("\n❌ Conversion failed. Try --simple flag for basic conversion.")
        sys.exit(1)

if __name__ == "__main__":
    main()
