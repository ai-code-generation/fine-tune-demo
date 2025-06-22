#!/usr/bin/env python3
"""
Example script demonstrating model merging functionality.
This shows how to merge LoRA adapters with base models for production deployment.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir / "src"))

from src.model_setup import ModelSetup


def create_example_configs():
    """Create example configuration files for demonstration."""
    
    # Example model configuration
    model_config = """
model:
  name: "codellama/CodeLlama-7b-Instruct-hf"
  model_type: "llama"
  torch_dtype: "bfloat16"
  device_map: "auto"
  trust_remote_code: true
  use_cache: false

tokenizer:
  name: "codellama/CodeLlama-7b-Instruct-hf"
  padding_side: "right"
  truncation_side: "right"
  add_eos_token: true
  add_bos_token: true

training:
  max_length: 2048
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  num_epochs: 3.0
  warmup_steps: 100
  logging_steps: 10
  save_steps: 500
  eval_steps: 500
  save_total_limit: 3
  dataloader_num_workers: 0
  remove_unused_columns: false
  optim: "adamw_torch"
  lr_scheduler_type: "cosine"
  weight_decay: 0.001
  max_grad_norm: 0.3
  group_by_length: true
  ddp_find_unused_parameters: false

quantization:
  load_in_4bit: false
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: "nf4"

memory_optimization:
  gradient_checkpointing: true
  dataloader_pin_memory: true
  fp16: false
  bf16: true
"""

    # Example LoRA configuration
    lora_config = """
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  bias: "none"
  task_type: "CAUSAL_LM"
"""

    # Write to temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(model_config)
        model_config_path = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(lora_config)
        lora_config_path = f.name
    
    return model_config_path, lora_config_path


def demonstrate_merge_info_creation():
    """Demonstrate how merge information is created."""
    print("=" * 60)
    print("DEMONSTRATING MERGE INFO CREATION")
    print("=" * 60)
    
    # Create example configs
    model_config_path, lora_config_path = create_example_configs()
    
    try:
        # Initialize model setup
        model_setup = ModelSetup(model_config_path, lora_config_path)
        
        # Create temporary directories for demonstration
        with tempfile.TemporaryDirectory() as temp_dir:
            lora_path = Path(temp_dir) / "lora_model"
            merged_path = Path(temp_dir) / "merged_model"
            merged_path.mkdir(parents=True)
            
            # Create merge info
            model_setup._create_merge_info(str(lora_path), str(merged_path))
            
            # Show the created merge info
            merge_info_path = merged_path / "merge_info.json"
            if merge_info_path.exists():
                print(f"✅ Merge info created at: {merge_info_path}")
                
                import json
                with open(merge_info_path, 'r') as f:
                    merge_info = json.load(f)
                
                print("\nMerge Information Contents:")
                for key, value in merge_info.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for sub_key, sub_value in value.items():
                            print(f"    {sub_key}: {sub_value}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print("❌ Merge info file was not created")
    
    finally:
        # Cleanup
        os.unlink(model_config_path)
        os.unlink(lora_config_path)


def demonstrate_merge_workflow():
    """Demonstrate the complete merge workflow (without actual model loading)."""
    print("\n" + "=" * 60)
    print("DEMONSTRATING MERGE WORKFLOW")
    print("=" * 60)
    
    print("This example shows the merge workflow without loading actual models.")
    print("In a real scenario, you would:")
    print()
    print("1. Train a LoRA model using the training pipeline:")
    print("   python train.py --model-config configs/model_configs/codellama_7b.yaml \\")
    print("                   --train-data data/train.yaml")
    print()
    print("2. The training pipeline automatically merges the model (default behavior)")
    print("   - Creates output/merged_model/ directory")
    print("   - Contains complete deployable model")
    print()
    print("3. Or merge manually using the standalone script:")
    print("   python scripts/merge_lora_model.py --lora-model output/my_model \\")
    print("                                      --output merged_models/production_model")
    print()
    print("4. Deploy the merged model to production:")
    print("   - Copy the merged_model directory to your production environment")
    print("   - Load using standard transformers library:")
    print("     from transformers import AutoModelForCausalLM, AutoTokenizer")
    print("     model = AutoModelForCausalLM.from_pretrained('path/to/merged_model')")
    print("     tokenizer = AutoTokenizer.from_pretrained('path/to/merged_model')")


def demonstrate_benefits():
    """Demonstrate the benefits of model merging."""
    print("\n" + "=" * 60)
    print("BENEFITS OF MODEL MERGING")
    print("=" * 60)
    
    benefits = [
        ("Self-contained deployment", "No need for base model in production"),
        ("Faster inference", "No adapter overhead during inference"),
        ("Simplified architecture", "Single model file instead of base + adapter"),
        ("Production ready", "Includes tokenizer and configuration"),
        ("Easy distribution", "Single directory contains everything needed"),
        ("Version control friendly", "Complete model state in one location")
    ]
    
    for benefit, description in benefits:
        print(f"✅ {benefit}: {description}")


def main():
    """Main demonstration function."""
    print("CodeLlama Fine-tuning Pipeline - Model Merging Example")
    print("This script demonstrates the model merging functionality.")
    print()
    
    try:
        # Demonstrate merge info creation
        demonstrate_merge_info_creation()
        
        # Demonstrate merge workflow
        demonstrate_merge_workflow()
        
        # Demonstrate benefits
        demonstrate_benefits()
        
        print("\n" + "=" * 60)
        print("EXAMPLE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("For actual model merging, use:")
        print("1. Automatic merging during training (default)")
        print("2. Manual merging with scripts/merge_lora_model.py")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
