"""
Utility functions for the CodeLlama fine-tuning pipeline.
"""

import os
import shutil
import logging
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logger = logging.getLogger(__name__)


def copy_model_to_output(source_dir: str, output_dir: str, model_name: str = None):
    """
    Copy trained model to final output directory with proper organization.
    
    Args:
        source_dir: Source directory containing the trained model
        output_dir: Final output directory
        model_name: Optional name for the model (defaults to timestamp)
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate model name if not provided
    if model_name is None:
        import time
        model_name = f"codellama-finetuned-{int(time.time())}"
    
    final_model_dir = output_path / model_name
    
    # Copy the model
    if final_model_dir.exists():
        shutil.rmtree(final_model_dir)
    
    shutil.copytree(source_path, final_model_dir)
    
    logger.info(f"Model copied to: {final_model_dir}")
    return str(final_model_dir)


def load_finetuned_model(model_path: str, base_model_name: str = None):
    """
    Load a fine-tuned model for inference.
    
    Args:
        model_path: Path to the fine-tuned model
        base_model_name: Base model name (auto-detected if not provided)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Try to load model info to get base model name
    if base_model_name is None:
        info_file = model_path / "model_info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                model_info = json.load(f)
                base_model_name = model_info.get('model_name')
    
    if base_model_name is None:
        raise ValueError("Base model name not found. Please provide base_model_name parameter.")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load base model
    logger.info(f"Loading base model {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load PEFT model
    logger.info(f"Loading PEFT adapters from {model_path}")
    model = PeftModel.from_pretrained(base_model, str(model_path))
    
    logger.info("Model loaded successfully")
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_length: int = 512, **kwargs):
    """
    Generate text using the fine-tuned model.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_length: Maximum generation length
        **kwargs: Additional generation parameters
        
    Returns:
        Generated text
    """
    # Format prompt for CodeLlama instruction format
    formatted_prompt = f"[INST] {prompt} [/INST]"
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the input prompt from the output
    response = generated_text[len(formatted_prompt):].strip()
    
    return response


def validate_model_output(model_dir: str) -> bool:
    """
    Validate that a model directory contains all necessary files.
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        True if valid, False otherwise
    """
    model_path = Path(model_dir)
    
    if not model_path.exists():
        logger.error(f"Model directory does not exist: {model_dir}")
        return False
    
    # Check for required files
    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "training_args.bin"
    ]
    
    for file_name in required_files:
        file_path = model_path / file_name
        if not file_path.exists():
            logger.error(f"Required file missing: {file_path}")
            return False
    
    logger.info(f"Model validation passed for: {model_dir}")
    return True


def get_model_size(model_dir: str) -> Dict[str, Any]:
    """
    Get information about model size and files.
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        Dictionary with size information
    """
    model_path = Path(model_dir)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    total_size = 0
    file_info = {}
    
    for file_path in model_path.rglob("*"):
        if file_path.is_file():
            size = file_path.stat().st_size
            total_size += size
            file_info[str(file_path.relative_to(model_path))] = {
                "size_bytes": size,
                "size_mb": size / (1024 * 1024)
            }
    
    return {
        "total_size_bytes": total_size,
        "total_size_mb": total_size / (1024 * 1024),
        "total_size_gb": total_size / (1024 * 1024 * 1024),
        "file_count": len(file_info),
        "files": file_info
    }


def cleanup_checkpoints(output_dir: str, keep_best: bool = True, keep_last: bool = True):
    """
    Clean up training checkpoints to save space.
    
    Args:
        output_dir: Training output directory
        keep_best: Whether to keep the best checkpoint
        keep_last: Whether to keep the last checkpoint
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        logger.warning(f"Output directory not found: {output_dir}")
        return
    
    # Find checkpoint directories
    checkpoint_dirs = [d for d in output_path.iterdir() 
                      if d.is_dir() and d.name.startswith("checkpoint-")]
    
    if not checkpoint_dirs:
        logger.info("No checkpoints found to clean up")
        return
    
    # Sort by checkpoint number
    checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[1]))
    
    # Determine which checkpoints to keep
    to_keep = set()
    
    if keep_last and checkpoint_dirs:
        to_keep.add(checkpoint_dirs[-1])
    
    if keep_best:
        # Look for trainer_state.json to find best checkpoint
        trainer_state_file = output_path / "trainer_state.json"
        if trainer_state_file.exists():
            with open(trainer_state_file, 'r') as f:
                trainer_state = json.load(f)
                best_checkpoint = trainer_state.get("best_model_checkpoint")
                if best_checkpoint:
                    best_path = Path(best_checkpoint)
                    if best_path.exists():
                        to_keep.add(best_path)
    
    # Remove checkpoints not in keep list
    removed_count = 0
    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir not in to_keep:
            shutil.rmtree(checkpoint_dir)
            removed_count += 1
            logger.info(f"Removed checkpoint: {checkpoint_dir.name}")
    
    logger.info(f"Cleanup complete. Removed {removed_count} checkpoints, kept {len(to_keep)}")


def create_model_card(model_dir: str, training_info: Dict[str, Any]):
    """
    Create a model card with training information.
    
    Args:
        model_dir: Model directory
        training_info: Training information dictionary
    """
    model_path = Path(model_dir)
    
    model_card_content = f"""# CodeLlama Fine-tuned Model

This model was fine-tuned using the CodeLlama fine-tuning pipeline.

## Model Information

- **Base Model**: {training_info.get('model_name', 'Unknown')}
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Rank**: {training_info.get('lora_r', 'Unknown')}
- **LoRA Alpha**: {training_info.get('lora_alpha', 'Unknown')}
- **Target Modules**: {', '.join(training_info.get('target_modules', []))}

## Training Information

- **Training Date**: {training_info.get('timestamp', 'Unknown')}
- **Max Sequence Length**: {training_info.get('max_length', 'Unknown')}
- **Output Directory**: {training_info.get('output_dir', 'Unknown')}

## Usage

```python
from src.utils import load_finetuned_model, generate_text

# Load the model
model, tokenizer = load_finetuned_model("{model_dir}")

# Generate text
prompt = "Write a Python function to calculate factorial"
response = generate_text(model, tokenizer, prompt)
print(response)
```

## Training Data Format

The model was trained on conversational data in YAML format with system/user/assistant roles.

## License

Please refer to the original CodeLlama license for usage terms.
"""
    
    model_card_path = model_path / "README.md"
    with open(model_card_path, 'w', encoding='utf-8') as f:
        f.write(model_card_content)
    
    logger.info(f"Model card created: {model_card_path}")
