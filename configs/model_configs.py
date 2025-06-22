"""
Model configurations optimized for NeMo 24.07
Focus on CodeLlama-13B and Llama3 models (8B-70B)
"""

from typing import Dict, Any

# Model configurations optimized for NeMo 24.07
MODEL_CONFIGS = {
    # CodeLlama models (require HF access approval)
    "codellama-13b": {
        "hf_model_name": "meta-llama/CodeLlama-13b-hf",
        "num_layers": 40,
        "hidden_size": 5120,
        "ffn_hidden_size": 13824,
        "num_attention_heads": 40,
        "num_query_groups": 40,
        "max_position_embeddings": 16384,  # Updated for longer sequences
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "devices": 4,
        "num_nodes": 1,
        "micro_batch_size": 1,
        "global_batch_size": 8,
        "adapter_dim": 32,
        "learning_rate": 1e-4,
        "sequence_length": 4096,
        "model_type": "llama",
    },
    
    # Llama3 models (require HF access approval)
    "llama3-8b": {
        "hf_model_name": "meta-llama/Meta-Llama-3-8B",
        "num_layers": 32,
        "hidden_size": 4096,
        "ffn_hidden_size": 14336,
        "num_attention_heads": 32,
        "num_query_groups": 8,
        "max_position_embeddings": 8192,
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "devices": 4,
        "num_nodes": 1,
        "micro_batch_size": 2,
        "global_batch_size": 16,
        "adapter_dim": 32,
        "learning_rate": 1e-4,
        "sequence_length": 4096,
        "model_type": "llama",
    },
    
    "llama3-70b": {
        "hf_model_name": "meta-llama/Meta-Llama-3-70B",
        "num_layers": 80,
        "hidden_size": 8192,
        "ffn_hidden_size": 28672,
        "num_attention_heads": 64,
        "num_query_groups": 8,
        "max_position_embeddings": 8192,
        "tensor_model_parallel_size": 8,
        "pipeline_model_parallel_size": 2,
        "devices": 16,
        "num_nodes": 2,
        "micro_batch_size": 1,
        "global_batch_size": 8,
        "adapter_dim": 64,
        "learning_rate": 5e-5,
        "sequence_length": 4096,
        "model_type": "llama",
    },

    # StarCoder2 models (no HF access approval required)
    "starcoder2-15b": {
        "hf_model_name": "bigcode/starcoder2-15b",
        "num_layers": 40,
        "hidden_size": 6144,
        "ffn_hidden_size": 24576,
        "num_attention_heads": 48,
        "num_query_groups": 48,
        "max_position_embeddings": 16384,
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "devices": 4,
        "num_nodes": 1,
        "micro_batch_size": 1,
        "global_batch_size": 8,
        "adapter_dim": 32,
        "learning_rate": 1e-4,
        "sequence_length": 4096,
        "model_type": "starcoder2",
    },
}

# Hardware requirements for each model
HARDWARE_REQUIREMENTS = {
    "codellama-13b": {
        "min_gpu_memory_gb": 24,
        "recommended_gpu_memory_gb": 40,
        "min_gpus": 4,
        "recommended_gpus": 8,
        "min_system_ram_gb": 64,
        "recommended_system_ram_gb": 128,
    },
    "llama3-8b": {
        "min_gpu_memory_gb": 16,
        "recommended_gpu_memory_gb": 24,
        "min_gpus": 4,
        "recommended_gpus": 8,
        "min_system_ram_gb": 32,
        "recommended_system_ram_gb": 64,
    },
    "llama3-70b": {
        "min_gpu_memory_gb": 80,
        "recommended_gpu_memory_gb": 80,
        "min_gpus": 16,
        "recommended_gpus": 16,
        "min_system_ram_gb": 256,
        "recommended_system_ram_gb": 512,
    },
    "starcoder2-15b": {
        "min_gpu_memory_gb": 24,
        "recommended_gpu_memory_gb": 40,
        "min_gpus": 4,
        "recommended_gpus": 8,
        "min_system_ram_gb": 64,
        "recommended_system_ram_gb": 128,
    },
}

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model."""
    if model_name not in MODEL_CONFIGS:
        available_models = list(MODEL_CONFIGS.keys())
        raise ValueError(f"Model {model_name} not found. Available models: {available_models}")
    
    return MODEL_CONFIGS[model_name].copy()

def get_hardware_requirements(model_name: str) -> Dict[str, Any]:
    """Get hardware requirements for a specific model."""
    if model_name not in HARDWARE_REQUIREMENTS:
        available_models = list(HARDWARE_REQUIREMENTS.keys())
        raise ValueError(f"Model {model_name} not found. Available models: {available_models}")
    
    return HARDWARE_REQUIREMENTS[model_name].copy()

def list_available_models() -> list:
    """List all available model configurations."""
    return list(MODEL_CONFIGS.keys())

def validate_hardware(model_name: str, available_gpus: int, gpu_memory_gb: int) -> bool:
    """
    Validate if the available hardware meets the minimum requirements for the model.
    
    Args:
        model_name: Name of the model
        available_gpus: Number of available GPUs
        gpu_memory_gb: Memory per GPU in GB
    
    Returns:
        True if hardware meets minimum requirements, False otherwise
    """
    requirements = get_hardware_requirements(model_name)
    
    if available_gpus < requirements["min_gpus"]:
        print(f"Insufficient GPUs: {available_gpus} available, {requirements['min_gpus']} required")
        return False
    
    if gpu_memory_gb < requirements["min_gpu_memory_gb"]:
        print(f"Insufficient GPU memory: {gpu_memory_gb}GB available, {requirements['min_gpu_memory_gb']}GB required")
        return False
    
    return True

def get_recommended_config(model_name: str) -> Dict[str, Any]:
    """Get recommended configuration with optimizations for the model."""
    config = get_model_config(model_name)
    
    # Add NeMo 24.07 specific optimizations
    if "70b" in model_name.lower():
        config.update({
            "activations_checkpoint_granularity": "selective",
            "sequence_parallel": True,
            "fp8": True,
            "use_flash_attention": True,
        })
    elif "13b" in model_name.lower():
        config.update({
            "activations_checkpoint_granularity": "selective",
            "sequence_parallel": False,
            "use_flash_attention": True,
        })
    elif "8b" in model_name.lower():
        config.update({
            "use_flash_attention": True,
        })
    
    return config

def get_training_config_template(model_name: str) -> str:
    """Get the appropriate training configuration template for the model."""
    if "codellama" in model_name.lower():
        return "configs/codellama_13b_config.yaml"
    elif "starcoder2" in model_name.lower():
        return "configs/starcoder2_15b_config.yaml"
    elif "llama3" in model_name.lower():
        if "8b" in model_name.lower():
            return "configs/llama3_8b_config.yaml"
        elif "70b" in model_name.lower():
            return "configs/llama3_70b_config.yaml"

    # Default fallback
    return "configs/llama3_8b_config.yaml"

# NeMo 24.07 specific settings
NEMO_24_07_SETTINGS = {
    "use_transformer_engine": True,
    "fp8_training": True,
    "flash_attention": True,
    "sequence_parallel": True,
    "gradient_accumulation_fusion": True,
    "bias_activation_fusion": True,
}

def get_nemo_24_07_optimizations(model_name: str) -> Dict[str, Any]:
    """Get NeMo 24.07 specific optimizations for the model."""
    base_settings = NEMO_24_07_SETTINGS.copy()
    
    # Model-specific adjustments
    if "70b" in model_name.lower():
        base_settings.update({
            "fp8_training": True,
            "sequence_parallel": True,
            "activations_checkpoint_granularity": "selective",
        })
    elif "13b" in model_name.lower():
        base_settings.update({
            "fp8_training": False,  # May not be needed for 13B
            "sequence_parallel": False,
        })
    
    return base_settings
