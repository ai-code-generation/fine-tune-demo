#!/usr/bin/env python3
"""
Model configuration definitions for different model sizes and types.
This file contains the model specifications for CodeLlama and Llama3 models.
"""

from typing import Dict, Any

# Model configurations for different sizes
MODEL_CONFIGS = {
    "codellama-7b": {
        "hf_model_name": "meta-llama/CodeLlama-7b-hf",
        "num_layers": 32,
        "hidden_size": 4096,
        "ffn_hidden_size": 11008,
        "num_attention_heads": 32,
        "num_query_groups": 32,
        "max_position_embeddings": 4096,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "devices": 2,
        "num_nodes": 1,
        "micro_batch_size": 2,
        "global_batch_size": 8,
        "adapter_dim": 16,
        "learning_rate": 1e-4,
    },
    "codellama-13b": {
        "hf_model_name": "meta-llama/CodeLlama-13b-hf",
        "num_layers": 40,
        "hidden_size": 5120,
        "ffn_hidden_size": 13824,
        "num_attention_heads": 40,
        "num_query_groups": 40,
        "max_position_embeddings": 4096,
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "devices": 4,
        "num_nodes": 1,
        "micro_batch_size": 1,
        "global_batch_size": 8,
        "adapter_dim": 32,
        "learning_rate": 1e-4,
    },
    "codellama-34b": {
        "hf_model_name": "meta-llama/CodeLlama-34b-hf",
        "num_layers": 48,
        "hidden_size": 8192,
        "ffn_hidden_size": 22016,
        "num_attention_heads": 64,
        "num_query_groups": 8,
        "max_position_embeddings": 4096,
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 2,
        "devices": 8,
        "num_nodes": 1,
        "micro_batch_size": 1,
        "global_batch_size": 8,
        "adapter_dim": 64,
        "learning_rate": 5e-5,
    },
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
        "micro_batch_size": 1,
        "global_batch_size": 8,
        "adapter_dim": 32,
        "learning_rate": 1e-4,
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
        "devices": 8,
        "num_nodes": 2,
        "micro_batch_size": 1,
        "global_batch_size": 16,
        "adapter_dim": 64,
        "learning_rate": 5e-5,
    },
}

# Hardware requirements for each model
HARDWARE_REQUIREMENTS = {
    "codellama-7b": {
        "min_gpu_memory_gb": 16,
        "recommended_gpu_memory_gb": 24,
        "min_gpus": 2,
        "recommended_gpus": 4,
    },
    "codellama-13b": {
        "min_gpu_memory_gb": 24,
        "recommended_gpu_memory_gb": 40,
        "min_gpus": 4,
        "recommended_gpus": 8,
    },
    "codellama-34b": {
        "min_gpu_memory_gb": 40,
        "recommended_gpu_memory_gb": 80,
        "min_gpus": 8,
        "recommended_gpus": 8,
    },
    "llama3-8b": {
        "min_gpu_memory_gb": 16,
        "recommended_gpu_memory_gb": 24,
        "min_gpus": 2,
        "recommended_gpus": 4,
    },
    "llama3-70b": {
        "min_gpu_memory_gb": 80,
        "recommended_gpu_memory_gb": 80,
        "min_gpus": 16,
        "recommended_gpus": 16,
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
    
    # Add recommended optimizations based on model size
    if "70b" in model_name.lower():
        config.update({
            "activations_checkpoint_granularity": "selective",
            "sequence_parallel": True,
            "fp8": True,
        })
    elif "34b" in model_name.lower():
        config.update({
            "activations_checkpoint_granularity": "selective",
            "sequence_parallel": False,
        })
    
    return config
