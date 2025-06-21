"""
Model configuration utilities.
"""

import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import logging

logger = logging.getLogger(__name__)


class ModelConfig:
    """Handles model configuration loading and management."""
    
    SUPPORTED_MODELS = {
        "llama2": {
            "7b": "configs/models/llama2_7b.yaml",
            "13b": "configs/models/llama2_13b.yaml",
            "70b": "configs/models/llama2_70b.yaml"
        },
        "llama3": {
            "8b": "configs/models/llama3_8b.yaml",
            "70b": "configs/models/llama3_70b.yaml"
        },
        "codellama": {
            "7b": "configs/models/codellama_7b.yaml",
            "13b": "configs/models/codellama_13b.yaml",
            "34b": "configs/models/codellama_34b.yaml"
        }
    }
    
    def __init__(self, model_type: str, model_size: str):
        """
        Initialize model configuration.
        
        Args:
            model_type: Type of model (llama2, llama3, codellama)
            model_size: Size of model (7b, 8b, 13b, etc.)
        """
        self.model_type = model_type.lower()
        self.model_size = model_size.lower()
        
        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported: {list(self.SUPPORTED_MODELS.keys())}")
        
        if self.model_size not in self.SUPPORTED_MODELS[self.model_type]:
            raise ValueError(f"Unsupported model size {model_size} for {model_type}. "
                           f"Supported: {list(self.SUPPORTED_MODELS[self.model_type].keys())}")
        
        self.config_path = self.SUPPORTED_MODELS[self.model_type][self.model_size]
        self.config = None
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> DictConfig:
        """
        Load model configuration from YAML file.
        
        Args:
            config_path: Optional custom config path
            
        Returns:
            OmegaConf configuration object
        """
        if config_path is None:
            config_path = self.config_path
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            self.config = OmegaConf.load(config_path)
            logger.info(f"Loaded config from {config_path}")
            return self.config
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {config_path}: {e}")
    
    def get_tokenizer_config(self) -> Dict[str, Any]:
        """Get tokenizer configuration."""
        if self.config is None:
            self.load_config()
        
        return OmegaConf.to_container(self.config.model.tokenizer, resolve=True)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model architecture configuration."""
        if self.config is None:
            self.load_config()
        
        # Extract model config excluding tokenizer and data
        model_config = OmegaConf.to_container(self.config.model, resolve=True)
        model_config.pop('tokenizer', None)
        model_config.pop('data', None)
        
        return model_config
    
    def get_lora_config(self) -> Dict[str, Any]:
        """Get LoRA configuration."""
        if self.config is None:
            self.load_config()
        
        if 'peft' in self.config.model and 'lora_tuning' in self.config.model.peft:
            return OmegaConf.to_container(self.config.model.peft.lora_tuning, resolve=True)
        
        return {}
    
    def get_optimizer_config(self) -> Dict[str, Any]:
        """Get optimizer configuration."""
        if self.config is None:
            self.load_config()
        
        return OmegaConf.to_container(self.config.model.optim, resolve=True)
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        if self.config is None:
            self.load_config()
        
        # Merge updates into config
        update_config = OmegaConf.create(updates)
        self.config = OmegaConf.merge(self.config, update_config)
        
        logger.info("Updated configuration with new values")
    
    def save_config(self, output_path: Union[str, Path]) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration
        """
        if self.config is None:
            raise RuntimeError("No configuration loaded")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            OmegaConf.save(self.config, f)
        
        logger.info(f"Saved configuration to {output_path}")
    
    def get_model_parallel_config(self) -> Dict[str, Any]:
        """Get model parallelism configuration."""
        if self.config is None:
            self.load_config()
        
        parallel_config = {}
        model_config = self.config.model
        
        parallel_keys = [
            'tensor_model_parallel_size',
            'pipeline_model_parallel_size',
            'virtual_pipeline_model_parallel_size',
            'sequence_parallel'
        ]
        
        for key in parallel_keys:
            if key in model_config:
                parallel_config[key] = model_config[key]
        
        return parallel_config
    
    def get_precision_config(self) -> str:
        """Get precision configuration."""
        if self.config is None:
            self.load_config()
        
        return self.config.model.get('precision', 'bf16')
    
    def get_sequence_length(self) -> int:
        """Get maximum sequence length."""
        if self.config is None:
            self.load_config()
        
        return self.config.model.get('encoder_seq_length', 2048)
    
    def get_special_tokens_config(self) -> Dict[str, str]:
        """Get special tokens configuration based on model type."""
        special_tokens = {}
        
        if self.model_type == "llama2":
            special_tokens = {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>"
            }
        elif self.model_type == "llama3":
            special_tokens = {
                "bos_token": "<|begin_of_text|>",
                "eos_token": "<|end_of_text|>",
                "pad_token": "<|pad|>"
            }
        elif self.model_type == "codellama":
            special_tokens = {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>"
            }
        
        return special_tokens
