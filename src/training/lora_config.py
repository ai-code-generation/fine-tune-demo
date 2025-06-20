"""
LoRA configuration management system.
"""

import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import logging

logger = logging.getLogger(__name__)


class LoRAConfig:
    """Manages LoRA configuration for different model types."""
    
    LORA_CONFIGS = {
        "llama2": "configs/lora/llama2_lora.yaml",
        "llama3": "configs/lora/llama3_lora.yaml", 
        "codellama": "configs/lora/codellama_lora.yaml"
    }
    
    def __init__(self, model_type: str):
        """
        Initialize LoRA configuration.
        
        Args:
            model_type: Type of model (llama2, llama3, codellama)
        """
        self.model_type = model_type.lower()
        
        if self.model_type not in self.LORA_CONFIGS:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported: {list(self.LORA_CONFIGS.keys())}")
        
        self.config_path = self.LORA_CONFIGS[self.model_type]
        self.config = None
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> DictConfig:
        """
        Load LoRA configuration from YAML file.
        
        Args:
            config_path: Optional custom config path
            
        Returns:
            OmegaConf configuration object
        """
        if config_path is None:
            config_path = self.config_path
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"LoRA config file not found: {config_path}")
        
        try:
            self.config = OmegaConf.load(config_path)
            logger.info(f"Loaded LoRA config from {config_path}")
            return self.config
        except Exception as e:
            raise RuntimeError(f"Failed to load LoRA config from {config_path}: {e}")
    
    def get_lora_params(self) -> Dict[str, Any]:
        """Get LoRA parameters."""
        if self.config is None:
            self.load_config()
        
        return OmegaConf.to_container(self.config.lora, resolve=True)
    
    def get_target_modules(self) -> List[str]:
        """Get target modules for LoRA adaptation."""
        if self.config is None:
            self.load_config()
        
        return self.config.lora.target_modules
    
    def get_rank_and_alpha(self) -> tuple[int, int]:
        """Get LoRA rank and alpha values."""
        if self.config is None:
            self.load_config()
        
        rank = self.config.lora.adapter_dim
        alpha = self.config.lora.alpha_scaling
        
        return rank, alpha
    
    def get_dropout(self) -> float:
        """Get LoRA dropout rate."""
        if self.config is None:
            self.load_config()
        
        return self.config.lora.adapter_dropout
    
    def get_layer_selection(self) -> Optional[List[int]]:
        """Get layer selection for LoRA (None means all layers)."""
        if self.config is None:
            self.load_config()
        
        return self.config.lora.get('layer_selection', None)
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """Get model-specific optimization configuration."""
        if self.config is None:
            self.load_config()
        
        if 'model_optimizations' in self.config:
            return OmegaConf.to_container(self.config.model_optimizations, resolve=True)
        
        return {}
    
    def get_quantization_config(self) -> Dict[str, Any]:
        """Get quantization configuration."""
        if self.config is None:
            self.load_config()
        
        return OmegaConf.to_container(self.config.lora.quantization, resolve=True)
    
    def is_quantization_enabled(self) -> bool:
        """Check if quantization is enabled."""
        quant_config = self.get_quantization_config()
        return quant_config.get('enabled', False)
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update LoRA configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        if self.config is None:
            self.load_config()
        
        # Merge updates into config
        update_config = OmegaConf.create(updates)
        self.config = OmegaConf.merge(self.config, update_config)
        
        logger.info("Updated LoRA configuration with new values")
    
    def save_config(self, output_path: Union[str, Path]) -> None:
        """
        Save current LoRA configuration to file.
        
        Args:
            output_path: Path to save configuration
        """
        if self.config is None:
            raise RuntimeError("No LoRA configuration loaded")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            OmegaConf.save(self.config, f)
        
        logger.info(f"Saved LoRA configuration to {output_path}")
    
    def create_nemo_lora_config(self) -> Dict[str, Any]:
        """
        Create NeMo-compatible LoRA configuration.
        
        Returns:
            Dictionary with NeMo LoRA configuration
        """
        if self.config is None:
            self.load_config()
        
        lora_params = self.get_lora_params()
        rank, alpha = self.get_rank_and_alpha()
        
        nemo_config = {
            "peft_scheme": "lora",
            "restore_from_path": None,
            "lora_tuning": {
                "target_modules": self.get_target_modules(),
                "adapter_dim": rank,
                "adapter_dropout": self.get_dropout(),
                "column_init_method": lora_params.get('column_init_method', 'xavier'),
                "row_init_method": lora_params.get('row_init_method', 'zero'),
                "layer_selection": self.get_layer_selection(),
                "weight_tying": lora_params.get('weight_tying', False),
                "module_filter": lora_params.get('module_filter', None)
            }
        }
        
        return nemo_config
    
    def get_recommended_batch_size(self, model_size: str, gpu_memory_gb: int = 24) -> int:
        """
        Get recommended batch size based on model size and LoRA configuration.
        
        Args:
            model_size: Size of the model (7b, 8b, 13b, etc.)
            gpu_memory_gb: Available GPU memory in GB
            
        Returns:
            Recommended batch size
        """
        if self.config is None:
            self.load_config()
        
        rank, _ = self.get_rank_and_alpha()
        
        # Base batch sizes for different model sizes (with LoRA)
        base_batch_sizes = {
            "7b": {"24gb": 4, "16gb": 2, "12gb": 1},
            "8b": {"24gb": 4, "16gb": 2, "12gb": 1},
            "13b": {"24gb": 2, "16gb": 1, "12gb": 1},
            "34b": {"24gb": 1, "16gb": 1, "12gb": 1},
            "70b": {"24gb": 1, "16gb": 1, "12gb": 1}
        }
        
        # Determine memory category
        if gpu_memory_gb >= 24:
            memory_cat = "24gb"
        elif gpu_memory_gb >= 16:
            memory_cat = "16gb"
        else:
            memory_cat = "12gb"
        
        base_batch = base_batch_sizes.get(model_size, {"24gb": 1, "16gb": 1, "12gb": 1})[memory_cat]
        
        # Adjust based on LoRA rank (higher rank = more memory)
        if rank <= 16:
            multiplier = 1.5
        elif rank <= 32:
            multiplier = 1.2
        elif rank <= 64:
            multiplier = 1.0
        else:
            multiplier = 0.8
        
        recommended_batch = max(1, int(base_batch * multiplier))
        
        logger.info(f"Recommended batch size for {model_size} with rank {rank}: {recommended_batch}")
        return recommended_batch
    
    def validate_config(self) -> bool:
        """
        Validate LoRA configuration.
        
        Returns:
            True if configuration is valid
        """
        if self.config is None:
            self.load_config()
        
        try:
            # Check required fields
            required_fields = ['adapter_dim', 'target_modules', 'adapter_dropout']
            
            for field in required_fields:
                if field not in self.config.lora:
                    logger.error(f"Missing required LoRA field: {field}")
                    return False
            
            # Validate rank and alpha
            rank, alpha = self.get_rank_and_alpha()
            if rank <= 0:
                logger.error(f"Invalid LoRA rank: {rank}")
                return False
            
            if alpha <= 0:
                logger.error(f"Invalid LoRA alpha: {alpha}")
                return False
            
            # Validate dropout
            dropout = self.get_dropout()
            if not (0.0 <= dropout <= 1.0):
                logger.error(f"Invalid LoRA dropout: {dropout}")
                return False
            
            # Validate target modules
            target_modules = self.get_target_modules()
            if not target_modules or not isinstance(target_modules, list):
                logger.error("Invalid target modules configuration")
                return False
            
            logger.info(f"LoRA configuration for {self.model_type} is valid")
            return True
            
        except Exception as e:
            logger.error(f"LoRA configuration validation failed: {e}")
            return False
