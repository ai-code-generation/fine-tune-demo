"""
Configuration management system for the fine-tuning pipeline.
"""

import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration files and settings for the pipeline."""
    
    CONFIG_TEMPLATES = {
        "llama2": "configs/training/llama2_training.yaml",
        "llama3": "configs/training/llama3_training.yaml",
        "codellama": "configs/training/codellama_training.yaml"
    }
    
    def __init__(self, model_type: str = "llama2"):
        """
        Initialize configuration manager.
        
        Args:
            model_type: Type of model (llama2, llama3, codellama)
        """
        self.model_type = model_type.lower()
        
        if self.model_type not in self.CONFIG_TEMPLATES:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported: {list(self.CONFIG_TEMPLATES.keys())}")
        
        self.config = None
        self.config_path = None
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> DictConfig:
        """
        Load configuration from file.
        
        Args:
            config_path: Optional path to config file (uses default if None)
            
        Returns:
            OmegaConf configuration object
        """
        if config_path is None:
            config_path = self.CONFIG_TEMPLATES[self.model_type]
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            self.config = OmegaConf.load(config_path)
            self.config_path = config_path
            logger.info(f"Loaded configuration from {config_path}")
            return self.config
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {config_path}: {e}")
    
    def create_config_from_template(self, 
                                   output_path: Union[str, Path],
                                   overrides: Optional[Dict[str, Any]] = None) -> DictConfig:
        """
        Create a new config file from template with optional overrides.
        
        Args:
            output_path: Path to save the new config
            overrides: Optional dictionary of config overrides
            
        Returns:
            OmegaConf configuration object
        """
        # Load template
        template_path = self.CONFIG_TEMPLATES[self.model_type]
        config = OmegaConf.load(template_path)
        
        # Apply overrides if provided
        if overrides:
            override_config = OmegaConf.create(overrides)
            config = OmegaConf.merge(config, override_config)
        
        # Save new config
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            OmegaConf.save(config, f)
        
        logger.info(f"Created config file: {output_path}")
        return config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update current configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        if self.config is None:
            raise RuntimeError("No configuration loaded. Call load_config() first.")
        
        update_config = OmegaConf.create(updates)
        self.config = OmegaConf.merge(self.config, update_config)
        
        logger.info("Updated configuration with new values")
    
    def save_config(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Optional path to save (uses current path if None)
        """
        if self.config is None:
            raise RuntimeError("No configuration to save")
        
        if output_path is None:
            output_path = self.config_path
        
        if output_path is None:
            raise ValueError("No output path specified and no current path available")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            OmegaConf.save(self.config, f)
        
        logger.info(f"Saved configuration to {output_path}")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration section."""
        if self.config is None:
            raise RuntimeError("No configuration loaded")
        
        return OmegaConf.to_container(self.config.model, resolve=True)
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration section."""
        if self.config is None:
            raise RuntimeError("No configuration loaded")
        
        return OmegaConf.to_container(self.config.data, resolve=True)
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration section."""
        if self.config is None:
            raise RuntimeError("No configuration loaded")
        
        return OmegaConf.to_container(self.config.training, resolve=True)
    
    def get_lora_config(self) -> Dict[str, Any]:
        """Get LoRA configuration section."""
        if self.config is None:
            raise RuntimeError("No configuration loaded")
        
        return OmegaConf.to_container(self.config.lora, resolve=True)
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware configuration section."""
        if self.config is None:
            raise RuntimeError("No configuration loaded")
        
        return OmegaConf.to_container(self.config.hardware, resolve=True)
    
    def get_deployment_config(self) -> Dict[str, Any]:
        """Get deployment configuration section."""
        if self.config is None:
            raise RuntimeError("No configuration loaded")
        
        return OmegaConf.to_container(self.config.deployment, resolve=True)
    
    def set_data_paths(self, 
                      train_file: str,
                      val_file: Optional[str] = None,
                      test_file: Optional[str] = None) -> None:
        """
        Set data file paths in configuration.
        
        Args:
            train_file: Path to training data
            val_file: Path to validation data
            test_file: Path to test data
        """
        if self.config is None:
            raise RuntimeError("No configuration loaded")
        
        self.config.data.train_file = train_file
        if val_file:
            self.config.data.val_file = val_file
        if test_file:
            self.config.data.test_file = test_file
        
        logger.info("Updated data paths in configuration")
    
    def set_model_path(self, base_model_path: str) -> None:
        """
        Set base model path in configuration.
        
        Args:
            base_model_path: Path to base model or HuggingFace model name
        """
        if self.config is None:
            raise RuntimeError("No configuration loaded")
        
        self.config.model.base_model_path = base_model_path
        logger.info(f"Set base model path to: {base_model_path}")
    
    def set_training_params(self, 
                           max_epochs: Optional[int] = None,
                           learning_rate: Optional[float] = None,
                           batch_size: Optional[int] = None,
                           gpus: Optional[int] = None) -> None:
        """
        Set training parameters in configuration.
        
        Args:
            max_epochs: Maximum number of epochs
            learning_rate: Learning rate
            batch_size: Batch size
            gpus: Number of GPUs
        """
        if self.config is None:
            raise RuntimeError("No configuration loaded")
        
        if max_epochs is not None:
            self.config.training.max_epochs = max_epochs
        if learning_rate is not None:
            self.config.training.learning_rate = learning_rate
            self.config.optimizer.lr = learning_rate
        if batch_size is not None:
            self.config.data.batch_size = batch_size
        if gpus is not None:
            self.config.hardware.gpus = gpus
        
        logger.info("Updated training parameters in configuration")
    
    def set_lora_params(self, 
                       rank: Optional[int] = None,
                       alpha: Optional[int] = None,
                       dropout: Optional[float] = None) -> None:
        """
        Set LoRA parameters in configuration.
        
        Args:
            rank: LoRA rank
            alpha: LoRA alpha
            dropout: LoRA dropout
        """
        if self.config is None:
            raise RuntimeError("No configuration loaded")
        
        if rank is not None:
            self.config.lora.rank = rank
        if alpha is not None:
            self.config.lora.alpha = alpha
        if dropout is not None:
            self.config.lora.dropout = dropout
        
        logger.info("Updated LoRA parameters in configuration")
    
    def optimize_for_hardware(self, 
                             gpu_memory_gb: int,
                             num_gpus: int = 1) -> None:
        """
        Optimize configuration for available hardware.
        
        Args:
            gpu_memory_gb: Available GPU memory in GB
            num_gpus: Number of available GPUs
        """
        if self.config is None:
            raise RuntimeError("No configuration loaded")
        
        # Adjust batch size based on GPU memory
        if gpu_memory_gb <= 12:
            batch_size = 1
            accumulate_grad_batches = 16
        elif gpu_memory_gb <= 16:
            batch_size = 2
            accumulate_grad_batches = 8
        elif gpu_memory_gb <= 24:
            batch_size = 4
            accumulate_grad_batches = 4
        else:
            batch_size = 8
            accumulate_grad_batches = 2
        
        # Adjust for model type
        if self.model_type == "codellama":
            batch_size = max(1, batch_size // 2)  # Smaller batches for long sequences
            accumulate_grad_batches *= 2
        
        self.config.data.batch_size = batch_size
        self.config.training.accumulate_grad_batches = accumulate_grad_batches
        self.config.hardware.gpus = num_gpus
        
        # Adjust strategy for multi-GPU
        if num_gpus > 1:
            self.config.training.strategy = "ddp"
        else:
            self.config.training.strategy = "auto"
        
        logger.info(f"Optimized config for {num_gpus} GPUs with {gpu_memory_gb}GB memory")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        if self.config is None:
            raise RuntimeError("No configuration loaded")
        
        summary = {
            "model_type": self.config.model.type,
            "model_size": self.config.model.size,
            "base_model": self.config.model.base_model_path,
            "max_epochs": self.config.training.max_epochs,
            "learning_rate": self.config.training.learning_rate,
            "batch_size": self.config.data.batch_size,
            "max_seq_length": self.config.data.max_seq_length,
            "lora_rank": self.config.lora.rank,
            "lora_alpha": self.config.lora.alpha,
            "gpus": self.config.hardware.gpus,
            "precision": self.config.training.precision
        }
        
        return summary
    
    def list_available_configs(self) -> List[str]:
        """List all available configuration templates."""
        return list(self.CONFIG_TEMPLATES.keys())
    
    def compare_configs(self, other_config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Compare current config with another config file.
        
        Args:
            other_config_path: Path to other config file
            
        Returns:
            Dictionary with comparison results
        """
        if self.config is None:
            raise RuntimeError("No configuration loaded")
        
        other_config = OmegaConf.load(other_config_path)
        
        current_dict = OmegaConf.to_container(self.config, resolve=True)
        other_dict = OmegaConf.to_container(other_config, resolve=True)
        
        differences = {}
        
        def compare_dicts(d1, d2, path=""):
            for key in set(d1.keys()) | set(d2.keys()):
                current_path = f"{path}.{key}" if path else key
                
                if key not in d1:
                    differences[current_path] = {"current": None, "other": d2[key]}
                elif key not in d2:
                    differences[current_path] = {"current": d1[key], "other": None}
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    compare_dicts(d1[key], d2[key], current_path)
                elif d1[key] != d2[key]:
                    differences[current_path] = {"current": d1[key], "other": d2[key]}
        
        compare_dicts(current_dict, other_dict)
        
        return {
            "current_config": str(self.config_path),
            "other_config": str(other_config_path),
            "differences": differences
        }
