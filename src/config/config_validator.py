"""
Configuration validation system.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates configuration files and settings."""
    
    REQUIRED_SECTIONS = [
        "model", "data", "training", "lora", "optimizer", 
        "scheduler", "evaluation", "logging", "checkpointing", "hardware"
    ]
    
    REQUIRED_MODEL_FIELDS = [
        "type", "size", "base_model_path"
    ]
    
    REQUIRED_DATA_FIELDS = [
        "train_file", "max_seq_length", "batch_size"
    ]
    
    REQUIRED_TRAINING_FIELDS = [
        "max_epochs", "learning_rate", "precision"
    ]
    
    REQUIRED_LORA_FIELDS = [
        "enabled", "rank", "alpha", "dropout", "target_modules"
    ]
    
    VALID_MODEL_TYPES = ["llama2", "llama3", "codellama"]
    VALID_MODEL_SIZES = ["7b", "8b", "13b", "34b", "70b"]
    VALID_PRECISIONS = ["16", "32", "bf16", "fp16"]
    VALID_OPTIMIZERS = ["adam", "adamw", "sgd", "fused_adam"]
    VALID_SCHEDULERS = ["linear", "cosine", "polynomial", "constant"]
    
    def __init__(self):
        """Initialize the configuration validator."""
        self.errors = []
        self.warnings = []
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate a complete configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if configuration is valid
        """
        self.errors = []
        self.warnings = []
        
        # Check required sections
        self._validate_required_sections(config)
        
        # Validate individual sections
        if "model" in config:
            self._validate_model_config(config["model"])
        
        if "data" in config:
            self._validate_data_config(config["data"])
        
        if "training" in config:
            self._validate_training_config(config["training"])
        
        if "lora" in config:
            self._validate_lora_config(config["lora"])
        
        if "optimizer" in config:
            self._validate_optimizer_config(config["optimizer"])
        
        if "scheduler" in config:
            self._validate_scheduler_config(config["scheduler"])
        
        if "hardware" in config:
            self._validate_hardware_config(config["hardware"])
        
        # Cross-validation checks
        self._validate_cross_dependencies(config)
        
        # Log results
        if self.errors:
            logger.error(f"Configuration validation failed with {len(self.errors)} errors")
            for error in self.errors:
                logger.error(f"  - {error}")
        
        if self.warnings:
            logger.warning(f"Configuration has {len(self.warnings)} warnings")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        return len(self.errors) == 0
    
    def _validate_required_sections(self, config: Dict[str, Any]) -> None:
        """Validate that all required sections are present."""
        for section in self.REQUIRED_SECTIONS:
            if section not in config:
                self.errors.append(f"Missing required section: {section}")
    
    def _validate_model_config(self, model_config: Dict[str, Any]) -> None:
        """Validate model configuration."""
        # Check required fields
        for field in self.REQUIRED_MODEL_FIELDS:
            if field not in model_config:
                self.errors.append(f"Missing required model field: {field}")
        
        # Validate model type
        if "type" in model_config:
            if model_config["type"] not in self.VALID_MODEL_TYPES:
                self.errors.append(f"Invalid model type: {model_config['type']}. "
                                 f"Valid types: {self.VALID_MODEL_TYPES}")
        
        # Validate model size
        if "size" in model_config:
            if model_config["size"] not in self.VALID_MODEL_SIZES:
                self.errors.append(f"Invalid model size: {model_config['size']}. "
                                 f"Valid sizes: {self.VALID_MODEL_SIZES}")
        
        # Check base model path
        if "base_model_path" in model_config and model_config["base_model_path"]:
            base_path = model_config["base_model_path"]
            # If it's a local path, check if it exists
            if "/" in base_path and not base_path.startswith("http"):
                if not Path(base_path).exists():
                    self.warnings.append(f"Base model path does not exist: {base_path}")
    
    def _validate_data_config(self, data_config: Dict[str, Any]) -> None:
        """Validate data configuration."""
        # Check required fields
        for field in self.REQUIRED_DATA_FIELDS:
            if field not in data_config:
                self.errors.append(f"Missing required data field: {field}")
        
        # Validate file paths
        for file_field in ["train_file", "val_file", "test_file"]:
            if file_field in data_config and data_config[file_field]:
                file_path = Path(data_config[file_field])
                if not file_path.exists():
                    self.warnings.append(f"Data file does not exist: {file_path}")
        
        # Validate numeric values
        if "batch_size" in data_config:
            if not isinstance(data_config["batch_size"], int) or data_config["batch_size"] <= 0:
                self.errors.append("Batch size must be a positive integer")
        
        if "max_seq_length" in data_config:
            if not isinstance(data_config["max_seq_length"], int) or data_config["max_seq_length"] <= 0:
                self.errors.append("Max sequence length must be a positive integer")
            elif data_config["max_seq_length"] > 32768:
                self.warnings.append("Very large sequence length may cause memory issues")
        
        if "num_workers" in data_config:
            if not isinstance(data_config["num_workers"], int) or data_config["num_workers"] < 0:
                self.errors.append("Number of workers must be a non-negative integer")
    
    def _validate_training_config(self, training_config: Dict[str, Any]) -> None:
        """Validate training configuration."""
        # Check required fields
        for field in self.REQUIRED_TRAINING_FIELDS:
            if field not in training_config:
                self.errors.append(f"Missing required training field: {field}")
        
        # Validate numeric values
        if "max_epochs" in training_config:
            if not isinstance(training_config["max_epochs"], int) or training_config["max_epochs"] <= 0:
                self.errors.append("Max epochs must be a positive integer")
        
        if "learning_rate" in training_config:
            lr = training_config["learning_rate"]
            if not isinstance(lr, (int, float)) or lr <= 0:
                self.errors.append("Learning rate must be a positive number")
            elif lr > 1.0:
                self.warnings.append("Learning rate is very high, may cause instability")
        
        if "weight_decay" in training_config:
            wd = training_config["weight_decay"]
            if not isinstance(wd, (int, float)) or wd < 0:
                self.errors.append("Weight decay must be a non-negative number")
        
        if "gradient_clip_val" in training_config:
            gc = training_config["gradient_clip_val"]
            if not isinstance(gc, (int, float)) or gc <= 0:
                self.errors.append("Gradient clip value must be a positive number")
        
        # Validate precision
        if "precision" in training_config:
            if training_config["precision"] not in self.VALID_PRECISIONS:
                self.errors.append(f"Invalid precision: {training_config['precision']}. "
                                 f"Valid precisions: {self.VALID_PRECISIONS}")
    
    def _validate_lora_config(self, lora_config: Dict[str, Any]) -> None:
        """Validate LoRA configuration."""
        # Check required fields
        for field in self.REQUIRED_LORA_FIELDS:
            if field not in lora_config:
                self.errors.append(f"Missing required LoRA field: {field}")
        
        # Validate LoRA parameters
        if "rank" in lora_config:
            rank = lora_config["rank"]
            if not isinstance(rank, int) or rank <= 0:
                self.errors.append("LoRA rank must be a positive integer")
            elif rank > 256:
                self.warnings.append("Very high LoRA rank may not be efficient")
        
        if "alpha" in lora_config:
            alpha = lora_config["alpha"]
            if not isinstance(alpha, int) or alpha <= 0:
                self.errors.append("LoRA alpha must be a positive integer")
        
        if "dropout" in lora_config:
            dropout = lora_config["dropout"]
            if not isinstance(dropout, (int, float)) or dropout < 0 or dropout >= 1:
                self.errors.append("LoRA dropout must be between 0 and 1")
        
        # Validate target modules
        if "target_modules" in lora_config:
            target_modules = lora_config["target_modules"]
            if not isinstance(target_modules, list) or len(target_modules) == 0:
                self.errors.append("Target modules must be a non-empty list")
    
    def _validate_optimizer_config(self, optimizer_config: Dict[str, Any]) -> None:
        """Validate optimizer configuration."""
        if "name" in optimizer_config:
            if optimizer_config["name"] not in self.VALID_OPTIMIZERS:
                self.errors.append(f"Invalid optimizer: {optimizer_config['name']}. "
                                 f"Valid optimizers: {self.VALID_OPTIMIZERS}")
        
        if "lr" in optimizer_config:
            lr = optimizer_config["lr"]
            if not isinstance(lr, (int, float)) or lr <= 0:
                self.errors.append("Optimizer learning rate must be a positive number")
        
        if "betas" in optimizer_config:
            betas = optimizer_config["betas"]
            if not isinstance(betas, list) or len(betas) != 2:
                self.errors.append("Optimizer betas must be a list of two values")
            else:
                for i, beta in enumerate(betas):
                    if not isinstance(beta, (int, float)) or beta < 0 or beta >= 1:
                        self.errors.append(f"Optimizer beta{i+1} must be between 0 and 1")
    
    def _validate_scheduler_config(self, scheduler_config: Dict[str, Any]) -> None:
        """Validate scheduler configuration."""
        if "name" in scheduler_config:
            if scheduler_config["name"] not in self.VALID_SCHEDULERS:
                self.errors.append(f"Invalid scheduler: {scheduler_config['name']}. "
                                 f"Valid schedulers: {self.VALID_SCHEDULERS}")
        
        if "warmup_steps" in scheduler_config:
            warmup = scheduler_config["warmup_steps"]
            if not isinstance(warmup, int) or warmup < 0:
                self.errors.append("Warmup steps must be a non-negative integer")
    
    def _validate_hardware_config(self, hardware_config: Dict[str, Any]) -> None:
        """Validate hardware configuration."""
        if "gpus" in hardware_config:
            gpus = hardware_config["gpus"]
            if not isinstance(gpus, int) or gpus < 0:
                self.errors.append("Number of GPUs must be a non-negative integer")
            elif gpus > 8:
                self.warnings.append("Very high number of GPUs specified")
        
        if "nodes" in hardware_config:
            nodes = hardware_config["nodes"]
            if not isinstance(nodes, int) or nodes <= 0:
                self.errors.append("Number of nodes must be a positive integer")
    
    def _validate_cross_dependencies(self, config: Dict[str, Any]) -> None:
        """Validate cross-dependencies between configuration sections."""
        # Check if LoRA rank and alpha are compatible
        if "lora" in config:
            lora_config = config["lora"]
            if "rank" in lora_config and "alpha" in lora_config:
                rank = lora_config["rank"]
                alpha = lora_config["alpha"]
                if alpha < rank:
                    self.warnings.append("LoRA alpha is less than rank, may reduce effectiveness")
        
        # Check if batch size is compatible with sequence length and GPU memory
        if "data" in config and "hardware" in config:
            data_config = config["data"]
            hardware_config = config["hardware"]
            
            if "batch_size" in data_config and "max_seq_length" in data_config and "gpus" in hardware_config:
                batch_size = data_config["batch_size"]
                seq_length = data_config["max_seq_length"]
                gpus = hardware_config["gpus"]
                
                # Rough memory estimation (very simplified)
                memory_per_sample = seq_length * 4 * 2  # Rough bytes per token
                total_memory = batch_size * memory_per_sample * 1e-9  # GB
                
                if total_memory > 20 and gpus == 1:
                    self.warnings.append("High memory usage expected, consider reducing batch size or sequence length")
        
        # Check if learning rate is compatible with optimizer
        if "training" in config and "optimizer" in config:
            training_config = config["training"]
            optimizer_config = config["optimizer"]
            
            if "learning_rate" in training_config and "lr" in optimizer_config:
                if training_config["learning_rate"] != optimizer_config["lr"]:
                    self.warnings.append("Learning rate mismatch between training and optimizer configs")
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get a detailed validation report."""
        return {
            "valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings)
        }
    
    def validate_file_paths(self, config: Dict[str, Any], base_path: Optional[Path] = None) -> bool:
        """
        Validate that all file paths in config exist.
        
        Args:
            config: Configuration dictionary
            base_path: Base path to resolve relative paths
            
        Returns:
            True if all paths are valid
        """
        if base_path is None:
            base_path = Path.cwd()
        
        path_errors = []
        
        # Check data files
        if "data" in config:
            for file_field in ["train_file", "val_file", "test_file"]:
                if file_field in config["data"] and config["data"][file_field]:
                    file_path = base_path / config["data"][file_field]
                    if not file_path.exists():
                        path_errors.append(f"Data file not found: {file_path}")
        
        # Check model path
        if "model" in config and "base_model_path" in config["model"]:
            model_path = config["model"]["base_model_path"]
            if model_path and "/" in model_path and not model_path.startswith("http"):
                full_path = base_path / model_path
                if not full_path.exists():
                    path_errors.append(f"Model path not found: {full_path}")
        
        if path_errors:
            logger.error("File path validation failed:")
            for error in path_errors:
                logger.error(f"  - {error}")
            return False
        
        return True
