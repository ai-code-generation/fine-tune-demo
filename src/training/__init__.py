try:
    # Try relative imports first (when used as a package)
    from .lora_config import LoRAConfig
    from .trainer import NeMoTrainer
except ImportError:
    # Fall back to absolute imports (when run directly)
    from training.lora_config import LoRAConfig
    from training.trainer import NeMoTrainer

__all__ = ["LoRAConfig", "NeMoTrainer"]
