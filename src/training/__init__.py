# Import handling for both package and direct execution
import sys
from pathlib import Path

# Add the src directory to Python path if not already there
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    # Try relative imports first (when used as a package)
    from .lora_config import LoRAConfig
    from .trainer import NeMoTrainer
except ImportError:
    try:
        # Fall back to absolute imports from src directory
        from training.lora_config import LoRAConfig
        from training.trainer import NeMoTrainer
    except ImportError:
        # Final fallback - import directly from files
        import importlib.util

        def import_from_path(module_path, module_name):
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        training_dir = current_dir

        lora_config_module = import_from_path(training_dir / "lora_config.py", "lora_config")
        LoRAConfig = lora_config_module.LoRAConfig

        trainer_module = import_from_path(training_dir / "trainer.py", "trainer")
        NeMoTrainer = trainer_module.NeMoTrainer

__all__ = ["LoRAConfig", "NeMoTrainer"]
