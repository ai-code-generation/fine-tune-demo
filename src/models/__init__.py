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
    from .model_config import ModelConfig
    from .model_factory import ModelFactory
except ImportError:
    try:
        # Fall back to absolute imports from src directory
        from models.model_config import ModelConfig
        from models.model_factory import ModelFactory
    except ImportError:
        # Final fallback - import directly from files
        import importlib.util

        def import_from_path(module_path, module_name):
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        models_dir = current_dir

        model_config_module = import_from_path(models_dir / "model_config.py", "model_config")
        ModelConfig = model_config_module.ModelConfig

        model_factory_module = import_from_path(models_dir / "model_factory.py", "model_factory")
        ModelFactory = model_factory_module.ModelFactory

__all__ = ["ModelConfig", "ModelFactory"]
