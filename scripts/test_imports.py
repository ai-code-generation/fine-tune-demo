#!/usr/bin/env python3
"""
Test script to verify all imports work correctly in the container.
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test all pipeline imports."""
    print("Testing pipeline imports...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    
    # Add src to path
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"
    sys.path.insert(0, str(src_path))
    
    print(f"Source path: {src_path}")
    print(f"Source path exists: {src_path.exists()}")
    
    if src_path.exists():
        print(f"Contents of src: {list(src_path.iterdir())}")
    
    # Test core dependencies
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA device count: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import pytorch_lightning as pl
        print(f"✅ PyTorch Lightning {pl.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch Lightning import failed: {e}")
        return False
    
    try:
        import nemo
        print(f"✅ NeMo imported successfully")
    except ImportError as e:
        print(f"❌ NeMo import failed: {e}")
        return False
    
    # Test pipeline modules
    modules_to_test = [
        ("training.trainer", "NeMoTrainer"),
        ("training.lora_config", "LoRAConfig"),
        ("models.model_config", "ModelConfig"),
        ("models.model_factory", "ModelFactory"),
        ("evaluation.evaluator", "ModelEvaluator"),
        ("deployment.deployer", "ModelDeployer"),
        ("deployment.converter", "ModelConverter"),
    ]

    print("\n🔍 Testing pipeline module imports...")

    # Test individual modules first
    individual_modules = [
        "training", "models", "data", "evaluation", "deployment"
    ]

    for module_name in individual_modules:
        try:
            module = __import__(module_name)
            print(f"✅ {module_name} module imported successfully")
        except ImportError as e:
            print(f"❌ {module_name} module import failed: {e}")
            return False
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name} imported successfully")
        except ImportError as e:
            print(f"❌ {module_name}.{class_name} import failed: {e}")
            return False
        except AttributeError as e:
            print(f"❌ {module_name}.{class_name} attribute error: {e}")
            return False
    
    print("\n🎉 All imports successful!")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
