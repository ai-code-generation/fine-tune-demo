#!/usr/bin/env python3
"""
Installation verification script for CodeLlama fine-tuning pipeline.
This script verifies that all dependencies are properly installed in the container.
"""

import sys
import subprocess
import importlib
import platform
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"🔍 {text}")
    print('='*60)


def print_status(text, status=True):
    """Print status with colored output."""
    symbol = "✅" if status else "❌"
    print(f"{symbol} {text}")


def print_info(text):
    """Print informational text."""
    print(f"ℹ️  {text}")


def check_python_version():
    """Check Python version."""
    print_header("Python Environment")
    
    version = sys.version_info
    print_info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    print_info(f"Python executable: {sys.executable}")
    print_info(f"Platform: {platform.platform()}")
    
    if version >= (3, 8):
        print_status("Python version is compatible")
        return True
    else:
        print_status("Python version is too old (requires 3.8+)", False)
        return False


def check_cuda_environment():
    """Check CUDA environment (self-contained in container)."""
    print_header("CUDA Environment (Self-Contained)")

    print_info("This container includes its own CUDA toolkit!")
    print_info("No host CUDA installation required.")

    try:
        # Check CUDA toolkit (installed in container)
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print_info("NVCC output:")
            print(result.stdout.strip())
            print_status("CUDA toolkit is installed in container")
        else:
            print_status("CUDA toolkit not found in container", False)
    except FileNotFoundError:
        print_status("NVCC not found in container", False)

    try:
        # Check nvidia-smi (requires GPU access from host)
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print_info("GPU information:")
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'NVIDIA' in line or 'GPU' in line or 'MiB' in line:
                    print(f"  {line}")
            print_status("GPU access available")
        else:
            print_status("GPU access not available - will use CPU mode", False)
    except FileNotFoundError:
        print_status("nvidia-smi not available - will use CPU mode", False)


def check_package_installation():
    """Check if required packages are installed."""
    print_header("Package Installation")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Hugging Face Transformers'),
        ('peft', 'Parameter Efficient Fine-Tuning'),
        ('datasets', 'Hugging Face Datasets'),
        ('accelerate', 'Hugging Face Accelerate'),
        ('bitsandbytes', 'BitsAndBytes'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('yaml', 'PyYAML'),
        ('sklearn', 'Scikit-learn'),
        ('scipy', 'SciPy'),
        ('tqdm', 'TQDM'),
    ]
    
    all_installed = True
    
    for package, name in required_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print_status(f"{name}: {version}")
        except ImportError:
            print_status(f"{name}: Not installed", False)
            all_installed = False
    
    return all_installed


def check_pytorch_cuda():
    """Check PyTorch CUDA support."""
    print_header("PyTorch CUDA Support")
    
    try:
        import torch
        
        print_info(f"PyTorch version: {torch.__version__}")
        print_info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print_info(f"CUDA version: {torch.version.cuda}")
            print_info(f"cuDNN version: {torch.backends.cudnn.version()}")
            print_info(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print_info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Test CUDA operations
            try:
                x = torch.randn(100, 100).cuda()
                y = torch.randn(100, 100).cuda()
                _ = torch.mm(x, y)  # Simple matrix multiplication test
                print_status("CUDA operations test passed")
                return True
            except Exception as e:
                print_status(f"CUDA operations test failed: {e}", False)
                return False
        else:
            print_status("CUDA not available", False)
            return False
            
    except ImportError:
        print_status("PyTorch not installed", False)
        return False


def check_pipeline_modules():
    """Check pipeline-specific modules."""
    print_header("Pipeline Modules")
    
    # Add src to path
    sys.path.insert(0, '/workspace/src')
    
    pipeline_modules = [
        ('src.data_handler', 'Data Handler'),
        ('src.model_setup', 'Model Setup'),
        ('src.training', 'Training Module'),
        ('src.utils', 'Utilities'),
    ]
    
    all_working = True
    
    for module_name, display_name in pipeline_modules:
        try:
            importlib.import_module(module_name)
            print_status(f"{display_name}")
        except ImportError as e:
            print_status(f"{display_name}: {e}", False)
            all_working = False
    
    return all_working


def check_file_structure():
    """Check if required files and directories exist."""
    print_header("File Structure")
    
    required_paths = [
        ('/workspace/src', 'Source directory'),
        ('/workspace/configs', 'Configs directory'),
        ('/workspace/data', 'Data directory'),
        ('/workspace/output', 'Output directory'),
        ('/workspace/cache', 'Cache directory'),
        ('/workspace/train.py', 'Main training script'),
        ('/workspace/requirements.txt', 'Requirements file'),
        ('/workspace/configs/model_configs/codellama_7b.yaml', 'CodeLlama 7B config'),
        ('/workspace/configs/lora_configs/lora_default.yaml', 'LoRA config'),
    ]
    
    all_exist = True
    
    for path, description in required_paths:
        if Path(path).exists():
            print_status(f"{description}")
        else:
            print_status(f"{description}: Missing", False)
            all_exist = False
    
    return all_exist


def test_basic_functionality():
    """Test basic pipeline functionality."""
    print_header("Basic Functionality Test")
    
    try:
        # Test data handler
        from src.data_handler import ConversationDataHandler  # noqa: F401
        print_status("Data handler import")

        # Test model setup
        from src.model_setup import ModelSetup  # noqa: F401
        print_status("Model setup import")

        # Test training module
        from src.training import CodeLlamaTrainer  # noqa: F401
        print_status("Training module import")

        # Test utilities
        from src.utils import validate_model_output  # noqa: F401
        print_status("Utilities import")

        return True
        
    except Exception as e:
        print_status(f"Basic functionality test failed: {e}", False)
        return False


def check_huggingface_cache():
    """Check Hugging Face cache configuration."""
    print_header("Hugging Face Configuration")
    
    import os
    
    cache_vars = [
        'HF_HOME',
        'TRANSFORMERS_CACHE',
        'HF_DATASETS_CACHE',
        'TORCH_HOME'
    ]
    
    for var in cache_vars:
        value = os.environ.get(var, 'Not set')
        print_info(f"{var}: {value}")
        if value != 'Not set' and Path(value).exists():
            print_status(f"{var} directory exists")
        elif value != 'Not set':
            print_status(f"{var} directory missing", False)


def run_comprehensive_test():
    """Run a comprehensive test of the installation."""
    print_header("Comprehensive Installation Test")
    
    tests = [
        ("Python Version", check_python_version),
        ("Package Installation", check_package_installation),
        ("CUDA Environment", check_cuda_environment),
        ("PyTorch CUDA", check_pytorch_cuda),
        ("Pipeline Modules", check_pipeline_modules),
        ("File Structure", check_file_structure),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_status(f"{test_name} failed with exception: {e}", False)
            results.append((test_name, False))
    
    # Check Hugging Face configuration (informational only)
    check_huggingface_cache()
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        print_status(f"{test_name}", result)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print_status("🎉 All tests passed! Installation is complete and working.")
        return True
    else:
        print_status(f"❌ {total - passed} test(s) failed. Please check the installation.", False)
        return False


def main():
    """Main function."""
    print("CodeLlama Fine-tuning Pipeline - Installation Verification")
    print("=" * 60)
    
    success = run_comprehensive_test()
    
    if success:
        print("\n🚀 Ready to start fine-tuning!")
        print("\nNext steps:")
        print("1. Create sample data: python train.py --create-sample-data")
        print("2. Start training: python train.py --model-config configs/model_configs/codellama_7b.yaml")
        sys.exit(0)
    else:
        print("\n❌ Installation verification failed!")
        print("Please check the error messages above and fix any issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
