#!/usr/bin/env python3
"""
Setup validation script for the NeMo Fine-Tuning Pipeline.

This script validates that:
1. All required dependencies are installed
2. Configuration files are valid
3. Sample data is properly formatted
4. Directory structure is correct
5. GPU availability (if applicable)
"""

import sys
import subprocess
import importlib
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_python_version():
    """Check Python version compatibility."""
    logger.info("Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    
    logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'torch',
        'transformers',
        'pytorch_lightning',
        'omegaconf',
        'pyyaml',
        'numpy',
        'pandas'
    ]
    
    optional_packages = [
        'nemo_toolkit',
        'wandb',
        'tensorboard',
        'rouge_score',
        'evaluate'
    ]
    
    missing_required = []
    missing_optional = []
    
    # Check required packages
    for package in required_packages:
        try:
            importlib.import_module(package)
            logger.info(f"✓ {package}")
        except ImportError:
            missing_required.append(package)
            logger.error(f"✗ {package} (required)")
    
    # Check optional packages
    for package in optional_packages:
        try:
            importlib.import_module(package)
            logger.info(f"✓ {package} (optional)")
        except ImportError:
            missing_optional.append(package)
            logger.warning(f"? {package} (optional)")
    
    if missing_required:
        logger.error(f"Missing required packages: {missing_required}")
        logger.error("Install with: pip install -r requirements.txt")
        return False
    
    if missing_optional:
        logger.warning(f"Missing optional packages: {missing_optional}")
        logger.warning("Some features may not be available")
    
    return True


def check_gpu_availability():
    """Check GPU availability and CUDA setup."""
    logger.info("Checking GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"✓ CUDA available with {gpu_count} GPU(s)")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            return True
        else:
            logger.warning("✗ CUDA not available - will use CPU training")
            return False
            
    except ImportError:
        logger.error("✗ PyTorch not available")
        return False


def check_directory_structure():
    """Check if directory structure is correct."""
    logger.info("Checking directory structure...")
    
    required_dirs = [
        "src",
        "configs",
        "configs/models",
        "configs/training", 
        "configs/lora",
        "data",
        "scripts",
        "tests",
        "logs",
        "checkpoints",
        "deploy"
    ]
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
            logger.error(f"✗ Missing directory: {dir_path}")
        else:
            logger.info(f"✓ {dir_path}")
    
    if missing_dirs:
        logger.error("Creating missing directories...")
        for dir_path in missing_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created: {dir_path}")
    
    return True


def check_configuration_files():
    """Check if configuration files are valid."""
    logger.info("Checking configuration files...")
    
    try:
        from config.config_manager import ConfigManager
        from config.config_validator import ConfigValidator
        
        validator = ConfigValidator()
        
        # Check each model type configuration
        model_types = ["llama2", "llama3", "codellama"]
        
        for model_type in model_types:
            try:
                config_manager = ConfigManager(model_type)
                config = config_manager.load_config()
                
                # Convert to dict for validation
                from omegaconf import OmegaConf
                config_dict = OmegaConf.to_container(config, resolve=True)
                
                is_valid = validator.validate_config(config_dict)
                
                if is_valid:
                    logger.info(f"✓ {model_type} configuration valid")
                else:
                    report = validator.get_validation_report()
                    logger.error(f"✗ {model_type} configuration invalid:")
                    for error in report["errors"]:
                        logger.error(f"  - {error}")
                    return False
                        
            except Exception as e:
                logger.error(f"✗ Failed to load {model_type} config: {e}")
                return False
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Cannot import configuration modules: {e}")
        return False


def check_sample_data():
    """Check if sample data is properly formatted."""
    logger.info("Checking sample data...")
    
    try:
        from data.data_processor import DataProcessor
        
        sample_data_file = "data/sample_data.yaml"
        
        if not Path(sample_data_file).exists():
            logger.warning(f"? Sample data file not found: {sample_data_file}")
            return True  # Not critical for setup validation
        
        processor = DataProcessor("llama2")
        conversations = processor.load_yaml_data(sample_data_file)
        
        if not conversations:
            logger.error("✗ No conversations found in sample data")
            return False
        
        # Validate conversations
        valid_count = 0
        for conv in conversations:
            if processor.validate_conversation(conv):
                valid_count += 1
        
        if valid_count == 0:
            logger.error("✗ No valid conversations in sample data")
            return False
        
        logger.info(f"✓ Sample data valid ({valid_count}/{len(conversations)} conversations)")
        
        # Get statistics
        stats = processor.get_statistics(conversations)
        logger.info(f"  Total messages: {stats['total_messages']}")
        logger.info(f"  Role distribution: {stats['role_distribution']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to validate sample data: {e}")
        return False


def check_model_configs():
    """Check if model configuration files exist and are valid."""
    logger.info("Checking model configuration files...")
    
    model_configs = [
        "configs/models/llama2_7b.yaml",
        "configs/models/llama3_8b.yaml", 
        "configs/models/codellama_7b.yaml"
    ]
    
    for config_file in model_configs:
        if not Path(config_file).exists():
            logger.error(f"✗ Missing model config: {config_file}")
            return False
        else:
            logger.info(f"✓ {config_file}")
    
    return True


def check_scripts():
    """Check if main scripts are executable."""
    logger.info("Checking scripts...")
    
    scripts = [
        "scripts/run_pipeline.py",
        "examples/basic_training.py",
        "examples/advanced_training.py",
        "examples/codellama_example.py"
    ]
    
    for script in scripts:
        script_path = Path(script)
        if not script_path.exists():
            logger.error(f"✗ Missing script: {script}")
            return False
        
        # Check if script is executable (on Unix systems)
        if hasattr(script_path, 'stat'):
            try:
                import stat
                mode = script_path.stat().st_mode
                if not (mode & stat.S_IXUSR):
                    logger.warning(f"? Script not executable: {script}")
            except:
                pass  # Skip on Windows or if stat fails
        
        logger.info(f"✓ {script}")
    
    return True


def run_basic_import_test():
    """Test basic imports of pipeline modules."""
    logger.info("Testing basic imports...")
    
    modules_to_test = [
        "data.data_processor",
        "data.instruction_formatter", 
        "data.dataset_builder",
        "models.model_config",
        "models.model_factory",
        "training.lora_config",
        "evaluation.metrics",
        "deployment.deployer",
        "config.config_manager",
        "config.config_validator"
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            logger.info(f"✓ {module}")
        except ImportError as e:
            failed_imports.append((module, str(e)))
            logger.error(f"✗ {module}: {e}")
    
    if failed_imports:
        logger.error("Some modules failed to import")
        return False
    
    return True


def main():
    """Run all validation checks."""
    logger.info("=" * 60)
    logger.info("NeMo Fine-Tuning Pipeline Setup Validation")
    logger.info("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("GPU Availability", check_gpu_availability),
        ("Directory Structure", check_directory_structure),
        ("Configuration Files", check_configuration_files),
        ("Model Configs", check_model_configs),
        ("Sample Data", check_sample_data),
        ("Scripts", check_scripts),
        ("Basic Imports", run_basic_import_test)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        logger.info(f"\n{'-' * 40}")
        logger.info(f"Running: {check_name}")
        logger.info(f"{'-' * 40}")
        
        try:
            result = check_func()
            results[check_name] = result
        except Exception as e:
            logger.error(f"Check failed with exception: {e}")
            results[check_name] = False
    
    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'=' * 60}")
    
    passed = 0
    failed = 0
    
    for check_name, result in results.items():
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        logger.info(f"{symbol} {check_name}: {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("\n🎉 All checks passed! The pipeline is ready to use.")
        logger.info("\nNext steps:")
        logger.info("1. Prepare your training data in YAML format")
        logger.info("2. Run: python scripts/run_pipeline.py --help")
        logger.info("3. Try the examples in the examples/ directory")
        return True
    else:
        logger.error(f"\n❌ {failed} checks failed. Please fix the issues above.")
        logger.error("\nFor help, check:")
        logger.error("- README.md for setup instructions")
        logger.error("- docs/USAGE.md for detailed usage guide")
        logger.error("- requirements.txt for dependency list")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
