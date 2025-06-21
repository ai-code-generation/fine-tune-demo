#!/usr/bin/env python3
"""
Test runner script for the CodeLlama fine-tuning pipeline.
Runs various tests to validate the pipeline functionality.
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✓ PASSED")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("✗ FAILED")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*60)
    print("Testing module imports...")
    print("="*60)
    
    modules_to_test = [
        'src.data_handler',
        'src.model_setup',
        'src.training',
        'src.utils'
    ]
    
    all_passed = True
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            all_passed = False
        except Exception as e:
            print(f"✗ {module}: Unexpected error: {e}")
            all_passed = False
    
    return all_passed


def test_dependencies():
    """Test that required dependencies are available."""
    print("\n" + "="*60)
    print("Testing dependencies...")
    print("="*60)
    
    dependencies = [
        'torch',
        'transformers',
        'peft',
        'datasets',
        'yaml',
        'numpy',
        'pandas'
    ]
    
    all_passed = True
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep}")
            all_passed = False
    
    return all_passed


def test_data_handler():
    """Test data handler functionality."""
    try:
        from tests.test_data_handler import run_manual_tests
        print("\n" + "="*60)
        print("Testing data handler...")
        print("="*60)
        run_manual_tests()
        return True
    except Exception as e:
        print(f"Data handler test failed: {e}")
        return False


def test_model_setup():
    """Test model setup functionality."""
    try:
        from tests.test_model_setup import run_manual_tests
        print("\n" + "="*60)
        print("Testing model setup...")
        print("="*60)
        run_manual_tests()
        return True
    except Exception as e:
        print(f"Model setup test failed: {e}")
        return False


def test_pipeline_integration():
    """Test pipeline integration."""
    try:
        from tests.test_pipeline import run_manual_integration_test
        print("\n" + "="*60)
        print("Testing pipeline integration...")
        print("="*60)
        run_manual_integration_test()
        return True
    except Exception as e:
        print(f"Pipeline integration test failed: {e}")
        return False


def test_sample_data_creation():
    """Test sample data creation."""
    return run_command(
        "python train.py --create-sample-data",
        "Sample data creation"
    )


def test_yaml_validation():
    """Test YAML validation."""
    # First create sample data if it doesn't exist
    if not os.path.exists("data/sample_train.yaml"):
        run_command("python train.py --create-sample-data", "Creating sample data for validation test")
    
    return run_command(
        "python train.py --train-data data/sample_train.yaml --validate-data --help",
        "YAML validation"
    )


def test_config_loading():
    """Test configuration loading."""
    return run_command(
        "python -c \"from src.model_setup import ModelSetup; ModelSetup('configs/model_configs/codellama_7b.yaml', 'configs/lora_configs/lora_default.yaml'); print('Config loading successful')\"",
        "Configuration loading"
    )


def run_pytest_tests():
    """Run pytest tests if pytest is available."""
    try:
        import pytest
        print("\n" + "="*60)
        print("Running pytest tests...")
        print("="*60)
        
        # Run pytest on the tests directory
        result = pytest.main(["-v", "tests/", "--tb=short"])
        return result == 0
    except ImportError:
        print("pytest not available, skipping pytest tests")
        return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run tests for CodeLlama fine-tuning pipeline")
    parser.add_argument("--quick", action="store_true", help="Run only quick tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--pytest", action="store_true", help="Run pytest tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if not any([args.quick, args.integration, args.pytest, args.all]):
        args.quick = True  # Default to quick tests
    
    print("CodeLlama Fine-tuning Pipeline Test Runner")
    print("=" * 60)
    
    test_results = []
    
    # Quick tests
    if args.quick or args.all:
        print("\n🚀 Running quick tests...")
        
        test_results.append(("Import tests", test_imports()))
        test_results.append(("Dependency tests", test_dependencies()))
        test_results.append(("Sample data creation", test_sample_data_creation()))
        test_results.append(("Config loading", test_config_loading()))
        
        # Manual unit tests
        test_results.append(("Data handler tests", test_data_handler()))
        test_results.append(("Model setup tests", test_model_setup()))
    
    # Integration tests
    if args.integration or args.all:
        print("\n🔧 Running integration tests...")
        test_results.append(("Pipeline integration", test_pipeline_integration()))
        test_results.append(("YAML validation", test_yaml_validation()))
    
    # Pytest tests
    if args.pytest or args.all:
        print("\n🧪 Running pytest tests...")
        test_results.append(("Pytest tests", run_pytest_tests()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {len(test_results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\n❌ {failed} test(s) failed!")
        sys.exit(1)
    else:
        print(f"\n✅ All {passed} tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
