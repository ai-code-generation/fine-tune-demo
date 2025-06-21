#!/usr/bin/env python3
"""
Setup script for CodeLlama fine-tuning pipeline.
Handles initial setup, dependency checking, and environment preparation.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True


def check_gpu_availability():
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("⚠️  No GPU detected. Training will use CPU (very slow)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed. Cannot check GPU availability")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        "data",
        "output",
        "logs",
        "cache",
        "configs/model_configs",
        "configs/lora_configs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}")
    
    return True


def create_sample_data():
    """Create sample training data."""
    print("📝 Creating sample training data...")
    
    try:
        subprocess.check_call([sys.executable, "train.py", "--create-sample-data"])
        print("✅ Sample data created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create sample data: {e}")
        return False


def run_tests():
    """Run basic tests to verify setup."""
    print("🧪 Running basic tests...")
    
    try:
        subprocess.check_call([sys.executable, "scripts/run_tests.py", "--quick"])
        print("✅ Basic tests passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e}")
        return False


def check_docker():
    """Check if Docker is available."""
    try:
        subprocess.check_output(["docker", "--version"], stderr=subprocess.STDOUT)
        print("✅ Docker is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Docker not found. Docker features will not be available")
        return False


def setup_git_hooks():
    """Setup git hooks for development."""
    if not Path(".git").exists():
        print("⚠️  Not a git repository. Skipping git hooks setup")
        return True
    
    print("🔧 Setting up git hooks...")
    
    # Create pre-commit hook
    pre_commit_hook = """#!/bin/bash
# Run tests before commit
python scripts/run_tests.py --quick
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
"""
    
    hooks_dir = Path(".git/hooks")
    hooks_dir.mkdir(exist_ok=True)
    
    pre_commit_path = hooks_dir / "pre-commit"
    with open(pre_commit_path, "w") as f:
        f.write(pre_commit_hook)
    
    # Make executable
    os.chmod(pre_commit_path, 0o755)
    
    print("✅ Git hooks setup complete")
    return True


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 Setup completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review and modify training data in data/ directory")
    print("2. Adjust model configurations in configs/ directory")
    print("3. Start training:")
    print("   python train.py --model-config configs/model_configs/codellama_7b.yaml \\")
    print("                   --train-data data/train.yaml \\")
    print("                   --eval-data data/validation.yaml")
    print("\nOr use Docker:")
    print("   ./scripts/docker_setup.sh setup")
    print("   ./scripts/docker_setup.sh train")
    print("\nFor help:")
    print("   python train.py --help")
    print("   ./scripts/docker_setup.sh help")
    print("   python scripts/run_tests.py --help")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup CodeLlama fine-tuning pipeline")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-sample-data", action="store_true", help="Skip creating sample data")
    parser.add_argument("--dev", action="store_true", help="Setup for development (includes git hooks)")
    
    args = parser.parse_args()
    
    print("CodeLlama Fine-tuning Pipeline Setup")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            sys.exit(1)
    
    # Check GPU
    check_gpu_availability()
    
    # Check Docker
    check_docker()
    
    # Create sample data
    if not args.skip_sample_data:
        if not create_sample_data():
            print("⚠️  Failed to create sample data, but continuing...")
    
    # Setup git hooks for development
    if args.dev:
        setup_git_hooks()
    
    # Run tests
    if not args.skip_tests:
        if not run_tests():
            print("⚠️  Some tests failed, but setup is complete")
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()
