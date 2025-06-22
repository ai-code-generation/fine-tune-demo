#!/usr/bin/env python3
"""
Script to check available NeMo conversion scripts and environment setup
"""

import os
import sys
import subprocess
from pathlib import Path

def print_status(message):
    print(f"✅ {message}")

def print_warning(message):
    print(f"⚠️  {message}")

def print_error(message):
    print(f"❌ {message}")

def print_info(message):
    print(f"ℹ️  {message}")

def check_nemo_installation():
    """Check if NeMo is installed and get version."""
    try:
        import nemo
        print_status(f"NeMo is installed (version: {nemo.__version__})")
        print_info(f"NeMo location: {nemo.__file__}")
        return True
    except ImportError:
        print_error("NeMo is not installed")
        return False

def find_conversion_scripts():
    """Find available conversion scripts."""
    print_info("Searching for NeMo conversion scripts...")
    
    possible_locations = [
        "/opt/NeMo",
        "/workspace/NeMo", 
        "/usr/local/lib/python*/site-packages/nemo",
        "/home/*/miniconda*/envs/*/lib/python*/site-packages/nemo",
        "/home/*/.local/lib/python*/site-packages/nemo"
    ]
    
    script_patterns = [
        "scripts/nlp_language_modeling/convert_hf_llama_to_nemo.py",
        "scripts/checkpoint_converters/convert_hf_llama_to_nemo.py",
        "scripts/nlp_language_modeling/convert_hf_gpt_to_nemo.py",
        "scripts/checkpoint_converters/convert_hf_gpt_to_nemo.py"
    ]
    
    found_scripts = []
    
    # Check common locations
    for location in possible_locations:
        if '*' in location:
            # Use glob for wildcard paths
            import glob
            expanded_paths = glob.glob(location)
            for path in expanded_paths:
                for pattern in script_patterns:
                    script_path = os.path.join(path, pattern)
                    if os.path.exists(script_path):
                        found_scripts.append(script_path)
        else:
            for pattern in script_patterns:
                script_path = os.path.join(location, pattern)
                if os.path.exists(script_path):
                    found_scripts.append(script_path)
    
    # Also try to find NeMo installation path
    try:
        import nemo
        nemo_path = Path(nemo.__file__).parent.parent
        for pattern in script_patterns:
            script_path = nemo_path / pattern
            if script_path.exists():
                found_scripts.append(str(script_path))
    except ImportError:
        pass
    
    if found_scripts:
        print_status("Found conversion scripts:")
        for script in set(found_scripts):  # Remove duplicates
            print(f"  - {script}")
    else:
        print_warning("No conversion scripts found")
    
    return found_scripts

def check_docker_environment():
    """Check if running in Docker and what's available."""
    print_info("Checking Docker environment...")
    
    # Check if in container
    if os.path.exists('/.dockerenv'):
        print_status("Running inside Docker container")
    else:
        print_info("Not running in Docker container")
    
    # Check common Docker paths
    docker_paths = [
        "/opt/NeMo",
        "/workspace",
        "/workspace/NeMo"
    ]
    
    for path in docker_paths:
        if os.path.exists(path):
            print_status(f"Found Docker path: {path}")
            # List contents
            try:
                contents = os.listdir(path)
                if contents:
                    print(f"  Contents: {', '.join(contents[:5])}{'...' if len(contents) > 5 else ''}")
            except PermissionError:
                print_warning(f"  No permission to list contents")
        else:
            print_info(f"Docker path not found: {path}")

def check_python_environment():
    """Check Python environment and packages."""
    print_info("Checking Python environment...")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check key packages
    packages = ['torch', 'transformers', 'nemo_toolkit', 'omegaconf', 'hydra-core']
    
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            print_status(f"{package} is available")
        except ImportError:
            print_warning(f"{package} is not available")

def suggest_solutions():
    """Suggest solutions based on findings."""
    print("\n" + "="*50)
    print("🔧 SUGGESTED SOLUTIONS")
    print("="*50)
    
    print("\n1. If no conversion scripts found:")
    print("   - Make sure you're using the correct NeMo container")
    print("   - Try: docker run --gpus all nvcr.io/nvidia/nemo:25.04.01.llama_nemotron_nano_vl")
    print("   - Check if NeMo is properly installed")
    
    print("\n2. Alternative approaches:")
    print("   - Use models that don't require conversion (like some StarCoder variants)")
    print("   - Use HuggingFace models directly with transformers library")
    print("   - Install NeMo from source with conversion scripts")
    
    print("\n3. For StarCoder models:")
    print("   - StarCoder models might not have direct NeMo conversion")
    print("   - Consider using transformers library directly")
    print("   - Or use CodeLlama/Llama models which have better NeMo support")

def main():
    print("🔍 NeMo Environment Checker")
    print("="*30)
    
    # Check NeMo installation
    nemo_installed = check_nemo_installation()
    print()
    
    # Check Python environment
    check_python_environment()
    print()
    
    # Check Docker environment
    check_docker_environment()
    print()
    
    # Find conversion scripts
    scripts = find_conversion_scripts()
    print()
    
    # Suggest solutions
    suggest_solutions()
    
    return 0 if (nemo_installed and scripts) else 1

if __name__ == "__main__":
    exit(main())
