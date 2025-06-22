#!/usr/bin/env python3
"""
Hugging Face Authentication Setup Script
This script helps users set up their Hugging Face authentication properly.
"""

import os
import sys
import subprocess
import getpass
from pathlib import Path

def print_status(message):
    print(f"✅ {message}")

def print_warning(message):
    print(f"⚠️  {message}")

def print_error(message):
    print(f"❌ {message}")

def print_info(message):
    print(f"ℹ️  {message}")

def check_hf_hub_installation():
    """Check if huggingface_hub is installed."""
    try:
        import huggingface_hub
        print_status(f"huggingface_hub is installed (version: {huggingface_hub.__version__})")
        return True
    except ImportError:
        print_warning("huggingface_hub is not installed")
        return False

def install_hf_hub():
    """Install huggingface_hub library."""
    print_info("Installing huggingface_hub...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        print_status("huggingface_hub installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install huggingface_hub: {e}")
        return False

def setup_hf_token():
    """Set up Hugging Face token."""
    print_info("Setting up Hugging Face authentication...")
    print()
    print("You need a Hugging Face access token to download models.")
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'read' permissions")
    print("3. Copy the token and paste it below")
    print()
    
    token = getpass.getpass("Enter your Hugging Face token (input will be hidden): ")
    
    if not token or not token.startswith('hf_'):
        print_error("Invalid token format. Hugging Face tokens should start with 'hf_'")
        return None
    
    return token

def test_token(token):
    """Test if the token works."""
    print_info("Testing Hugging Face token...")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        
        # Test by getting user info
        user_info = api.whoami()
        print_status(f"Token is valid! Logged in as: {user_info['name']}")
        return True
        
    except Exception as e:
        print_error(f"Token test failed: {e}")
        print_error("Please check your token and try again")
        return False

def save_token_to_env_file(token):
    """Save token to .env file for docker-compose."""
    env_file = Path(".env")
    
    # Read existing .env file if it exists
    env_content = ""
    if env_file.exists():
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        # Remove existing HF_TOKEN line
        env_content = "".join(line for line in lines if not line.startswith('HF_TOKEN='))
    
    # Add the new token
    env_content += f"HF_TOKEN={token}\n"
    
    # Write back to .env file
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print_status("Token saved to .env file for docker-compose")

def main():
    print("🤗 Hugging Face Authentication Setup")
    print("===================================")
    print()
    
    # Check if huggingface_hub is installed
    if not check_hf_hub_installation():
        if not install_hf_hub():
            print_error("Failed to install required dependencies")
            return 1
    
    # Set up token
    token = setup_hf_token()
    if not token:
        return 1
    
    # Test token
    if not test_token(token):
        return 1
    
    # Save token to .env file
    save_token_to_env_file(token)
    
    print()
    print_status("Hugging Face authentication setup complete!")
    print()
    print("Now you can:")
    print("1. Use docker-compose (token will be loaded automatically):")
    print("   docker-compose up nemo-finetuning")
    print()
    print("2. Use the pipeline directly:")
    print(f"   python finetune_pipeline.py --model codellama-13b --data your_data.yaml --hf-token [your_token]")
    print()
    print("3. Use the quick start script:")
    print("   ./quick_start.sh")
    print()
    
    return 0

if __name__ == "__main__":
    exit(main())
