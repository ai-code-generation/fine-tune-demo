#!/usr/bin/env python3
"""
Test script to verify Hugging Face token authentication
"""

import os
import sys
import subprocess
import argparse

def test_token_with_git(token, model_name="meta-llama/CodeLlama-13b-hf"):
    """Test token with git clone method."""
    print(f"🔍 Testing token with git clone method...")
    
    # Create test directory
    test_dir = "test_model_download"
    
    # Clean up if exists
    if os.path.exists(test_dir):
        subprocess.run(["rm", "-rf", test_dir], check=True)
    
    # Test git clone with token
    authenticated_url = f"https://oauth:{token}@huggingface.co/{model_name}"
    
    cmd = ["git", "clone", authenticated_url, test_dir]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ Git clone with token authentication successful!")
            # Clean up
            subprocess.run(["rm", "-rf", test_dir], check=True)
            return True
        else:
            print(f"❌ Git clone failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Git clone timed out")
        return False
    except Exception as e:
        print(f"❌ Git clone error: {e}")
        return False

def test_token_with_hub(token, model_name="meta-llama/CodeLlama-13b-hf"):
    """Test token with huggingface_hub library."""
    print(f"🔍 Testing token with huggingface_hub library...")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        
        # Test by getting user info
        user_info = api.whoami()
        print(f"✅ Token is valid! Logged in as: {user_info['name']}")
        
        # Test model access
        try:
            model_info = api.model_info(model_name, token=token)
            print(f"✅ Model access confirmed: {model_info.modelId}")
            return True
        except Exception as e:
            print(f"❌ Model access failed: {e}")
            print("This might be a gated model - check if you have access")
            return False
            
    except ImportError:
        print("❌ huggingface_hub library not installed")
        print("Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        return test_token_with_hub(token, model_name)
    except Exception as e:
        print(f"❌ Token test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test Hugging Face token authentication")
    parser.add_argument("--token", help="Hugging Face token to test")
    parser.add_argument("--model", default="meta-llama/CodeLlama-13b-hf", help="Model to test access")
    
    args = parser.parse_args()
    
    # Get token from argument, environment, or .env file
    token = args.token
    
    if not token:
        token = os.environ.get('HF_TOKEN')
    
    if not token and os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if line.startswith('HF_TOKEN='):
                    token = line.split('=', 1)[1].strip()
                    break
    
    if not token:
        print("❌ No Hugging Face token found!")
        print("Please provide token via:")
        print("1. --token argument")
        print("2. HF_TOKEN environment variable")
        print("3. HF_TOKEN in .env file")
        print("4. Run: python setup_hf_auth.py")
        return 1
    
    print(f"🧪 Testing Hugging Face Token Authentication")
    print(f"Model: {args.model}")
    print(f"Token: {token[:10]}...")
    print("=" * 50)
    
    # Test with huggingface_hub first (more reliable)
    hub_success = test_token_with_hub(token, args.model)
    print()
    
    # Test with git clone
    git_success = test_token_with_git(token, args.model)
    print()
    
    if hub_success or git_success:
        print("🎉 Token authentication working!")
        print("You can now run the fine-tuning pipeline.")
        return 0
    else:
        print("❌ Token authentication failed!")
        print("Please check your token and try again.")
        print("Get a new token from: https://huggingface.co/settings/tokens")
        return 1

if __name__ == "__main__":
    exit(main())
