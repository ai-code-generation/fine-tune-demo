"""
Initialization script and usage guide for Simple AI Chat
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check Python version"""
    required_version = (3, 7, 0)
    current_version = sys.version_info
    if current_version < required_version:
        print(f"⚠️  Python 3.7+ required but current version is {current_version.major}.{current_version.minor}.{current_version.micro}")
        return False
    else:
        print(f"✓ Python version: {current_version.major}.{current_version.minor}.{current_version.micro}")
        return True

def check_gpu():
    """Check GPU and CUDA"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"✓ GPU available: {device_name} (CUDA: {torch.version.cuda})")
            return True
        else:
            print("⚠️  GPU not available. Training will be slower on CPU.")
            return False
    except ImportError:
        print("⚠️  PyTorch is not installed.")
        return False
    except Exception as e:
        print(f"⚠️  Error checking GPU: {e}")
        return False

def install_requirements():
    """Install required packages"""
    print("\n🔄 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Required packages installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Error installing packages: {e}")
        return False

def show_welcome_message():
    """Display welcome message and instructions"""
    welcome_text = """
╔══════════════════════════════════════════════════════════╗
║                   SIMPLE AI CHAT - GPT-2                 ║
║                                                          ║
║     Simple AI chat app using GPT-2 Small and LoRA        ║
╚══════════════════════════════════════════════════════════╝
    
Usage Instructions:

1. Prepare Data:
   - Add PDF files to the data/pdfs/ folder
   - Run: python manage_pdfs.py --add <path_to_pdf_file>

2. Process and Prepare Data:
   - Run: python main.py process

3. Train the Model:
   - Run: python main.py train

4. Chat with the Trained Model:
   - Run: python main.py chat

Or perform all steps at once:
   - Run: python main.py all

Other Utility Tools:
   - List PDFs: python manage_pdfs.py --list
   - Test model: python test_model.py --prompt "your question"
   - Create sample PDF: python create_pdf.py --sample
"""
    print(welcome_text)

def setup():
    """Initial setup for the project"""
    print("🔍 Checking environment...\n")
    
    # Check Python version
    if not check_python_version():
        print("⚠️  Please upgrade to Python 3.7 or higher.")
    
    # Install required packages
    install_requirements()
    
    # Check GPU
    check_gpu()
    
    # Create necessary directories    
    try:
        os.makedirs("data/pdfs", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("models/base", exist_ok=True)
        os.makedirs("models/fine_tuned", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        print("✓ Directory structure created")
    except Exception as e:
        print(f"⚠️  Error creating directories: {e}")
    
    # Check configuration file
    if os.path.exists("config.yaml"):
        print("✓ Configuration file config.yaml found")
    else:
        print("⚠️  Configuration file config.yaml not found")
    
    # Hiển thị thông báo chào mừng
    show_welcome_message()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("      KHỞI TẠO SIMPLE AI CHAT - GPT-2 SMALL")
    print("="*60 + "\n")
    
    setup()
