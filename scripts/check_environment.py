#!/usr/bin/env python3
"""
Environment checker script to recommend the best configuration.
Checks GPU availability, CUDA support, and bitsandbytes functionality.
"""

import sys
import subprocess
import importlib


def check_gpu_availability():
    """Check if GPU is available and working."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"✅ GPU Available: {gpu_name}")
            print(f"   GPU Count: {gpu_count}")
            print(f"   GPU Memory: {gpu_memory:.1f} GB")
            return True, gpu_memory
        else:
            print("❌ No GPU available")
            return False, 0
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False, 0


def check_bitsandbytes():
    """Check if bitsandbytes is working properly."""
    try:
        import bitsandbytes as bnb
        
        # Try to create a simple quantization config
        from transformers import BitsAndBytesConfig
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4"
        )
        print("✅ BitsAndBytes working properly")
        return True
    except Exception as e:
        print(f"❌ BitsAndBytes not working: {e}")
        return False


def check_memory_requirements(gpu_memory_gb):
    """Check if GPU memory is sufficient for different model sizes."""
    print(f"\n📊 Memory Requirements Analysis:")
    
    requirements = {
        "CodeLlama 7B (Full Precision)": 28,
        "CodeLlama 7B (4-bit Quantized)": 8,
        "CodeLlama 13B (Full Precision)": 52,
        "CodeLlama 13B (4-bit Quantized)": 14
    }
    
    recommendations = []
    
    for model, required_gb in requirements.items():
        if gpu_memory_gb >= required_gb:
            print(f"   ✅ {model}: {required_gb} GB required, {gpu_memory_gb:.1f} GB available")
            recommendations.append(model)
        else:
            print(f"   ❌ {model}: {required_gb} GB required, {gpu_memory_gb:.1f} GB available")
    
    return recommendations


def recommend_configuration():
    """Recommend the best configuration based on environment."""
    print("🔍 Checking Environment for CodeLlama Fine-tuning")
    print("=" * 50)
    
    # Check GPU
    has_gpu, gpu_memory = check_gpu_availability()
    
    # Check bitsandbytes
    has_bnb = check_bitsandbytes()
    
    print("\n" + "=" * 50)
    print("📋 RECOMMENDATIONS")
    print("=" * 50)
    
    if not has_gpu:
        print("\n🖥️  CPU-Only Setup Detected")
        print("Recommended configuration:")
        print("  python train.py --model-config configs/model_configs/codellama_7b_cpu.yaml")
        print("\nNote: CPU training will be very slow. Consider using a smaller dataset.")
        return
    
    # GPU available
    recommendations = check_memory_requirements(gpu_memory)
    
    if has_bnb and gpu_memory >= 8:
        print("\n🚀 GPU with Quantization (Recommended)")
        print("Best configuration:")
        if "CodeLlama 7B (4-bit Quantized)" in recommendations:
            print("  python train.py --model-config configs/model_configs/codellama_7b_quantized.yaml")
        print("\nThis uses 4-bit quantization for memory efficiency.")
        
    elif gpu_memory >= 28:
        print("\n💪 High-Memory GPU Setup")
        print("Recommended configuration:")
        print("  python train.py --model-config configs/model_configs/codellama_7b.yaml")
        print("\nFull precision training with plenty of memory.")
        
    elif gpu_memory >= 8:
        print("\n⚠️  GPU Available but BitsAndBytes Issues")
        print("Recommended configuration:")
        print("  python train.py --model-config configs/model_configs/codellama_7b.yaml")
        print("\nNote: Reduce batch size if you encounter out-of-memory errors.")
        
    else:
        print("\n⚠️  Limited GPU Memory")
        print("Recommended configurations:")
        print("  1. Try CPU mode: --model-config configs/model_configs/codellama_7b_cpu.yaml")
        print("  2. Use smaller batch size with: --model-config configs/model_configs/codellama_7b.yaml")
    
    print("\n" + "=" * 50)
    print("💡 TIPS")
    print("=" * 50)
    print("• Start with a small dataset to test your setup")
    print("• Monitor GPU memory usage with: watch -n 1 nvidia-smi")
    print("• If training fails, try the CPU configuration first")
    print("• Create sample data with: python train.py --create-sample-data")


def main():
    """Main function."""
    try:
        recommend_configuration()
    except Exception as e:
        print(f"Error during environment check: {e}")
        print("\nFallback recommendation:")
        print("  python train.py --model-config configs/model_configs/codellama_7b_cpu.yaml")


if __name__ == "__main__":
    main()
