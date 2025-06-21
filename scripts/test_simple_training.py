#!/usr/bin/env python3
"""
Simple test script for the fine-tuning pipeline with fallback capabilities.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_simple_training():
    """Test the simplified training pipeline."""
    print("🧪 Testing Simple Fine-tuning Pipeline")
    print("=" * 50)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from training.trainer import NeMoTrainer
        print("✅ NeMoTrainer imported successfully")
        
        # Test trainer initialization
        print("\n🔧 Testing trainer initialization...")
        trainer = NeMoTrainer(
            model_type="llama2",
            model_size="7b"
        )
        print(f"✅ Trainer initialized in '{trainer.training_mode}' mode")
        
        # Test model setup with fallback
        print("\n🤖 Testing model setup...")
        try:
            trainer.setup_model()
            print("✅ Model setup successful")
            print(f"   Model type: {type(trainer.model).__name__}")
            if hasattr(trainer, 'tokenizer'):
                print(f"   Tokenizer: {type(trainer.tokenizer).__name__}")
        except Exception as e:
            print(f"❌ Model setup failed: {e}")
            return False
        
        # Test training info
        print("\n📊 Testing training info...")
        try:
            info = trainer.get_training_info()
            print("✅ Training info retrieved:")
            for key, value in info.items():
                print(f"   {key}: {value}")
        except Exception as e:
            print(f"⚠️  Training info failed: {e}")
        
        print("\n🎉 Simple training pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_script():
    """Test the main pipeline script with minimal parameters."""
    print("\n🚀 Testing Pipeline Script")
    print("=" * 50)
    
    try:
        # Import the pipeline
        from scripts.run_pipeline import FineTunePipeline
        
        print("📦 Pipeline script imported successfully")
        
        # Test pipeline initialization
        pipeline = FineTunePipeline(
            model_type="llama2",
            model_size="7b"
        )
        print("✅ Pipeline initialized")
        
        # Test setup (this is where the error was occurring)
        print("\n🔧 Testing pipeline setup...")
        try:
            pipeline.setup_pipeline()
            print("✅ Pipeline setup successful")
            print(f"   Trainer mode: {pipeline.trainer.training_mode}")
        except Exception as e:
            print(f"❌ Pipeline setup failed: {e}")
            return False
        
        print("\n🎉 Pipeline script test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline script test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Simple Fine-tuning Pipeline Tests")
    print("=" * 60)
    
    # Test 1: Simple training
    success1 = test_simple_training()
    
    # Test 2: Pipeline script
    success2 = test_pipeline_script()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    print(f"Simple Training: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Pipeline Script: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! The pipeline is ready for use.")
        print("\n💡 Next steps:")
        print("   1. Prepare your training data in YAML format")
        print("   2. Run: python scripts/run_pipeline.py --model-type llama2 --model-size 7b --train-file data/train.yaml")
        return True
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
