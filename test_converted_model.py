#!/usr/bin/env python3
"""
Test the converted HuggingFace model for code generation.
"""

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

def test_model(model_path, prompts=None, max_length=200, temperature=0.7):
    """Test the converted model with code generation prompts."""
    
    print(f"📥 Loading model from: {model_path}")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model info:")
        print(f"   - Parameters: {model.num_parameters():,}")
        print(f"   - Device: {next(model.parameters()).device}")
        print(f"   - Dtype: {next(model.parameters()).dtype}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Default test prompts
    if prompts is None:
        prompts = [
            "def factorial(n):",
            "# Create a function to calculate the sum of two numbers\ndef add_numbers(",
            "class Calculator:",
            "// Function to reverse a string\nfunction reverseString(",
            "# Write a Python function to check if a number is prime\ndef is_prime(",
        ]
    
    print("\n🧪 Testing model with code generation prompts...")
    print("=" * 60)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n🔍 Test {i}: {prompt[:50]}...")
        print("-" * 40)
        
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"📝 Input: {prompt}")
            print(f"🤖 Generated:")
            print(generated_text)
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ Generation failed for prompt {i}: {e}")
            continue
    
    return True

def interactive_test(model_path):
    """Interactive testing mode."""
    
    print(f"📥 Loading model from: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    print("\n🎮 Interactive Mode - Enter your prompts (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        try:
            prompt = input("\n📝 Enter prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt:
                continue
            
            print("🤖 Generating...")
            
            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"\n📤 Generated:")
            print(generated_text)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    return True

def benchmark_model(model_path):
    """Simple benchmark of the model."""
    
    print(f"📊 Benchmarking model: {model_path}")
    
    try:
        import time
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Warm up
        prompt = "def hello_world():"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        print("🔥 Warming up...")
        with torch.no_grad():
            _ = model.generate(**inputs, max_length=50, do_sample=False)
        
        # Benchmark
        print("⏱️ Benchmarking...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                do_sample=False,
                num_return_sequences=1
            )
        
        end_time = time.time()
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generation_time = end_time - start_time
        tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
        tokens_per_second = tokens_generated / generation_time
        
        print(f"📊 Benchmark Results:")
        print(f"   - Generation time: {generation_time:.2f} seconds")
        print(f"   - Tokens generated: {tokens_generated}")
        print(f"   - Tokens per second: {tokens_per_second:.2f}")
        print(f"   - Generated text: {generated_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test converted HuggingFace model")
    parser.add_argument("--model-path", default="./converted_model", 
                       help="Path to converted HuggingFace model")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark")
    parser.add_argument("--max-length", type=int, default=200,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature")
    
    args = parser.parse_args()
    
    print("🧪 HuggingFace Model Tester")
    print("=" * 30)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        print("💡 Make sure you've run the conversion script first")
        sys.exit(1)
    
    # Check for required files
    required_files = ["config.json", "pytorch_model.bin"]
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(args.model_path, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing files: {missing_files}")
        print("💡 Model might not be fully converted")
    
    print(f"📁 Model path: {args.model_path}")
    
    if args.benchmark:
        benchmark_model(args.model_path)
    elif args.interactive:
        interactive_test(args.model_path)
    else:
        test_model(args.model_path, max_length=args.max_length, temperature=args.temperature)
    
    print("\n✅ Testing completed!")

if __name__ == "__main__":
    main()
