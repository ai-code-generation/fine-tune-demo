import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os

def test_model(model_path, prompt):
    """Test a fine-tuned GPT-2 model with a specific prompt"""
    try:
        print(f"Loading model from {model_path}...")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Format prompt
        full_prompt = f"Human: {prompt}\nAssistant:"
        print(f"Prompt: {full_prompt}")
        
        # Tokenize
        inputs = tokenizer.encode(full_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        # Generate
        print("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant part
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
            
        print("\nResponse:")
        print(response)
        return response
        
    except Exception as e:
        print(f"Error testing model: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a fine-tuned GPT-2 model")
    parser.add_argument("--model", type=str, default="./models/fine_tuned",
                        help="Path to the fine-tuned model")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt to test with")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Model path does not exist: {args.model}")
        print("Make sure to train the model first using: python main.py train")
        exit(1)
        
    test_model(args.model, args.prompt)
