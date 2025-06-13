"""
Ví dụ sử dụng mô hình đã fine-tune trong một ứng dụng đơn giản
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import os
from typing import List, Dict

class SimpleAIChatbot:
    """Demo class for using the fine-tuned model in applications"""
    
    def __init__(self, model_path: str = "./models/fine_tuned"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        # Load model config
        if os.path.exists("config.yaml"):
            with open("config.yaml", "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                "chat": {
                    "max_length": 150,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
    
    def load_model(self) -> bool:
        """Load the fine-tuned model"""
        try:
            if not os.path.exists(self.model_path):
                print(f"Model not found at {self.model_path}")
                return False
            
            print(f"Loading model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate_response(self, user_input: str) -> str:
        """Generate a response to the user input"""
        if self.model is None or self.tokenizer is None:
            if not self.load_model():
                return "Error: Model could not be loaded"
        
        try:
            # Format input
            prompt = f"Human: {user_input}\nAssistant:"
            
            # Tokenize
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            # Get generation parameters from config
            chat_config = self.config.get("chat", {})
            max_new_tokens = chat_config.get("max_length", 150)
            temperature = chat_config.get("temperature", 0.7)
            top_p = chat_config.get("top_p", 0.9)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant part
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            return response
        except Exception as e:
            return f"Error generating response: {e}"
    
    def batch_process_questions(self, questions: List[str]) -> List[Dict]:
        """Process multiple questions and generate responses"""
        results = []
        
        for i, question in enumerate(questions):
            print(f"Processing question {i+1}/{len(questions)}: {question}")
            response = self.generate_response(question)
            results.append({"question": question, "answer": response})
        
        return results

# Demo usage
if __name__ == "__main__":
    print("="*50)
    print("Simple AI Chatbot Demo")
    print("="*50)
    
    # Create chatbot instance
    chatbot = SimpleAIChatbot()
    
    # Example questions
    example_questions = [
        "Bạn có thể giải thích về GPT-2 Small không?",
        "Kỹ thuật LoRA dùng để làm gì?",
        "Tóm tắt về xử lý tài liệu PDF"
    ]
    
    # Check if model exists first
    if not os.path.exists("./models/fine_tuned"):
        print("\n⚠️  Model chưa được huấn luyện!")
        print("Vui lòng chạy:")
        print("  python main.py train")
        print("trước khi sử dụng demo này.")
    else:
        # Process example questions
        print("\nXử lý các câu hỏi mẫu:")
        results = chatbot.batch_process_questions(example_questions)
        
        # Print results
        for i, result in enumerate(results):
            print(f"\nCâu hỏi {i+1}: {result['question']}")
            print(f"Trả lời: {result['answer']}")
        
        # Interactive mode
        print("\n" + "-"*50)
        print("Chế độ tương tác (nhập 'q' để thoát):")
        while True:
            user_input = input("\nBạn: ")
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
            
            response = chatbot.generate_response(user_input)
            print(f"AI: {response}")
