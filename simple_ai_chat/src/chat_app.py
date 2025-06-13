import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import time

class ChatApp:
    """Terminal chat interface"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        
    def load_model(self):
        """Load fine-tuned model"""
        print("Loading fine-tuned model...")
        
        try:
            if not os.path.exists(self.model_path):
                print(f"Model not found at {self.model_path}")
                return False
                
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            print("Model loaded successfully!")
            
            # Check which device the model is using
            device_info = "CPU"
            if torch.cuda.is_available():
                device_info = f"GPU: {torch.cuda.get_device_name(0)}"
            print(f"Model is running on: {device_info}")
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate_response(self, user_input: str) -> str:
        """Generate response từ user input"""
        # Format input
        prompt = f"Human: {user_input}\nAssistant:"
        
        try:
            # Tokenize
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            # Generate
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 150,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            end_time = time.time()
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant part
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
                print(f"[Generated in {end_time - start_time:.2f}s]")
            return response
        except Exception as e:
            return f"Error generating response: {e}"
    def display_welcome(self):
        """Display welcome message"""
        print("\n" + "="*50)
        print("   AI CHAT - Fine-tuned on Your Documents")
        print("      (Using GPT-2 Small)")
        print("="*50)
        print("Commands:")
        print("  /clear  - Clear chat history")
        print("  /quit   - Exit application")
        print("  /help   - Display help")
        print("-"*50 + "\n")
    def handle_command(self, user_input: str) -> bool:
        """Handle special commands"""
        if user_input == "/quit":
            print("Goodbye!")
            return True
        elif user_input == "/clear":
            self.conversation_history = []
            print("Chat history cleared.")
            return False
        elif user_input == "/help":
            self.display_welcome()
            return False
        return False
    def run(self):
        """Main chat loop"""
        if not self.load_model():
            print("Cannot run chat due to error loading model")
            return
            
        self.display_welcome()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    if self.handle_command(user_input):
                        break
                    continue
                
                # Generate response
                print("AI: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                
                # Update history
                self.conversation_history.append({
                    "user": user_input,
                    "assistant": response
                })
                print()  # New line
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
