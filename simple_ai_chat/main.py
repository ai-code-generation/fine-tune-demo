import argparse
import yaml
import os
import sys

class SimpleAIChat:
    """Main application class"""
    
    def __init__(self, config_path: str = "config.yaml"):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                print(f"Loaded configuration from {config_path}")
        except FileNotFoundError:
            print(f"Configuration file not found: {config_path}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            "data/pdfs",
            "data/processed", 
            "models/base",
            "models/fine_tuned",
            "logs"
        ]
        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)
        print("✓ Directories created")
    
    def process_pdfs(self):
        """Process PDF files"""
        from src.pdf_processor import PDFProcessor
        from src.data_creator import DatasetCreator
        
        print("🔄 Processing PDF files...")
        
        # Process PDFs
        processor = PDFProcessor(self.config['pdf_folder'])
        documents = processor.process_all_pdfs()
        
        if not documents:
            print("⚠️ No PDF documents processed. Please add PDF files to the data/pdfs folder.")
            return False
            
        print(f"✓ Processed {len(documents)} PDF files")
        
        # Create training dataset
        creator = DatasetCreator()
        training_data = creator.create_training_dataset(documents)
        
        if not training_data:
            print("⚠️ No training data could be generated from the PDFs.")
            return False
            
        creator.save_dataset(training_data, "data/training_data.json")
        print(f"✓ Created {len(training_data)} training examples")
        return True
    
    def train_model(self):
        """Train the model"""
        from src.model_trainer import ModelTrainer
        
        print("🚀 Starting model training...")
        
        trainer = ModelTrainer(self.config['model_name'])
        if not trainer.load_model_and_tokenizer():
            print("⚠️ Failed to load model and tokenizer")
            return False
            
        if not trainer.setup_lora():
            print("⚠️ Failed to setup LoRA")
            return False
        
        dataset = trainer.prepare_dataset("data/training_data.json")
        if dataset is None:
            print("⚠️ Failed to prepare dataset")
            return False
            
        success = trainer.train(dataset, self.config['output_model_path'])
        
        if success:
            print("✓ Model training completed!")
            return True
        else:
            print("⚠️ Model training failed")
            return False
    
    def start_chat(self):
        """Start chat interface"""
        from src.chat_app import ChatApp
        
        chat = ChatApp(self.config['output_model_path'])
        chat.run()
    
    def check_requirements(self):
        """Check if all requirements are met"""
        try:
            import torch
            import transformers
            import peft
            import datasets
            import PyPDF2
            
            print("✓ All required packages are installed")
            
            # Check for CUDA availability
            if torch.cuda.is_available():
                print(f"✓ CUDA is available: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️ CUDA is not available. Training will be slower on CPU")
                
            return True
        except ImportError as e:
            print(f"⚠️ Missing required package: {e}")
            print("Please install all required packages with: pip install -r requirements.txt")
            return False
    
    def run(self, mode: str):
        """Main run method"""
        if not self.check_requirements() and mode != "setup":
            print("Please resolve the requirements issues first")
            return
            
        if mode == "setup":
            self.setup_directories()
        elif mode == "process":
            self.process_pdfs()
        elif mode == "train":
            self.train_model()
        elif mode == "chat":
            self.start_chat()
        elif mode == "all":
            self.setup_directories()
            if self.process_pdfs():
                if self.train_model():
                    print("\n🎉 Setup completed! Now you can chat:")
                    print("python main.py chat")
                else:
                    print("\n⚠️ Training failed. Please check the errors above.")
            else:
                print("\n⚠️ Processing failed. Please check the errors above.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple AI Chat Application")
    parser.add_argument("mode", choices=["setup", "process", "train", "chat", "all"],
                       help="Mode to run the application")
    
    args = parser.parse_args()
    
    app = SimpleAIChat()
    app.run(args.mode)
