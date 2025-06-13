import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
import os

class ModelTrainer:
    """Fine-tune GPT-2 Small với LoRA"""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def load_model_and_tokenizer(self):
        """Load base model và tokenizer"""
        print(f"Loading model: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Add pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print(f"Model loaded successfully with {sum(p.numel() for p in self.model.parameters())} parameters")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def setup_lora(self):
        """Setup LoRA configuration cho GPT-2 Small"""
        try:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["c_attn", "c_proj"]  # For GPT-2 models
            )
            
            self.model = get_peft_model(self.model, lora_config)
            print(f"LoRA model setup completed. Trainable parameters: {self.model.print_trainable_parameters()}")
            return True
        except Exception as e:
            print(f"Error setting up LoRA: {e}")
            return False
    
    def prepare_dataset(self, training_data_path: str):
        """Chuẩn bị dataset cho training"""
        try:
            if not os.path.exists(training_data_path):
                print(f"Training data not found at {training_data_path}")
                return None
                
            with open(training_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"Loaded {len(data)} training examples")
              # Format data cho conversation
            formatted_data = []
            for i, item in enumerate(data):
                try:
                    text = f"Human: {item['input']}\nAssistant: {item['output']}"
                    formatted_data.append({"text": text})
                except Exception as e:
                    print(f"Error processing item {i}: {e}")
                    print(f"Problematic item: {item}")
                    continue
              # Tokenize
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    # Bỏ return_tensors="pt" ra khỏi hàm map để tránh lỗi tensor shape
                )
              # Tạo dataset và kiểm tra format
            print(f"Sample data item format: {formatted_data[0] if formatted_data else 'No data'}")
            dataset = Dataset.from_list(formatted_data)
            
            # In thông tin về dataset để debug
            print(f"Dataset columns before tokenization: {dataset.column_names}")
            
            # Tokenize với xử lý lỗi tốt hơn
            try:
                tokenized_dataset = dataset.map(
                    tokenize_function, 
                    batched=True, 
                    remove_columns=["text"],
                    desc="Tokenizing dataset"
                )
                print(f"Dataset columns after tokenization: {tokenized_dataset.column_names}")
            except Exception as e:                
                print(f"Error during tokenization: {e}")
                # Try a different approach if the above fails
                print("Trying alternative tokenization approach...")
                
                # Create tokenized dataset manually
                tokenized_data = []
                for item in formatted_data:
                    try:
                        encoding = self.tokenizer(
                            item["text"],
                            truncation=True, 
                            padding="max_length",
                            max_length=512
                        )
                        tokenized_data.append(encoding)
                    except Exception as e:
                        print(f"Error tokenizing item: {e}")
                
                if tokenized_data:
                    tokenized_dataset = Dataset.from_list(tokenized_data)
                else:
                    print("Failed to create any tokenized data")
                    return None
            
            print(f"Dataset prepared with {len(tokenized_dataset)} examples")
            return tokenized_dataset
        except Exception as e:
            print(f"Error preparing dataset: {e}")
            return None
    def train(self, dataset, output_dir: str = "./models/fine_tuned"):
        """Perform fine-tuning"""
        try:
            if dataset is None:
                print("No dataset provided for training")
                return False
                
            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                num_train_epochs=3,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=100,
                logging_steps=50,
                save_steps=500,
                learning_rate=2e-4,
                fp16=torch.cuda.is_available(),
                logging_dir="./logs",
                remove_unused_columns=False,
            )            # Use DataCollatorForLanguageModeling with appropriate configuration
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # Causal language modeling, not masked language modeling
                pad_to_multiple_of=8,  # Ensure padding is compatible with batch size
            )            # Check and verify dataset before training
            print(f"Training dataset size: {len(dataset)}")
            print(f"Training dataset keys: {list(dataset[0].keys()) if len(dataset) > 0 else 'No data'}")
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
            )
            print("Starting training...")
            trainer.train()
            
            print("Saving model...")
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            print(f"Model saved to {output_dir}")
            return True
        except Exception as e:
            print(f"Error during training: {e}")
            return False
