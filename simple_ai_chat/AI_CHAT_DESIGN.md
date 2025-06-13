# AI Chat Application - Thiết Kế Đơn Giản

## Tổng Quan Hệ Thống

Ứng dụng chat AI đơn giản với workflow: PDF Documents → Fine-tuning → Chat Terminal

## Kiến Trúc Đơn Giản

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Files     │───▶│  Process & Train │───▶│  Terminal Chat  │
│   (Folder)      │    │                  │    │   Interface     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Workflow Chính
1. **Input**: Folder chứa file PDF
2. **Process**: Trích xuất text từ PDF + tạo dataset
3. **Train**: Fine-tune model với LoRA
4. **Chat**: Terminal interface để chat với AI

## 1. Cấu Trúc Thư Mục Đơn Giản

```
simple_ai_chat/
├── README.md
├── requirements.txt
├── config.yaml                # Configuration file
├── data/
│   ├── pdfs/                  # Input PDF files folder
│   ├── processed/             # Processed text data
│   └── training_data.json     # Training dataset
├── models/
│   ├── base/                  # Downloaded base model
│   └── fine_tuned/            # Fine-tuned model
├── src/
│   ├── pdf_processor.py       # PDF to text extraction
│   ├── data_creator.py        # Create training dataset
│   ├── model_trainer.py       # Fine-tuning logic
│   └── chat_app.py           # Terminal chat interface
├── logs/
└── main.py                    # Main entry point
```

## 2. Core Files Chi Tiết

### 2.1 `pdf_processor.py` - Xử Lý PDF
```python
import PyPDF2
import os
from typing import List, Dict

class PDFProcessor:
    """Xử lý PDF files từ folder input"""
    
    def __init__(self, pdf_folder: str):
        self.pdf_folder = pdf_folder
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Trích xuất text từ 1 file PDF"""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def process_all_pdfs(self) -> List[Dict]:
        """Process tất cả PDF trong folder"""
        documents = []
        for filename in os.listdir(self.pdf_folder):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_folder, filename)
                text = self.extract_text_from_pdf(pdf_path)
                documents.append({
                    'filename': filename,
                    'content': self.clean_text(text)
                })
        return documents
    
    def clean_text(self, text: str) -> str:
        """Làm sạch text cơ bản"""
        # Loại bỏ newlines thừa, ký tự đặc biệt
        text = text.replace('\n', ' ').replace('\t', ' ')
        # Loại bỏ khoảng trắng thừa
        text = ' '.join(text.split())
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Chia text thành chunks nhỏ"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
```

### 2.2 `data_creator.py` - Tạo Training Dataset
```python
import json
import random
from typing import List, Dict

class DatasetCreator:
    """Tạo training dataset từ processed documents"""
    
    def __init__(self):
        self.question_templates = [
            "Tóm tắt nội dung về {}",
            "Giải thích về {}",
            "Thông tin chi tiết về {}",
            "Nói về {}",
            "Mô tả {}",
        ]
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords đơn giản từ text"""
        # Đơn giản: lấy những từ dài > 4 ký tự
        words = text.split()
        keywords = [word for word in words 
                   if len(word) > 4 and word.isalpha()]
        return list(set(keywords))[:10]  # Top 10 keywords
    
    def generate_qa_from_chunk(self, chunk: str) -> List[Dict]:
        """Generate Q&A pairs từ text chunk"""
        qa_pairs = []
        keywords = self.extract_keywords(chunk)
        
        for keyword in keywords[:3]:  # 3 Q&A per chunk
            question = random.choice(self.question_templates).format(keyword)
            qa_pairs.append({
                "input": question,
                "output": chunk,
                "context": f"Dựa trên tài liệu: {chunk[:100]}..."
            })
        
        return qa_pairs
    
    def create_training_dataset(self, documents: List[Dict]) -> List[Dict]:
        """Tạo complete training dataset"""
        training_data = []
        
        for doc in documents:
            # Chia document thành chunks
            chunks = self.chunk_text(doc['content'])
            
            for chunk in chunks:
                qa_pairs = self.generate_qa_from_chunk(chunk)
                training_data.extend(qa_pairs)
        
        return training_data
    
    def chunk_text(self, text: str, chunk_size: int = 300) -> List[str]:
        """Chia text thành chunks"""
        sentences = text.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + "."
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "."
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def save_dataset(self, dataset: List[Dict], output_path: str):
        """Lưu dataset ra file JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
```

### 2.3 `model_trainer.py` - Fine-tuning Model
```python
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

class ModelTrainer:
    """Fine-tune model với LoRA"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def load_model_and_tokenizer(self):
        """Load base model và tokenizer"""
        print(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Add pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def setup_lora(self):
        """Setup LoRA configuration"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj"]  # For GPT models
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print(f"LoRA model parameters: {self.model.num_parameters()}")
    
    def prepare_dataset(self, training_data_path: str):
        """Chuẩn bị dataset cho training"""
        with open(training_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Format data cho conversation
        formatted_data = []
        for item in data:
            text = f"Human: {item['input']}\nAssistant: {item['output']}"
            formatted_data.append({"text": text})
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
        
        dataset = Dataset.from_list(formatted_data)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def train(self, dataset, output_dir: str = "./models/fine_tuned"):
        """Thực hiện fine-tuning"""
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
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        print("Bắt đầu training...")
        trainer.train()
        
        print("Lưu model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
```

### 2.4 `chat_app.py` - Terminal Chat Interface
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

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
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        print("Model loaded successfully!")
    
    def generate_response(self, user_input: str) -> str:
        """Generate response từ user input"""
        # Format input
        prompt = f"Human: {user_input}\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant part
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response
    
    def display_welcome(self):
        """Hiển thị welcome message"""
        print("\n" + "="*50)
        print("   AI CHAT - Fine-tuned trên Documents của bạn")
        print("="*50)
        print("Commands:")
        print("  /clear  - Xóa lịch sử chat")
        print("  /quit   - Thoát ứng dụng")
        print("  /help   - Hiển thị help")
        print("-"*50 + "\n")
    
    def handle_command(self, user_input: str) -> bool:
        """Xử lý special commands"""
        if user_input == "/quit":
            print("Tạm biệt!")
            return True
        elif user_input == "/clear":
            self.conversation_history = []
            print("Đã xóa lịch sử chat.")
            return False
        elif user_input == "/help":
            self.display_welcome()
            return False
        return False
    
    def run(self):
        """Main chat loop"""
        self.load_model()
        self.display_welcome()
        
        while True:
            try:
                user_input = input("Bạn: ").strip()
                
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
                print("\n\nTạm biệt!")
                break
            except Exception as e:
                print(f"Lỗi: {e}")
```

### 2.5 `main.py` - Entry Point
```python
import argparse
import yaml
import os

class SimpleAIChat:
    """Main application class"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def setup_directories(self):
        """Tạo các thư mục cần thiết"""
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
        print(f"✓ Processed {len(documents)} PDF files")
        
        # Create training dataset
        creator = DatasetCreator()
        training_data = creator.create_training_dataset(documents)
        creator.save_dataset(training_data, "data/training_data.json")
        print(f"✓ Created {len(training_data)} training examples")
    
    def train_model(self):
        """Train the model"""
        from src.model_trainer import ModelTrainer
        
        print("🚀 Starting model training...")
        
        trainer = ModelTrainer(self.config['model_name'])
        trainer.load_model_and_tokenizer()
        trainer.setup_lora()
        
        dataset = trainer.prepare_dataset("data/training_data.json")
        trainer.train(dataset, self.config['output_model_path'])
        
        print("✓ Model training completed!")
    
    def start_chat(self):
        """Start chat interface"""
        from src.chat_app import ChatApp
        
        chat = ChatApp(self.config['output_model_path'])
        chat.run()
    
    def run(self, mode: str):
        """Main run method"""
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
            self.process_pdfs()
            self.train_model()
            print("\n🎉 Setup completed! Now you can chat:")
            print("python main.py chat")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple AI Chat Application")
    parser.add_argument("mode", choices=["setup", "process", "train", "chat", "all"],
                       help="Mode to run the application")
    
    args = parser.parse_args()
    
    app = SimpleAIChat()
    app.run(args.mode)
```

## 3. Configuration File

### `config.yaml`
```yaml
# Model Configuration
model_name: "microsoft/DialoGPT-small"  # Lighter model for easier setup
output_model_path: "./models/fine_tuned"

# Data Configuration  
pdf_folder: "./data/pdfs"  # Folder chứa PDF files
chunk_size: 300
max_training_examples: 1000

# Training Configuration
training:
  epochs: 3
  batch_size: 2
  learning_rate: 2e-4
  gradient_accumulation_steps: 4
  warmup_steps: 100

# Chat Configuration
chat:
  max_length: 150
  temperature: 0.7
  top_p: 0.9
```

## 4. Requirements File

### `requirements.txt`
```txt
torch>=1.9.0
transformers>=4.20.0
peft>=0.4.0
datasets>=2.0.0
PyPDF2>=3.0.0
pyyaml>=6.0
accelerate>=0.20.0
```

## 5. Simple Usage Workflow

### Bước 1: Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Create directories
python main.py setup
```

### Bước 2: Chuẩn bị Data
```bash
# Copy PDF files vào folder data/pdfs/
# Process PDFs và tạo training data
python main.py process
```

### Bước 3: Fine-tuning
```bash
# Train model
python main.py train
```

### Bước 4: Chat
```bash
# Start chat
python main.py chat
```

### Hoặc chạy all-in-one:
```bash
python main.py all
```

## 6. Model Recommendation - Đơn Giản

### **Alternative: GPT-2 Small**
- **Size**: ~500MB
- **Parameters**: 124M  
- **Memory**: ~3GB RAM
- **Tốt cho**: Chất lượng response tốt hơn

## 7. Expected Results

- **Setup time**: 5 phút
- **Processing time**: 10-30 phút (tùy số lượng PDF)
- **Training time**: 30-90 phút
- **Chat response time**: 2-5 giây
- **Total project size**: ~1GB

## 8. Troubleshooting Tips

1. **Lỗi memory**: Giảm batch_size trong config
2. **PDF không đọc được**: Thử convert PDF sang text trước
3. **Training chậm**: Sử dụng model nhỏ hơn
4. **Response kém**: Tăng số lượng training data

## 9. Future Simple Improvements

- Thêm web interface đơn giản với Streamlit
- Support thêm file formats (TXT, DOCX)
- Tự động backup model checkpoints
- Simple evaluation metrics
