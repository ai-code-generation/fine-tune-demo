# Simple AI Chat - Fine-tuned on Your PDF Documents

A simple AI chatbot application that can be fine-tuned on your PDF documents using GPT-2 Small and LoRA.

## Features

- Extract text from PDF documents
- Generate training dataset from PDF content
- Fine-tune GPT-2 Small model with LoRA
- Chat with the trained model through a simple terminal interface

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- PyPDF2
- CUDA-compatible GPU recommended for faster training

## Project Structure

```
simple_ai_chat/
├── data/
│   ├── pdfs/                  # Place your PDF files here
│   └── processed/             # Processed data will be stored here
├── models/
│   ├── base/                  # Base model storage
│   └── fine_tuned/            # Fine-tuned model will be stored here
├── src/
│   ├── pdf_processor.py       # PDF text extraction
│   ├── data_creator.py        # Training data creation
│   ├── model_trainer.py       # Model fine-tuning
│   └── chat_app.py            # Chat interface
├── logs/                      # Training logs
├── config.yaml                # Configuration settings
├── requirements.txt           # Dependencies
└── main.py                    # Main entry point
```

## Setup and Usage

### 1. Install Dependencies

```
pip install -r requirements.txt
```

### 2. Initial Setup

Create necessary directories:

```
python main.py setup
```

### 3. Prepare Your Data

Copy your PDF files to the `data/pdfs/` folder, then process them:

```
python main.py process
```

### 4. Train the Model

Fine-tune the model on your processed data:

```
python main.py train
```

### 5. Chat with the Model

Start a chat session with your fine-tuned model:

```
python main.py chat
```

### Or Run Everything at Once

```
python main.py all
```

## Configuration

Edit `config.yaml` to customize settings:

- Model parameters
- Training settings 
- Chat parameters

## Chat Commands

During chat sessions, you can use these commands:
- `/clear` - Clear chat history
- `/quit` - Exit the application
- `/help` - Show help information

## Troubleshooting

- **Memory Errors**: Reduce batch size in config.yaml
- **PDF Reading Issues**: Try converting to text format first
- **Slow Training**: Use a smaller model or reduce training parameters
- **Poor Responses**: Add more training data or increase training epochs

## Notes about GPT-2 Small

This implementation uses GPT-2 Small (124M parameters), which:
- Requires ~500MB disk space
- Uses ~3GB RAM during operation
- Provides reasonable quality responses
- Can run on consumer-grade hardware
