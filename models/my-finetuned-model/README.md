# My Fine-tuned Model

This directory should contain your fine-tuned model files.

## Required Files:
- `config.json` - Model configuration
- `pytorch_model.bin` or `*.safetensors` - Model weights

## Optional Files:
- `tokenizer_config.json` - Tokenizer configuration
- `tokenizer.json` - Tokenizer data
- `vocab.txt` - Vocabulary file
- `merges.txt` - BPE merges (for GPT-2 style models)
- `special_tokens_map.json` - Special tokens mapping

## Usage:
1. Replace this README with your actual fine-tuned model files
2. Make sure all required files are present
3. Deploy with `./deploy.sh`

## Example Download Command:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Download a model to this directory
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')

model.save_pretrained('./models/my-finetuned-model')
tokenizer.save_pretrained('./models/my-finetuned-model')
```
