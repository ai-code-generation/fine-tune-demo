# 🎯 Simple NeMo Fine-tuning for CodeLlama

A streamlined, production-ready pipeline for fine-tuning CodeLlama models using NVIDIA NeMo 24.07 framework.

## 🚀 Quick Start

### **Two Optimized Options:**

#### **🔥 CodeLlama-7B (Resource Efficient)**
- **Requirements**: 2 GPUs with 16GB+ VRAM each
- **Use case**: Limited resources, experimentation, faster training
- **Quality**: Good for most code generation tasks

#### **⭐ CodeLlama-13B (Higher Quality)**  
- **Requirements**: 4 GPUs with 24GB+ VRAM each
- **Use case**: Production deployment, maximum quality
- **Quality**: Best for complex code generation tasks

## 📋 Prerequisites

- Docker with GPU support
- NVIDIA drivers and nvidia-docker2
- HuggingFace account with access to CodeLlama models
- GPU requirements (see options above)

## 🎯 Usage

### **Step 1: Start NeMo Container**
```bash
./simple_docker_run.sh
```

### **Step 2: Choose Your Model**

#### **For CodeLlama-7B (Resource Efficient)**
```bash
# Inside container
export HF_TOKEN="your_huggingface_token"
python simple_nemo_finetune_7b.py \
    --data example_training_data.yaml \
    --max-steps 50 \
    --hf-token $HF_TOKEN
```

#### **For CodeLlama-13B (Higher Quality)**
```bash
# Inside container  
export HF_TOKEN="your_huggingface_token"
python simple_nemo_finetune.py \
    --data example_training_data.yaml \
    --max-steps 50 \
    --hf-token $HF_TOKEN
```

### **Step 3: Debug Issues (if needed)**
```bash
# Analyze and fix JSONL file issues
python debug_jsonl_files.py
```

## 📁 Project Structure

```
simple-nemo-finetune/
├── 📄 README.md                           # This file
├── 📄 simple_nemo_finetune_7b.py         # CodeLlama-7B script
├── 📄 simple_nemo_finetune.py            # CodeLlama-13B script  
├── 📄 debug_jsonl_files.py               # JSONL debugging tool
├── 📄 simple_docker_run.sh               # Container runner
├── 📄 requirements.txt                   # Python dependencies
├── 📄 example_training_data.yaml         # Sample training data
├── 📄 SIMPLE_NEMO_GUIDE.md              # Detailed guide for 13B
├── 📄 CODELLAMA_7B_GUIDE.md             # Detailed guide for 7B
└── 📄 LICENSE                           # License file
```

## 🔧 Key Features

- ✅ **Simplified Scripts**: Two optimized scripts for different resource levels
- ✅ **Robust Error Handling**: Comprehensive CUDA, JSON, and memmap error fixes
- ✅ **Official NeMo Workflow**: Follows NeMo 24.07 documentation exactly
- ✅ **Resource Optimization**: Configurations tuned for 7B vs 13B models
- ✅ **Debugging Tools**: JSONL validation and fixing utilities
- ✅ **Production Ready**: Tested, working implementations

## 📊 Model Comparison

| **Aspect** | **CodeLlama-7B** | **CodeLlama-13B** |
|------------|------------------|-------------------|
| **GPUs Required** | 2 x 16GB+ | 4 x 24GB+ |
| **Training Speed** | ⚡ Faster | 🐌 Slower |
| **Memory Usage** | 💚 Lower | 🔴 Higher |
| **Model Quality** | ✅ Good | ⭐ Better |
| **Inference Speed** | ⚡ Fast | 🐌 Slower |
| **Deployment** | 💚 Easier | 🔴 Harder |

## 🎯 When to Use Each Model

### **Use CodeLlama-7B when:**
- ✅ Limited GPU resources (2 GPUs available)
- ✅ Need faster training/inference
- ✅ Prototyping and experimentation
- ✅ Good enough quality for your use case

### **Use CodeLlama-13B when:**
- ⭐ Maximum code generation quality needed
- ⭐ Have sufficient GPU resources (4+ GPUs)
- ⭐ Production deployment with quality priority
- ⭐ Complex code generation tasks

## 📚 Documentation

- **[SIMPLE_NEMO_GUIDE.md](SIMPLE_NEMO_GUIDE.md)** - Complete guide for CodeLlama-13B
- **[CODELLAMA_7B_GUIDE.md](CODELLAMA_7B_GUIDE.md)** - Complete guide for CodeLlama-7B

## 🐛 Troubleshooting

### **Common Issues:**
1. **CUDA errors** - Run `./check_cuda_setup.sh` for diagnosis
2. **JSON parsing errors** - Run `python debug_jsonl_files.py`
3. **Memory issues** - Use CodeLlama-7B or reduce batch sizes
4. **Container issues** - Check Docker GPU access with `nvidia-smi`

### **Quick Fixes:**
```bash
# Check GPU access
nvidia-smi

# Debug JSONL files
python debug_jsonl_files.py

# Clean start (remove cache/index files)
rm -rf cache/ *.jsonl.idx.*
```

## 🎉 Success Stories

This pipeline has been tested with:
- ✅ S32K14 automotive microcontroller code generation
- ✅ Java test automation frameworks
- ✅ Embedded systems programming
- ✅ Multi-language code generation tasks

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test your changes with both 7B and 13B models
4. Submit a pull request

## 🙏 Acknowledgments

- NVIDIA NeMo team for the excellent framework
- Meta for the CodeLlama models
- Community contributors and testers
