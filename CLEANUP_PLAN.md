# 🧹 Project Cleanup Plan

## 📋 Current Analysis

After reviewing the project, I found significant duplication and unnecessary files. Here's what we have:

### 🎯 **ESSENTIAL FILES (Keep)**
```
📁 Core Scripts:
├── simple_nemo_finetune_7b.py     # CodeLlama-7B (2 GPUs, optimized)
├── simple_nemo_finetune.py        # CodeLlama-13B (4 GPUs, higher quality)
├── debug_jsonl_files.py            # JSONL debugging tool
└── example_training_data.yaml      # Training data

📁 Documentation:
├── README.md                       # Main project documentation
├── SIMPLE_NEMO_GUIDE.md           # Guide for 13B model
└── CODELLAMA_7B_GUIDE.md          # Guide for 7B model

📁 Container Scripts:
├── simple_docker_run.sh           # Simple container runner
└── requirements.txt               # Python dependencies
```

### 🗑️ **REDUNDANT/OBSOLETE FILES (Remove)**
```
❌ Complex Pipeline (Replaced by simple scripts):
├── finetune_pipeline.py           # Complex, buggy implementation
├── data_preprocessing.py          # Functionality moved to simple scripts
├── setup_hf_auth.py              # Not needed with token parameter
└── configs/                       # Complex config system not needed

❌ Multiple Docker Scripts (Keep only one):
├── docker-compose.yml             # Overly complex
├── docker_start.sh                # Redundant
├── run_nemo_container.sh          # Redundant
└── fix_cache_permissions.sh       # Issues fixed in simple scripts

❌ Obsolete Documentation:
├── DOCKER_GUIDE.md               # Replaced by simple guides
├── team-assignments.md           # Project-specific, not needed
├── technical-specs.md            # Outdated specifications
└── quick_start.sh                # Replaced by simple scripts

❌ Generated/Cache Files:
├── __pycache__/                  # Python cache
├── outputs/                      # Empty output directories
└── text_memmap_dataset.py        # Downloaded for debugging, not needed
```

## 🎯 **Recommended Clean Project Structure**

```
simple-nemo-finetune/
├── 📄 README.md                           # Main documentation
├── 📄 simple_nemo_finetune_7b.py         # CodeLlama-7B script
├── 📄 simple_nemo_finetune.py            # CodeLlama-13B script  
├── 📄 debug_jsonl_files.py               # Debugging tool
├── 📄 simple_docker_run.sh               # Container runner
├── 📄 requirements.txt                   # Dependencies
├── 📄 example_training_data.yaml         # Sample data
├── 📄 SIMPLE_NEMO_GUIDE.md              # 13B guide
├── 📄 CODELLAMA_7B_GUIDE.md             # 7B guide
└── 📄 LICENSE                           # License file
```

## 🔄 **Key Differences Between Scripts**

### **simple_nemo_finetune.py (CodeLlama-13B)**
- **Target**: Higher quality, production use
- **Resources**: 4 GPUs, 24GB+ VRAM each
- **Settings**: TP=2, Global batch=8, LR=1e-5
- **Use case**: Maximum quality needed

### **simple_nemo_finetune_7b.py (CodeLlama-7B)**  
- **Target**: Resource efficient, experimentation
- **Resources**: 2 GPUs, 16GB+ VRAM each
- **Settings**: TP=1, Global batch=16, LR=2e-5
- **Use case**: Limited resources, faster training

## ✅ **Benefits of Cleanup**

1. **🎯 Simplified Structure** - Only essential files
2. **📚 Clear Documentation** - Focused guides for each model
3. **🔧 Robust Scripts** - Tested, working implementations
4. **🐛 Debugging Tools** - JSONL validation and fixing
5. **📦 Easy Deployment** - Single container script

## 🚀 **Usage After Cleanup**

### **For CodeLlama-7B (Resource Efficient)**
```bash
./simple_docker_run.sh
# Inside container:
python simple_nemo_finetune_7b.py --data example_training_data.yaml --hf-token $HF_TOKEN
```

### **For CodeLlama-13B (Higher Quality)**
```bash
./simple_docker_run.sh  
# Inside container:
python simple_nemo_finetune.py --data example_training_data.yaml --hf-token $HF_TOKEN
```

### **Debug JSONL Issues**
```bash
python debug_jsonl_files.py
```
