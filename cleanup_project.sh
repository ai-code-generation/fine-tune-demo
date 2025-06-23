#!/bin/bash

# 🧹 Project Cleanup Script
# This script removes redundant and obsolete files, keeping only essential components

echo "🧹 Starting project cleanup..."
echo "This will remove redundant files and keep only essential components."
echo ""

# Show what will be removed
echo "📋 Files and directories that will be REMOVED:"
echo "❌ Complex pipeline files:"
echo "   - finetune_pipeline.py"
echo "   - data_preprocessing.py" 
echo "   - setup_hf_auth.py"
echo "   - configs/"
echo ""
echo "❌ Redundant Docker scripts:"
echo "   - docker-compose.yml"
echo "   - docker_start.sh"
echo "   - run_nemo_container.sh"
echo "   - fix_cache_permissions.sh"
echo ""
echo "❌ Obsolete documentation:"
echo "   - DOCKER_GUIDE.md"
echo "   - team-assignments.md"
echo "   - technical-specs.md"
echo "   - quick_start.sh"
echo ""
echo "❌ Generated/cache files:"
echo "   - __pycache__/"
echo "   - outputs/"
echo "   - text_memmap_dataset.py"
echo ""

echo "✅ Files that will be KEPT:"
echo "   - simple_nemo_finetune.py (CodeLlama-13B)"
echo "   - simple_nemo_finetune_7b.py (CodeLlama-7B)"
echo "   - debug_jsonl_files.py"
echo "   - simple_docker_run.sh"
echo "   - example_training_data.yaml"
echo "   - README.md"
echo "   - SIMPLE_NEMO_GUIDE.md"
echo "   - CODELLAMA_7B_GUIDE.md"
echo "   - requirements.txt"
echo "   - LICENSE"
echo ""

# Ask for confirmation
read -p "🤔 Do you want to proceed with cleanup? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled."
    exit 1
fi

echo "🚀 Starting cleanup..."

# Remove complex pipeline files
echo "🗑️  Removing complex pipeline files..."
rm -f finetune_pipeline.py
rm -f data_preprocessing.py
rm -f setup_hf_auth.py
rm -rf configs/

# Remove redundant Docker scripts
echo "🗑️  Removing redundant Docker scripts..."
rm -f docker-compose.yml
rm -f docker_start.sh
rm -f run_nemo_container.sh
rm -f fix_cache_permissions.sh

# Remove obsolete documentation
echo "🗑️  Removing obsolete documentation..."
rm -f DOCKER_GUIDE.md
rm -f team-assignments.md
rm -f technical-specs.md
rm -f quick_start.sh

# Remove generated/cache files
echo "🗑️  Removing generated/cache files..."
rm -rf __pycache__/
rm -rf outputs/
rm -f text_memmap_dataset.py

# Remove cleanup files themselves
echo "🗑️  Removing cleanup files..."
rm -f CLEANUP_PLAN.md
rm -f cleanup_project.sh

echo ""
echo "✅ Cleanup completed successfully!"
echo ""
echo "📁 Final project structure:"
echo "├── 📄 README.md"
echo "├── 📄 simple_nemo_finetune.py (CodeLlama-13B)"
echo "├── 📄 simple_nemo_finetune_7b.py (CodeLlama-7B)"
echo "├── 📄 debug_jsonl_files.py"
echo "├── 📄 simple_docker_run.sh"
echo "├── 📄 requirements.txt"
echo "├── 📄 example_training_data.yaml"
echo "├── 📄 SIMPLE_NEMO_GUIDE.md"
echo "├── 📄 CODELLAMA_7B_GUIDE.md"
echo "└── 📄 LICENSE"
echo ""
echo "🎯 Usage:"
echo "For CodeLlama-7B:  python simple_nemo_finetune_7b.py --data example_training_data.yaml --hf-token \$HF_TOKEN"
echo "For CodeLlama-13B: python simple_nemo_finetune.py --data example_training_data.yaml --hf-token \$HF_TOKEN"
echo "Debug JSONL:       python debug_jsonl_files.py"
echo ""
echo "🎉 Project is now clean and ready to use!"
