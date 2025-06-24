#!/bin/bash

# 🔧 Fix File Permissions Script
# This script sets proper permissions for all files in the project

echo "🔧 Fixing file permissions for NeMo fine-tuning project..."

# Set execute permissions for scripts
echo "📝 Setting execute permissions for Python scripts..."
chmod 755 *.py

echo "📝 Setting execute permissions for shell scripts..."
chmod 755 *.sh

# Set read permissions for documentation and data files
echo "📚 Setting read permissions for documentation files..."
chmod 644 *.md *.yaml *.txt LICENSE .gitignore 2>/dev/null || true

# Set directory permissions
echo "📁 Setting directory permissions..."
find . -type d -exec chmod 755 {} \; 2>/dev/null || true

echo "✅ File permissions fixed successfully!"
echo ""
echo "📋 Current permissions:"
ls -la *.py *.sh *.md *.yaml *.txt LICENSE 2>/dev/null || true
echo ""
echo "🚀 You can now run the scripts:"
echo "   ./simple_docker_run.sh"
echo "   python simple_nemo_finetune_7b.py --help"
echo "   python simple_nemo_finetune.py --help"
echo "   python debug_jsonl_files.py"
