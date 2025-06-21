#!/bin/bash

# Script to install all optional dependencies inside the Docker container
# This includes apex and flash-attn which are excluded from the main build

echo "Installing optional dependencies for enhanced performance..."
echo "This process may take 10-15 minutes and requires significant memory."
echo ""

# Check available memory
AVAILABLE_MEM=$(free -g | awk 'NR==2{printf "%.0f", $7}')
echo "Available memory: ${AVAILABLE_MEM}GB"

if [ "$AVAILABLE_MEM" -lt 6 ]; then
    echo "Warning: Less than 6GB of memory available. Installation may fail."
    echo "Consider increasing Docker memory allocation."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "=== Installing NVIDIA Apex ==="
./scripts/install_apex.sh
APEX_STATUS=$?

echo ""
echo "=== Installing Flash Attention ==="
./scripts/install_flash_attn.sh
FLASH_STATUS=$?

echo ""
echo "=== Installation Summary ==="
if [ $APEX_STATUS -eq 0 ]; then
    echo "✅ Apex: Successfully installed"
else
    echo "❌ Apex: Installation failed"
fi

if [ $FLASH_STATUS -eq 0 ]; then
    echo "✅ Flash Attention: Successfully installed"
else
    echo "❌ Flash Attention: Installation failed"
fi

echo ""
echo "=== Testing Installations ==="
python -c "
try:
    import apex
    print('✅ Apex is available')
except ImportError:
    print('❌ Apex is not available')

try:
    import flash_attn
    print('✅ Flash Attention is available')
except ImportError:
    print('❌ Flash Attention is not available')

print('')
print('Note: The pipeline will work without these optional dependencies.')
print('They provide performance optimizations but are not required.')
"
