#!/bin/bash

# Quick test script for manual LLM server

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default server URL (matching run_manual.sh default)
SERVER_URL="http://127.0.0.1:8884"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --url)
            SERVER_URL="$2"
            shift 2
            ;;
        --port)
            SERVER_URL="http://127.0.0.1:$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--url URL] [--port PORT]"
            echo "Quick test for manual LLM server"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_info "Testing LLM Server at: $SERVER_URL"
echo

# Test 1: Health check
print_info "Test 1: Health check"
if curl -s -f "$SERVER_URL/v1/health/ready" > /dev/null; then
    print_info "✅ Health check passed"
else
    print_error "❌ Health check failed"
    print_error "Is the server running? Try: ./run_manual.sh"
    exit 1
fi

# Test 2: Service info
print_info "Test 2: Service info"
response=$(curl -s "$SERVER_URL/")
if echo "$response" | grep -q "LLM Server"; then
    print_info "✅ Service info OK"
    echo "   $(echo "$response" | grep -o '"service":"[^"]*"' | cut -d'"' -f4)"
else
    print_warning "⚠️  Service info response unexpected"
fi

# Test 3: Models endpoint
print_info "Test 3: Models list"
if curl -s -f "$SERVER_URL/v1/models" > /dev/null; then
    print_info "✅ Models endpoint OK"
else
    print_warning "⚠️  Models endpoint failed"
fi

# Test 4: Simple chat completion
print_info "Test 4: Chat completion"
chat_response=$(curl -s -X POST "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "max_tokens": 10
  }')

if echo "$chat_response" | grep -q "choices"; then
    print_info "✅ Chat completion working"
    content=$(echo "$chat_response" | grep -o '"content":"[^"]*"' | head -1 | cut -d'"' -f4)
    print_info "   Response: $content"
else
    print_error "❌ Chat completion failed"
    print_error "Response: $chat_response"
    exit 1
fi

echo
print_info "🎉 All tests passed! Server is working correctly."
print_info "Ready for development and debugging."
