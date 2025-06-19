# LLM Server API Documentation

## Overview

The LLM Server provides OpenAI-compatible chat completion API endpoints using local HuggingFace models. It's designed to be a drop-in replacement for NVIDIA NIM or OpenAI API.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required for local deployment.

## API Endpoints

### 1. Chat Completions

**Endpoint:** `POST /v1/chat/completions`

Generate chat completions using the loaded language model.

#### Request

**Headers:**
```
Content-Type: application/json
```

**Body:**
```json
{
  "model": "local-model",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user", 
      "content": "Hello, how are you?"
    }
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 1.0,
  "stream": false
}
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Yes | - | Model identifier (use "local-model") |
| `messages` | array | Yes | - | List of conversation messages |
| `max_tokens` | integer | No | 512 | Maximum tokens to generate |
| `temperature` | float | No | 0.7 | Sampling temperature (0.0-2.0) |
| `top_p` | float | No | 1.0 | Nucleus sampling parameter |
| `stream` | boolean | No | false | Whether to stream responses |

**Message Object:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | string | Yes | Role: "system", "user", or "assistant" |
| `content` | string | Yes | Message content |

#### Response

**Success (200 OK):**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "local-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking. How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 18,
    "total_tokens": 38
  }
}
```

**Error Responses:**

```json
// 400 Bad Request
{
  "error": {
    "message": "Invalid request format",
    "type": "invalid_request_error",
    "code": "bad_request"
  }
}

// 500 Internal Server Error  
{
  "error": {
    "message": "Model generation failed",
    "type": "internal_error", 
    "code": "model_error"
  }
}
```

### 2. List Models

**Endpoint:** `GET /v1/models`

List available models.

#### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "local-model",
      "object": "model",
      "created": 1677652288,
      "owned_by": "local"
    }
  ]
}
```

### 3. Health Check Endpoints

#### Ready Check

**Endpoint:** `GET /v1/health/ready`

Check if the service is ready to accept requests.

**Response:**
```json
{
  "status": "ready",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Liveness Check

**Endpoint:** `GET /v1/health/live`

Check if the service is alive.

**Response:**
```json
{
  "status": "alive",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 4. Service Information

**Endpoint:** `GET /`

Get basic service information.

**Response:**
```json
{
  "service": "LLM Server",
  "version": "1.0.0",
  "model_path": "./models/my-finetuned-model",
  "status": "running"
}
```

## Error Handling

The API uses standard HTTP status codes:

- **200 OK**: Request successful
- **400 Bad Request**: Invalid request format or parameters
- **404 Not Found**: Endpoint not found
- **500 Internal Server Error**: Server or model error
- **503 Service Unavailable**: Model not loaded or service starting

## Rate Limiting

Currently, no rate limiting is implemented for local deployment.

## Examples

### cURL Examples

#### Basic Chat Completion
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100
  }'
```

#### Multi-turn Conversation
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Write a Python function to calculate fibonacci numbers"},
      {"role": "assistant", "content": "Here is a Python function to calculate Fibonacci numbers:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"},
      {"role": "user", "content": "Can you optimize it?"}
    ],
    "temperature": 0.3
  }'
```

#### Health Check
```bash
curl http://localhost:8000/v1/health/ready
```

### Python Examples

#### Using OpenAI SDK
```python
from openai import OpenAI

# Initialize client pointing to local server
client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

# Create chat completion
response = client.chat.completions.create(
    model="local-model",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].message.content)
```

#### Using Requests
```python
import requests

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "local-model",
    "messages": [
        {"role": "user", "content": "Explain quantum computing"}
    ],
    "max_tokens": 200,
    "temperature": 0.5
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
print(result["choices"][0]["message"]["content"])
```

## Performance Considerations

- **Model Loading**: First request may take longer as the model loads into memory
- **Memory Usage**: Large models require significant RAM/VRAM
- **Concurrent Requests**: Currently processed sequentially; consider load balancing for high traffic
- **Response Time**: Varies based on model size, hardware, and prompt length

## Compatibility

This API is compatible with:
- ✅ OpenAI Python SDK
- ✅ OpenAI JavaScript SDK  
- ✅ NVIDIA NIM API format
- ✅ Standard HTTP clients (cURL, Postman, etc.)

## Troubleshooting

### Common Issues

1. **Model not loading**: Check `HF_MODEL_LOCAL_PATH` and model file structure
2. **Out of memory**: Reduce model size or increase available RAM/VRAM
3. **Slow responses**: Consider using smaller model or GPU acceleration
4. **Connection refused**: Ensure server is running and port 8000 is accessible

### Debug Endpoints

- `GET /v1/health/ready` - Check model loading status
- `GET /` - View current configuration
- Check Docker logs: `docker logs llm-server`
