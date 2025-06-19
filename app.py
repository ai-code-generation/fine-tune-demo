import os
import time
import uvicorn
import logging
import asyncio
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Server - HuggingFace Backend",
    description="OpenAI/NIM compatible LLM inference server using HuggingFace models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for HuggingFace backend
llm_backend = None

# Pydantic models matching OpenAI/NIM API
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message (system, user, assistant)")
    content: str = Field(..., description="Content of the message")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="llama2", description="Model to use")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    max_tokens: Optional[int] = Field(default=512, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Temperature")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling")
    stream: Optional[bool] = Field(default=False, description="Stream response")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop sequences")

class CompletionRequest(BaseModel):
    model: str = Field(default="llama2", description="Model to use")
    prompt: str = Field(..., description="Prompt to complete")
    max_tokens: Optional[int] = Field(default=512, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.7, description="Temperature")
    top_p: Optional[float] = Field(default=1.0, description="Top-p sampling")
    stream: Optional[bool] = Field(default=False, description="Stream response")

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str] = "stop"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage

class Model(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "opensource"

class ModelList(BaseModel):
    object: str = "list"
    data: List[Model]

# Optimized HuggingFace Backend for Code Models
class HuggingFaceBackend:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        import os.path
        
        # Get model configuration from environment
        model_name_or_path = os.getenv("HF_MODEL_NAME", "microsoft/DialoGPT-medium")
        model_local_path = os.getenv("HF_MODEL_LOCAL_PATH", None)
        device = os.getenv("DEVICE", "auto")
        trust_remote_code = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"
        torch_dtype = os.getenv("TORCH_DTYPE", "auto")
        low_cpu_mem_usage = os.getenv("LOW_CPU_MEM_USAGE", "true").lower() == "true"
        
        # Initialize model type detection
        self.model_type = None
        self.is_code_model = False
        
        # Determine model source (prioritize local path)
        if model_local_path and os.path.exists(model_local_path):
            model_path = model_local_path
            logger.info(f"Loading HuggingFace model from local path: {model_path}")
            self.model_name = os.path.basename(model_path) or "local-model"
            use_local = True
        else:
            if model_local_path:
                logger.warning(f"Local model path not found: {model_local_path}, falling back to Hub")
            model_path = model_name_or_path
            logger.info(f"Loading HuggingFace model from Hub: {model_path}")
            self.model_name = model_path
            use_local = False
        
        # Detect model type for optimized handling
        self._detect_model_type(model_path)
        
        # Smart device and dtype detection for code models
        device, torch_dtype = self._optimize_device_dtype(device, torch_dtype)
        
        try:
            # Load tokenizer
            logger.info("Loading tokenizer...")
            tokenizer_kwargs = {
                "trust_remote_code": trust_remote_code,
                "local_files_only": use_local
            }
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
            
            # Fix tokenizer special tokens
            if not self.tokenizer.eos_token:
                self.tokenizer.add_special_tokens({'eos_token': '[EOS]'})
            if not self.tokenizer.bos_token:
                self.tokenizer.add_special_tokens({'bos_token': '[BOS]'})
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            elif self.tokenizer.pad_token == self.tokenizer.eos_token:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            # Load model with optimized settings
            logger.info(f"Loading model with device: {device}, dtype: {torch_dtype}")
            model_kwargs = {
                "trust_remote_code": trust_remote_code,
                "local_files_only": use_local,
                "low_cpu_mem_usage": low_cpu_mem_usage
            }
            
            # Configure dtype
            if torch_dtype == "float32":
                model_kwargs["torch_dtype"] = torch.float32
                logger.info("Using torch.float32 precision")
            elif torch_dtype == "float16":
                model_kwargs["torch_dtype"] = torch.float16
                logger.info("Using torch.float16 precision")
            
            # Configure device settings
            if device == "cuda" and torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
                logger.info("Using CUDA with automatic device mapping")
            elif device == "cpu":
                logger.info("Using CPU")
            
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            
            # Configure model
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
            
            self.device = device
            self.model_path = model_path
            
            # Log model information
            if hasattr(self.model, 'config'):
                config = self.model.config
                logger.info(f"Model config: {config.architectures if hasattr(config, 'architectures') else 'Unknown'}")
                logger.info(f"Vocab size: {config.vocab_size if hasattr(config, 'vocab_size') else 'Unknown'}")
            
            logger.info("HuggingFace model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model from {model_path}: {e}")
            raise e
    
    def _detect_model_type(self, model_path: str):
        """Detect model type and set optimization parameters"""
        model_name_lower = model_path.lower()
        
        if any(keyword in model_name_lower for keyword in ['codellama', 'code-llama', 'code_llama']):
            self.model_type = "codellama"
            self.is_code_model = True
            logger.info("Detected CodeLlama model - optimizing for code generation")
            
        elif any(keyword in model_name_lower for keyword in ['starcoder', 'star-coder', 'bigcode']):
            self.model_type = "starcoder"
            self.is_code_model = True
            logger.info("Detected StarCoder model - optimizing for code generation")
            
        elif any(keyword in model_name_lower for keyword in ['deepseek', 'deep-seek', 'deepseek-coder']):
            self.model_type = "deepseek"
            self.is_code_model = True
            logger.info("Detected DeepSeek-Coder model - optimizing for code generation")
            
        elif any(keyword in model_name_lower for keyword in ['mistral', 'mixtral']):
            self.model_type = "mistral"
            self.is_code_model = False
            logger.info("Detected Mistral model - optimizing for general conversation")
            
        elif any(keyword in model_name_lower for keyword in ['llama', 'alpaca', 'vicuna']):
            self.model_type = "llama"
            self.is_code_model = False
            logger.info("Detected Llama-based model - optimizing for general conversation")
            
        else:
            self.model_type = "general"
            self.is_code_model = False
            logger.info("Using general model optimizations")
    
    def _optimize_device_dtype(self, device: str, torch_dtype: str):
        """Optimize device and dtype based on model type"""
        import torch
        
        # Device optimization
        if device == "auto":
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                if self.is_code_model and gpu_memory < 16:  # Large code models need more memory
                    logger.warning(f"GPU memory ({gpu_memory:.1f}GB) may be insufficient for large code models")
                device = "cuda"
            else:
                device = "cpu"
                logger.info("CUDA not available, using CPU")
        
        # Dtype optimization based on model type and device
        if torch_dtype == "auto":
            if device == "cpu":
                torch_dtype = "float32"  # CPU works better with float32
                logger.info("Using float32 for CPU")
            elif self.is_code_model:
                torch_dtype = "float16"  # Code models can use float16 for efficiency
                logger.info("Using float16 for code model on GPU")
            else:
                torch_dtype = "float16"  # General models use float16 by default
                logger.info("Using float16 for general model on GPU")
        
        return device, torch_dtype
    
    def _get_model_specific_config(self):
        """Get model-specific configuration for generation"""
        if self.model_type == "codellama":
            return {
                "max_new_tokens_default": 1024,
                "temperature_default": 0.1,  # Lower temperature for code
                "top_p_default": 0.9,
                "repetition_penalty": 1.05,
                "do_sample": True,
                "use_cache": True
            }
        elif self.model_type == "starcoder":
            return {
                "max_new_tokens_default": 512,
                "temperature_default": 0.2,
                "top_p_default": 0.95,
                "repetition_penalty": 1.1,
                "do_sample": True,
                "use_cache": True
            }
        elif self.model_type == "deepseek":
            return {
                "max_new_tokens_default": 1024,  # DeepSeek can handle longer sequences
                "temperature_default": 0.1,
                "top_p_default": 0.9,
                "repetition_penalty": 1.02,
                "do_sample": True,
                "use_cache": True
            }
        else:
            return {
                "max_new_tokens_default": 512,
                "temperature_default": 0.7,
                "top_p_default": 1.0,
                "repetition_penalty": 1.1,
                "do_sample": True,
                "use_cache": True
            }
    
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        import torch
        
        try:
            # Get model-specific configuration
            model_config = self._get_model_specific_config()
            
            # Convert messages to prompt using model-specific formatting
            prompt = self._convert_messages_to_prompt(request.messages)
            
            logger.debug(f"Original prompt: {prompt[:200]}...")
            
            # Tokenize input with model-specific settings
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
                add_special_tokens=True,
                truncation=True,
                max_length=4096 if self.is_code_model else 2048  # Code models need longer context
            )
            inputs = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            
            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = inputs.to("cuda")
                attention_mask = attention_mask.to("cuda")

            # Generation parameters with model-specific optimizations
            max_new_tokens = min(
                request.max_tokens or model_config["max_new_tokens_default"], 
                2048 if self.is_code_model else 1024
            )
            temperature = request.temperature or model_config["temperature_default"]
            top_p = request.top_p or model_config["top_p_default"]
            
            logger.info(f"Generation params - Model: {self.model_type}, Max tokens: {max_new_tokens}, Temp: {temperature}, Top-p: {top_p}")
            
            # Generate response with model-optimized parameters
            with torch.no_grad():
                generation_kwargs = {
                    "input_ids": inputs,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "repetition_penalty": model_config["repetition_penalty"],
                    "do_sample": model_config["do_sample"],
                    "use_cache": model_config["use_cache"]
                }
                
                # Add sampling parameters only if do_sample is True
                if model_config["do_sample"]:
                    generation_kwargs.update({
                        "temperature": temperature,
                        "top_p": top_p
                    })
                
                # Model-specific generation strategies
                if self.model_type in ["codellama", "starcoder", "deepseek"]:
                    # Code models: more deterministic for low temperature
                    if temperature < 0.3:
                        generation_kwargs.update({
                            "do_sample": False,
                            "num_beams": 4,
                            "early_stopping": True
                        })
                        generation_kwargs.pop("temperature", None)
                        generation_kwargs.pop("top_p", None)
                
                outputs = self.model.generate(**generation_kwargs)

            # Decode full response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (exclude the original prompt)
            response_text = self._extract_response_only(full_response, prompt)
            
            # Post-process response based on model type
            response_text = self._post_process_response(response_text)
            
            logger.debug(f"Generated response: {response_text[:200]}...")
            
            return ChatCompletionResponse(
                id=f"chatcmpl-{int(time.time())}",
                created=int(time.time()),
                model=request.model,
                choices=[ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )],
                usage=Usage(
                    prompt_tokens=inputs.shape[1],
                    completion_tokens=outputs.shape[1] - inputs.shape[1],
                    total_tokens=outputs.shape[1]
                )
            )
            
        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"HuggingFace backend error: {str(e)}")
    
    def _extract_response_only(self, full_response: str, original_prompt: str) -> str:
        """Extract only the generated response, excluding the original prompt"""
        try:
            # Method 1: Direct prompt removal (most reliable)
            if full_response.startswith(original_prompt):
                response = full_response[len(original_prompt):].strip()
                if response:
                    return response
            
            # Method 2: Find the last Assistant marker
            assistant_markers = ["Assistant:", "assistant:", "[/INST]", "### Response:", "# Solution:"]
            for marker in assistant_markers:
                if marker in full_response:
                    parts = full_response.split(marker)
                    if len(parts) > 1:
                        response = parts[-1].strip()
                        if response and len(response) > 10:
                            return response
            
            # Method 3: Fallback cleanup
            logger.warning("Could not extract clean response, using fallback cleanup")
            return self._cleanup_response(full_response)
            
        except Exception as e:
            logger.error(f"Error extracting response: {e}")
            return "I apologize, but I encountered an issue generating a proper response."
    
    def _cleanup_response(self, text: str) -> str:
        """Clean up response text from common artifacts"""
        # Remove common prompt artifacts
        artifacts = ["Human:", "User:", "System:", "Assistant:", "AI:", "Bot:", "[INST]", "[/INST]"]
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Skip lines that are just markers
                is_artifact = any(line.startswith(artifact) and len(line) <= len(artifact) + 5 
                                for artifact in artifacts)
                if not is_artifact:
                    cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines).strip()
        return result if result else "I understand your message. How can I help you further?"
    
    def _post_process_response(self, response_text: str) -> str:
        """Post-process response based on model type"""
        if not response_text or len(response_text.strip()) < 5:
            return "I understand your message. How can I help you further?"
        
        # Code model specific post-processing
        if self.is_code_model:
            response_text = self._clean_code_response(response_text)
        
        # General cleanup
        response_text = response_text.strip()
        
        # Remove repetitive patterns
        response_text = self._remove_repetition(response_text)
        
        return response_text
    
    def _clean_code_response(self, text: str) -> str:
        """Clean code responses for better formatting"""
        import re
        
        # Ensure code blocks are properly formatted
        if text.count("```") % 2 == 1:
            text += "\n```"
        
        # Remove duplicate code block markers
        text = re.sub(r'```+', '```', text)
        
        return text
    
    def _remove_repetition(self, text: str) -> str:
        """Remove repetitive patterns in response"""
        import re
        
        # Remove repeated sentences
        sentences = text.split('. ')
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                unique_sentences.append(sentence)
                seen.add(sentence)
        
        return '. '.join(unique_sentences)
    
    def _convert_messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to a prompt format optimized for specific model types"""
        
        if self.model_type == "codellama":
            # CodeLlama instruction format
            prompt = ""
            system_msg = None
            
            for message in messages:
                if message.role == "system":
                    system_msg = message.content
                elif message.role == "user":
                    if system_msg:
                        prompt += f"[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{message.content} [/INST]"
                        system_msg = None
                    else:
                        prompt += f"[INST] {message.content} [/INST]"
                elif message.role == "assistant":
                    prompt += f" {message.content}</s><s>"
            
            return prompt
            
        elif self.model_type == "starcoder":
            # StarCoder code-focused prompts
            prompt = ""
            for message in messages:
                if message.role == "system":
                    prompt += f"# System: {message.content}\n"
                elif message.role == "user":
                    prompt += f"# User Request:\n{message.content}\n\n# Solution:\n"
                elif message.role == "assistant":
                    prompt += f"{message.content}\n\n"
            return prompt
            
        elif self.model_type == "deepseek":
            # DeepSeek-Coder format
            prompt = ""
            for message in messages:
                if message.role == "system":
                    prompt += f"You are an AI programming assistant.\n{message.content}\n\n"
                elif message.role == "user":
                    prompt += f"### Instruction:\n{message.content}\n\n### Response:\n"
                elif message.role == "assistant":
                    prompt += f"{message.content}\n\n"
            return prompt
            
        else:
            # General chat format
            prompt = ""
            for message in messages:
                if message.role == "system":
                    prompt += f"System: {message.content}\n"
                elif message.role == "user":
                    prompt += f"Human: {message.content}\n"
                elif message.role == "assistant":
                    prompt += f"Assistant: {message.content}\n"
            
            prompt += "Assistant: "
            return prompt


@app.on_event("startup")
async def startup_event():
    global llm_backend
    
    logger.info("Initializing HuggingFace backend...")
    
    try:
        llm_backend = HuggingFaceBackend()
        logger.info("HuggingFace backend initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize HuggingFace backend: {e}")
        raise e

# Health endpoints (NIM compatible)
@app.get("/v1/health/ready")
@app.get("/health")
async def health_check():
    return {
        "status": "ready" if llm_backend else "not_ready",
        "backend": "huggingface",
        "model": getattr(llm_backend, 'model_name', 'unknown') if llm_backend else 'not_loaded',
        "model_type": getattr(llm_backend, 'model_type', 'unknown') if llm_backend else 'unknown',
        "is_code_model": getattr(llm_backend, 'is_code_model', False) if llm_backend else False,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/v1/health/live")
async def liveness_check():
    return {"status": "alive"}

# Model endpoints (NIM compatible)
@app.get("/v1/models")
async def list_models():
    models = []
    if llm_backend:
        model_name = getattr(llm_backend, 'model_name', 'huggingface-model')
        models.append(Model(
            id=model_name,
            created=int(time.time()),
            owned_by="opensource"
        ))
    
    return ModelList(data=models)

# Chat completion endpoint (NIM compatible)
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if llm_backend is None:
        raise HTTPException(status_code=503, detail="Backend not initialized")
    
    logger.info(f"Chat completion request for model: {request.model}")
    return await llm_backend.chat_completion(request)

# Text completion endpoint (NIM compatible)
@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    # Convert to chat completion format
    chat_request = ChatCompletionRequest(
        model=request.model,
        messages=[ChatMessage(role="user", content=request.prompt)],
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=request.stream
    )
    
    chat_response = await chat_completions(chat_request)
    
    # Convert back to completion format
    return CompletionResponse(
        id=chat_response.id.replace("chatcmpl-", "cmpl-"),
        object="text_completion",
        created=chat_response.created,
        model=chat_response.model,
        choices=[CompletionChoice(
            index=choice.index,
            text=choice.message.content,
            finish_reason=choice.finish_reason
        ) for choice in chat_response.choices],
        usage=chat_response.usage
    )

# Root endpoint
@app.get("/")
async def root():
    model_path = getattr(llm_backend, 'model_path', 'unknown') if llm_backend else 'not_loaded'
    return {
        "service": "LLM Server - Optimized for Code Models",
        "version": "1.0.0",
        "backend": "huggingface",
        "model": getattr(llm_backend, 'model_name', 'unknown') if llm_backend else 'not_loaded',
        "model_type": getattr(llm_backend, 'model_type', 'unknown') if llm_backend else 'unknown',
        "is_code_model": getattr(llm_backend, 'is_code_model', False) if llm_backend else False,
        "model_path": model_path,
        "status": "ready" if llm_backend else "initializing",
    }

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("API_PORT", 8884)))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level
    )
