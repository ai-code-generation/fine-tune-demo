"""
Model factory for creating and configuring models.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging

# Import transformers first (always available)
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Create a dummy class for type hints when transformers is not available
    class AutoTokenizer:
        pass

# Import NeMo components
try:
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
    from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    logging.warning("NeMo not available. Model creation will be limited.")

from .model_config import ModelConfig

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating and configuring models."""
    
    def __init__(self):
        """Initialize the model factory."""
        if not NEMO_AVAILABLE:
            logger.warning("NeMo not available. Some functionality will be limited.")
    
    def create_model(self, 
                    model_type: str,
                    model_size: str,
                    base_model_path: Optional[str] = None,
                    custom_config: Optional[Dict[str, Any]] = None) -> Optional[object]:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model (llama2, llama3, codellama)
            model_size: Size of model (7b, 8b, 13b, etc.)
            base_model_path: Path to base model or HuggingFace model name
            custom_config: Custom configuration overrides
            
        Returns:
            Model instance or None if NeMo not available
        """
        if not NEMO_AVAILABLE:
            logger.error("Cannot create model: NeMo not available")
            return None
        
        # Load model configuration
        model_config = ModelConfig(model_type, model_size)
        config = model_config.load_config()
        
        # Apply custom configuration if provided
        if custom_config:
            model_config.update_config(custom_config)
            config = model_config.config
        
        # Set base model path if provided
        if base_model_path:
            if 'tokenizer' not in config.model:
                config.model.tokenizer = {}
            config.model.tokenizer.type = base_model_path
        
        try:
            # Create model from config
            model = MegatronGPTModel.from_pretrained_config(config.model)
            logger.info(f"Created {model_type} {model_size} model")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            return None
    
    def setup_tokenizer(self,
                       model_type: str,
                       tokenizer_name_or_path: str) -> Optional[AutoTokenizer]:
        """
        Setup tokenizer for the model.

        Args:
            model_type: Type of model
            tokenizer_name_or_path: Tokenizer name or path

        Returns:
            Configured tokenizer
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Cannot setup tokenizer: transformers not available")
            return None

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Configure special tokens based on model type
            model_config = ModelConfig(model_type, "7b")  # Size doesn't matter for tokens
            special_tokens = model_config.get_special_tokens_config()
            
            # Set special tokens
            for token_name, token_value in special_tokens.items():
                if getattr(tokenizer, token_name, None) is None:
                    setattr(tokenizer, token_name, token_value)
            
            # Ensure pad token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"Setup tokenizer for {model_type}")
            return tokenizer
            
        except Exception as e:
            logger.error(f"Failed to setup tokenizer: {e}")
            return None
    
    def get_model_info(self, model_type: str, model_size: str) -> Dict[str, Any]:
        """
        Get information about a model configuration.
        
        Args:
            model_type: Type of model
            model_size: Size of model
            
        Returns:
            Dictionary with model information
        """
        try:
            model_config = ModelConfig(model_type, model_size)
            config = model_config.load_config()
            
            info = {
                "model_type": model_type,
                "model_size": model_size,
                "num_layers": config.model.get('num_layers', 'unknown'),
                "hidden_size": config.model.get('hidden_size', 'unknown'),
                "num_attention_heads": config.model.get('num_attention_heads', 'unknown'),
                "max_seq_length": config.model.get('encoder_seq_length', 'unknown'),
                "precision": config.model.get('precision', 'unknown'),
                "tokenizer_type": config.model.tokenizer.get('type', 'unknown'),
                "lora_enabled": 'peft' in config.model and config.model.peft.get('peft_scheme') == 'lora'
            }
            
            if info["lora_enabled"]:
                lora_config = model_config.get_lora_config()
                info["lora_rank"] = lora_config.get('adapter_dim', 'unknown')
                info["lora_alpha"] = lora_config.get('adapter_dim', 'unknown') * 2  # Default alpha
                info["lora_dropout"] = lora_config.get('adapter_dropout', 'unknown')
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}
    
    def validate_model_config(self, model_type: str, model_size: str) -> bool:
        """
        Validate model configuration.
        
        Args:
            model_type: Type of model
            model_size: Size of model
            
        Returns:
            True if configuration is valid
        """
        try:
            model_config = ModelConfig(model_type, model_size)
            config = model_config.load_config()
            
            # Check required fields
            required_fields = [
                'num_layers',
                'hidden_size',
                'num_attention_heads',
                'encoder_seq_length'
            ]
            
            for field in required_fields:
                if field not in config.model:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # Check tokenizer config
            if 'tokenizer' not in config.model:
                logger.error("Missing tokenizer configuration")
                return False
            
            logger.info(f"Configuration for {model_type} {model_size} is valid")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def list_available_models(self) -> Dict[str, list]:
        """
        List all available model configurations.
        
        Returns:
            Dictionary mapping model types to available sizes
        """
        return ModelConfig.SUPPORTED_MODELS.copy()
    
    def create_training_strategy(self, 
                               model_type: str,
                               model_size: str,
                               num_gpus: int = 1) -> Optional[object]:
        """
        Create training strategy for the model.
        
        Args:
            model_type: Type of model
            model_size: Size of model
            num_gpus: Number of GPUs to use
            
        Returns:
            Training strategy or None if NeMo not available
        """
        if not NEMO_AVAILABLE:
            logger.error("Cannot create training strategy: NeMo not available")
            return None
        
        try:
            model_config = ModelConfig(model_type, model_size)
            config = model_config.load_config()
            
            # Get model parallel configuration
            parallel_config = model_config.get_model_parallel_config()
            
            # Create strategy based on configuration
            if num_gpus > 1:
                strategy = NLPDDPStrategy(
                    no_ddp_communication_hook=True,
                    gradient_as_bucket_view=True,
                    find_unused_parameters=False,
                )
            else:
                strategy = "auto"
            
            logger.info(f"Created training strategy for {model_type} {model_size}")
            return strategy
            
        except Exception as e:
            logger.error(f"Failed to create training strategy: {e}")
            return None
