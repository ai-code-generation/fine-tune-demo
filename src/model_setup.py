"""
Model setup utilities for CodeLlama fine-tuning with LoRA.
Handles model loading, tokenizer setup, and LoRA configuration.
"""

import torch
import yaml
import logging
from typing import Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)


class ModelSetup:
    """Handles model and tokenizer setup for CodeLlama fine-tuning."""
    
    def __init__(self, model_config_path: str, lora_config_path: str):
        """
        Initialize model setup with configuration files.
        
        Args:
            model_config_path: Path to model configuration YAML
            lora_config_path: Path to LoRA configuration YAML
        """
        self.model_config = self._load_config(model_config_path)
        self.lora_config = self._load_config(lora_config_path)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            raise
    
    def setup_tokenizer(self) -> AutoTokenizer:
        """
        Setup and configure the tokenizer.
        
        Returns:
            Configured tokenizer
        """
        tokenizer_config = self.model_config['tokenizer']
        
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_config['name'],
            trust_remote_code=self.model_config['model'].get('trust_remote_code', True)
        )
        
        # Configure tokenizer settings
        tokenizer.padding_side = tokenizer_config.get('padding_side', 'right')
        tokenizer.truncation_side = tokenizer_config.get('truncation_side', 'right')
        
        # Add special tokens if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Add EOS and BOS tokens if specified
        if tokenizer_config.get('add_eos_token', True):
            tokenizer.add_eos_token = True
        if tokenizer_config.get('add_bos_token', True):
            tokenizer.add_bos_token = True
            
        logger.info(f"Tokenizer setup complete. Vocab size: {len(tokenizer)}")
        return tokenizer
    
    def setup_quantization_config(self) -> BitsAndBytesConfig:
        """
        Setup quantization configuration for memory efficiency.

        Returns:
            BitsAndBytesConfig for 4-bit quantization or None if not available
        """
        quant_config = self.model_config.get('quantization', {})

        if not quant_config.get('load_in_4bit', False):
            return None

        try:
            # Test if bitsandbytes is working properly
            import bitsandbytes  # noqa: F401

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, quant_config.get('bnb_4bit_compute_dtype', 'bfloat16')),
                bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', True),
                bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', 'nf4')
            )

            logger.info("4-bit quantization configuration setup complete")
            return bnb_config

        except Exception as e:
            logger.warning(f"Quantization setup failed: {e}")
            logger.warning("Falling back to full precision training")
            return None
    
    def setup_model(self, tokenizer: AutoTokenizer) -> AutoModelForCausalLM:
        """
        Setup and configure the base model.
        
        Args:
            tokenizer: Configured tokenizer
            
        Returns:
            Configured model ready for LoRA adaptation
        """
        model_config = self.model_config['model']
        
        # Setup quantization if enabled
        quantization_config = self.setup_quantization_config()
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_config['name'],
            torch_dtype=getattr(torch, model_config.get('torch_dtype', 'bfloat16')),
            device_map=model_config.get('device_map', 'auto'),
            trust_remote_code=model_config.get('trust_remote_code', True),
            quantization_config=quantization_config,
            use_cache=model_config.get('use_cache', False)
        )
        
        # Resize token embeddings if tokenizer was modified
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
            logger.info(f"Resized token embeddings to {len(tokenizer)}")
        
        # Prepare model for k-bit training if quantization is used
        if quantization_config is not None:
            try:
                model = prepare_model_for_kbit_training(model)
                logger.info("Model prepared for k-bit training")
            except Exception as e:
                logger.warning(f"Failed to prepare model for k-bit training: {e}")
                logger.warning("Continuing with standard training")
        
        # Enable gradient checkpointing for memory efficiency
        if self.model_config.get('memory_optimization', {}).get('gradient_checkpointing', True):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        logger.info(f"Model setup complete. Parameters: {model.num_parameters():,}")
        return model
    
    def setup_lora_config(self) -> LoraConfig:
        """
        Setup LoRA configuration.
        
        Returns:
            LoraConfig for PEFT
        """
        lora_params = self.lora_config['lora']
        
        lora_config = LoraConfig(
            r=lora_params['r'],
            lora_alpha=lora_params['lora_alpha'],
            lora_dropout=lora_params['lora_dropout'],
            target_modules=lora_params['target_modules'],
            bias=lora_params.get('bias', 'none'),
            task_type=lora_params.get('task_type', 'CAUSAL_LM'),
            use_rslora=lora_params.get('use_rslora', False),
            use_dora=lora_params.get('use_dora', False)
        )
        
        logger.info(f"LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}, "
                   f"dropout={lora_config.lora_dropout}")
        return lora_config
    
    def apply_lora(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """
        Apply LoRA adaptation to the model.
        
        Args:
            model: Base model
            
        Returns:
            Model with LoRA adaptation applied
        """
        lora_config = self.setup_lora_config()
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters info
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        return model
    
    def setup_training_arguments(self, output_dir: str) -> TrainingArguments:
        """
        Setup training arguments from configuration.
        
        Args:
            output_dir: Directory to save training outputs
            
        Returns:
            TrainingArguments for the trainer
        """
        training_config = self.model_config['training']
        memory_config = self.model_config.get('memory_optimization', {})
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=training_config['batch_size'],
            per_device_eval_batch_size=training_config['batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=training_config['learning_rate'],
            num_train_epochs=training_config['num_epochs'],
            warmup_steps=training_config['warmup_steps'],
            logging_steps=training_config['logging_steps'],
            save_steps=training_config['save_steps'],
            eval_steps=training_config['eval_steps'],
            save_total_limit=training_config['save_total_limit'],
            dataloader_num_workers=training_config['dataloader_num_workers'],
            remove_unused_columns=training_config['remove_unused_columns'],
            optim=training_config['optim'],
            lr_scheduler_type=training_config['lr_scheduler_type'],
            weight_decay=training_config['weight_decay'],
            max_grad_norm=training_config['max_grad_norm'],
            group_by_length=training_config['group_by_length'],
            ddp_find_unused_parameters=training_config['ddp_find_unused_parameters'],
            fp16=memory_config.get('fp16', False),
            bf16=memory_config.get('bf16', True),
            dataloader_pin_memory=memory_config.get('dataloader_pin_memory', True),
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],  # No external reporting
            run_name=f"codellama-finetune-{training_config.get('run_name', 'default')}"
        )
        
        logger.info("Training arguments setup complete")
        return training_args
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration information."""
        return {
            'model_name': self.model_config['model']['name'],
            'max_length': self.model_config['training']['max_length'],
            'lora_r': self.lora_config['lora']['r'],
            'lora_alpha': self.lora_config['lora']['lora_alpha'],
            'target_modules': self.lora_config['lora']['target_modules']
        }
