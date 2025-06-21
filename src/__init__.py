"""
CodeLlama Fine-tuning Pipeline

A comprehensive pipeline for fine-tuning CodeLlama models using LoRA.
"""

__version__ = "1.0.0"
__author__ = "CodeLlama Fine-tuning Pipeline"

from .data_handler import ConversationDataHandler, validate_yaml_format
from .model_setup import ModelSetup
from .training import CodeLlamaTrainer
from .utils import (
    copy_model_to_output,
    load_finetuned_model,
    generate_text,
    validate_model_output,
    get_model_size
)

__all__ = [
    "ConversationDataHandler",
    "validate_yaml_format",
    "ModelSetup", 
    "CodeLlamaTrainer",
    "copy_model_to_output",
    "load_finetuned_model",
    "generate_text",
    "validate_model_output",
    "get_model_size"
]
