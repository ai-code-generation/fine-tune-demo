"""
Dataset builder for NeMo training.
"""

import json
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class ConversationDataset(Dataset):
    """PyTorch Dataset for conversation data."""
    
    def __init__(self, 
                 formatted_conversations: List[str],
                 tokenizer: AutoTokenizer,
                 max_length: int = 2048,
                 padding: str = "max_length",
                 truncation: bool = True):
        """
        Initialize the dataset.
        
        Args:
            formatted_conversations: List of formatted conversation strings
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate sequences
        """
        self.conversations = formatted_conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self) -> int:
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        conversation = self.conversations[idx]
        
        # Tokenize the conversation
        encoding = self.tokenizer(
            conversation,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt"
        )
        
        # For causal language modeling, labels are the same as input_ids
        # but shifted by one position (handled by the model)
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone()
        }
        
        return item


class DatasetBuilder:
    """Builds datasets for NeMo training."""
    
    def __init__(self, model_type: str = "llama2"):
        """
        Initialize the dataset builder.
        
        Args:
            model_type: Type of model (llama2, llama3, codellama)
        """
        self.model_type = model_type.lower()
        self.tokenizer = None
    
    def setup_tokenizer(self, tokenizer_name_or_path: str) -> AutoTokenizer:
        """
        Setup and configure the tokenizer.
        
        Args:
            tokenizer_name_or_path: Path or name of the tokenizer
            
        Returns:
            Configured tokenizer
        """
        try:
            # Check for HuggingFace authentication
            auth_kwargs = {}
            hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
            if hf_token and hf_token.strip():
                auth_kwargs['token'] = hf_token
                logger.info(f"🔑 Using HuggingFace token for tokenizer: {tokenizer_name_or_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                trust_remote_code=True,
                use_fast=True,
                **auth_kwargs
            )
            
            # Configure special tokens based on model type
            special_tokens = self._get_special_tokens()
            
            # Add special tokens if they don't exist
            tokens_to_add = []
            for token_name, token_value in special_tokens.items():
                if token_name == "additional_special_tokens":
                    for token in token_value:
                        if token not in self.tokenizer.get_vocab():
                            tokens_to_add.append(token)
                else:
                    if getattr(self.tokenizer, token_name, None) is None:
                        setattr(self.tokenizer, token_name, token_value)
            
            if tokens_to_add:
                self.tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"Tokenizer setup complete. Vocab size: {len(self.tokenizer)}")
            return self.tokenizer
            
        except Exception as e:
            raise RuntimeError(f"Failed to setup tokenizer: {e}")
    
    def _get_special_tokens(self) -> Dict[str, Union[str, List[str]]]:
        """Get special tokens for the model type."""
        if self.model_type == "llama2":
            return {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>"
            }
        elif self.model_type == "llama3":
            return {
                "bos_token": "<|begin_of_text|>",
                "eos_token": "<|end_of_text|>",
                "pad_token": "<|pad|>",
                "additional_special_tokens": [
                    "<|start_header_id|>",
                    "<|end_header_id|>",
                    "<|eot_id|>"
                ]
            }
        elif self.model_type == "codellama":
            return {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "additional_special_tokens": [
                    "<PRE>",
                    "<SUF>",
                    "<MID>"
                ]
            }
        return {}
    
    def create_dataset(self, 
                      formatted_conversations: List[str],
                      max_length: int = 2048,
                      padding: str = "max_length",
                      truncation: bool = True) -> ConversationDataset:
        """
        Create a PyTorch dataset from formatted conversations.
        
        Args:
            formatted_conversations: List of formatted conversation strings
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate sequences
            
        Returns:
            ConversationDataset instance
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not setup. Call setup_tokenizer() first.")
        
        dataset = ConversationDataset(
            formatted_conversations=formatted_conversations,
            tokenizer=self.tokenizer,
            max_length=max_length,
            padding=padding,
            truncation=truncation
        )
        
        logger.info(f"Created dataset with {len(dataset)} examples")
        return dataset
    
    def save_dataset_info(self, 
                         dataset: ConversationDataset,
                         output_path: Union[str, Path]) -> None:
        """
        Save dataset information to file.
        
        Args:
            dataset: Dataset to save info for
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Calculate some statistics
        total_tokens = 0
        max_tokens = 0
        min_tokens = float('inf')
        
        for i in range(min(100, len(dataset))):  # Sample first 100 examples
            item = dataset[i]
            num_tokens = (item["attention_mask"] == 1).sum().item()
            total_tokens += num_tokens
            max_tokens = max(max_tokens, num_tokens)
            min_tokens = min(min_tokens, num_tokens)
        
        avg_tokens = total_tokens / min(100, len(dataset))
        
        info = {
            "dataset_size": len(dataset),
            "model_type": self.model_type,
            "max_length": dataset.max_length,
            "tokenizer_vocab_size": len(dataset.tokenizer),
            "statistics": {
                "avg_tokens_per_example": avg_tokens,
                "max_tokens_per_example": max_tokens,
                "min_tokens_per_example": min_tokens if min_tokens != float('inf') else 0
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved dataset info to {output_path}")
