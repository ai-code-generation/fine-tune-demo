"""
Training implementation for CodeLlama fine-tuning with LoRA.
Handles the complete training pipeline including data loading, training, and evaluation.
"""

import os
import torch
import logging
from typing import Dict, Any, Optional, List
from transformers import Trainer, DataCollatorForLanguageModeling, PreTrainedTokenizer
from datasets import Dataset
from pathlib import Path
import json
import time

from .model_setup import ModelSetup
from .data_handler import ConversationDataHandler

logger = logging.getLogger(__name__)


class CustomDataCollator:
    """Custom data collator that handles variable sequence lengths properly."""

    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        # Extract input_ids and labels
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        # Find max length in batch
        max_length = max(len(ids) for ids in input_ids)

        # Pad to multiple of pad_to_multiple_of if specified
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        # Pad sequences
        padded_input_ids = []
        padded_labels = []
        attention_masks = []

        for ids, lbls in zip(input_ids, labels):
            # Calculate padding needed
            padding_length = max_length - len(ids)

            # Pad input_ids and labels
            padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
            padded_lbls = lbls + [-100] * padding_length  # -100 is ignored in loss calculation

            # Create attention mask
            attention_mask = [1] * len(ids) + [0] * padding_length

            padded_input_ids.append(padded_ids)
            padded_labels.append(padded_lbls)
            attention_masks.append(attention_mask)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long)
        }


class CodeLlamaTrainer:
    """Main trainer class for CodeLlama fine-tuning."""
    
    def __init__(self,
                 model_config_path: str,
                 lora_config_path: str,
                 output_dir: str = "./output",
                 merge_after_training: bool = True):
        """
        Initialize the trainer.

        Args:
            model_config_path: Path to model configuration
            lora_config_path: Path to LoRA configuration
            output_dir: Directory to save outputs
            merge_after_training: Whether to merge LoRA with base model after training
        """
        self.model_setup = ModelSetup(model_config_path, lora_config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.merge_after_training = merge_after_training

        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer."""
        logger.info("Setting up tokenizer...")
        self.tokenizer = self.model_setup.setup_tokenizer()
        
        logger.info("Setting up model...")
        self.model = self.model_setup.setup_model(self.tokenizer)
        
        logger.info("Applying LoRA...")
        self.model = self.model_setup.apply_lora(self.model)
        
    def prepare_datasets(self,
                        train_data_path: Optional[str],
                        eval_data_path: Optional[str] = None) -> Dict[str, Dataset]:
        """
        Prepare training and evaluation datasets.

        Args:
            train_data_path: Path to training YAML file (optional for evaluation-only)
            eval_data_path: Path to evaluation YAML file (optional)

        Returns:
            Dictionary containing train and eval datasets
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call setup_model_and_tokenizer() first.")

        # Get max length from model config
        max_length = self.model_setup.model_config['training']['max_length']

        # Initialize data handler
        data_handler = ConversationDataHandler(self.tokenizer, max_length)

        datasets = {}

        # Process training data if provided
        if train_data_path and os.path.exists(train_data_path):
            logger.info(f"Processing training data from {train_data_path}")
            train_dataset = data_handler.process_yaml_file(train_data_path)
            datasets["train"] = train_dataset

            # Log training data info
            logger.info(f"Training dataset size: {len(train_dataset)} samples")
        else:
            logger.info("No training data provided or file not found")
        
        # Process evaluation data if provided
        if eval_data_path and os.path.exists(eval_data_path):
            logger.info(f"Processing evaluation data from {eval_data_path}")
            eval_dataset = data_handler.process_yaml_file(eval_data_path)
            datasets["eval"] = eval_dataset

            # Log evaluation data info
            logger.info(f"Evaluation dataset size: {len(eval_dataset)} samples")
        else:
            logger.info("No evaluation data provided or file not found")
            
        return datasets
    
    def setup_trainer(self, datasets: Dict[str, Dataset]):
        """
        Setup the Hugging Face trainer.
        
        Args:
            datasets: Dictionary containing train and eval datasets
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer not initialized.")
        
        # Setup training arguments
        training_args = self.model_setup.setup_training_arguments(str(self.output_dir))
        
        # Setup custom data collator to handle variable sequence lengths
        data_collator = CustomDataCollator(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8  # For efficiency with tensor cores
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets.get("eval"),
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        logger.info("Trainer setup complete")
    
    def train(self, 
              train_data_path: str, 
              eval_data_path: Optional[str] = None,
              resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            train_data_path: Path to training YAML file
            eval_data_path: Path to evaluation YAML file (optional)
            resume_from_checkpoint: Path to checkpoint to resume from (optional)
            
        Returns:
            Training metrics and information
        """
        start_time = time.time()
        
        try:
            # Setup model and tokenizer
            self.setup_model_and_tokenizer()
            
            # Prepare datasets
            datasets = self.prepare_datasets(train_data_path, eval_data_path)

            # Log dataset information
            train_size = len(datasets.get("train", []))
            eval_size = len(datasets.get("eval", [])) if datasets.get("eval") else 0
            logger.info(f"Dataset summary: {train_size} training samples, {eval_size} evaluation samples")

            # Setup trainer
            self.setup_trainer(datasets)

            # Save and log model info
            model_info = self._save_model_info()
            self._log_model_info(model_info)
            
            # Start training
            logger.info("Starting training...")
            train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            # Save the final model
            logger.info("Saving final model...")
            self.trainer.save_model()
            self.trainer.save_state()

            # Merge LoRA with base model if requested
            merged_model_path = None
            if self.merge_after_training:
                logger.info("Merging LoRA adapter with base model...")
                merged_model_path = self._merge_model()

            # Calculate training time
            training_time = time.time() - start_time
            
            # Prepare results
            results = {
                "train_runtime": train_result.metrics.get("train_runtime", 0),
                "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
                "train_steps_per_second": train_result.metrics.get("train_steps_per_second", 0),
                "total_flos": train_result.metrics.get("total_flos", 0),
                "train_loss": train_result.metrics.get("train_loss", 0),
                "total_training_time": training_time,
                "output_dir": str(self.output_dir),
                "merged_model_path": merged_model_path,
                "model_merged": self.merge_after_training
            }
            
            # Save training results
            self._save_training_results(results)

            # Log training completion summary
            self._log_training_summary(results, train_size, eval_size)

            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _save_model_info(self):
        """Save model configuration information."""
        model_info = self.model_setup.get_model_info()
        model_info["output_dir"] = str(self.output_dir)
        model_info["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        info_path = self.output_dir / "model_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2)

        logger.info(f"Model info saved to {info_path}")
        return model_info

    def _log_model_info(self, model_info: Dict[str, Any]):
        """Log model configuration information."""
        logger.info("=" * 60)
        logger.info("MODEL CONFIGURATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Model: {model_info['model_name']}")
        logger.info(f"Model Type: {model_info['model_type']}")
        logger.info(f"Device: {model_info['device_map']}")
        logger.info(f"Data Type: {model_info['torch_dtype']}")
        logger.info("")
        logger.info("Training Configuration:")
        logger.info(f"  Max Length: {model_info['max_length']}")
        logger.info(f"  Batch Size: {model_info['batch_size']}")
        logger.info(f"  Learning Rate: {model_info['learning_rate']}")
        logger.info(f"  Epochs: {model_info['num_epochs']}")
        logger.info(f"  Optimizer: {model_info['optimizer']}")
        logger.info("")
        logger.info("LoRA Configuration:")
        logger.info(f"  Rank (r): {model_info['lora_r']}")
        logger.info(f"  Alpha: {model_info['lora_alpha']}")
        logger.info(f"  Dropout: {model_info['lora_dropout']}")
        logger.info(f"  Target Modules: {model_info['target_modules']}")
        logger.info(f"  Bias: {model_info['lora_bias']}")
        logger.info("")
        logger.info("Memory Optimization:")
        logger.info(f"  Quantization: {model_info['quantization_enabled']}")
        logger.info(f"  Gradient Checkpointing: {model_info['gradient_checkpointing']}")
        logger.info(f"  FP16: {model_info['fp16']}")
        logger.info(f"  BF16: {model_info['bf16']}")
        logger.info("=" * 60)

    def _save_training_results(self, results: Dict[str, Any]):
        """Save training results to file."""
        results_path = self.output_dir / "training_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Training results saved to {results_path}")

    def _log_training_summary(self, results: Dict[str, Any], train_size: int, eval_size: int):
        """Log training completion summary."""
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Training Time: {results['total_training_time']:.2f} seconds ({results['total_training_time']/60:.1f} minutes)")
        logger.info(f"Final Train Loss: {results['train_loss']:.4f}")
        logger.info(f"Training Samples: {train_size}")
        logger.info(f"Evaluation Samples: {eval_size}")
        logger.info(f"Samples per Second: {results['train_samples_per_second']:.2f}")
        logger.info(f"Steps per Second: {results['train_steps_per_second']:.2f}")
        logger.info(f"Model saved to: {results['output_dir']}")
        if results.get('model_merged') and results.get('merged_model_path'):
            logger.info(f"Merged model saved to: {results['merged_model_path']}")
        logger.info("=" * 60)

    def _merge_model(self) -> str:
        """
        Merge LoRA adapter with base model to create a complete deployable model.

        Returns:
            Path to the merged model directory
        """
        # Create merged model directory
        merged_dir = self.output_dir / "merged_model"

        try:
            # Use the model setup's merge functionality
            merged_path = self.model_setup.merge_lora_with_base_model(
                model_path=str(self.output_dir),
                output_path=str(merged_dir)
            )

            logger.info(f"Model successfully merged and saved to: {merged_path}")
            return merged_path

        except Exception as e:
            logger.error(f"Failed to merge model: {e}")
            logger.warning("Continuing without merged model...")
            return None

    def evaluate(self, eval_data_path: str) -> Dict[str, Any]:
        """
        Evaluate the trained model.

        Args:
            eval_data_path: Path to evaluation YAML file

        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Run training first.")

        # Prepare evaluation dataset
        datasets = self.prepare_datasets(None, eval_data_path)  # No train path for evaluation
        eval_dataset = datasets.get("eval")

        if eval_dataset is None:
            raise ValueError("No evaluation dataset found")

        # Run evaluation
        logger.info("Running evaluation...")
        eval_results = self.trainer.evaluate(eval_dataset=eval_dataset)

        logger.info(f"Evaluation completed. Eval loss: {eval_results.get('eval_loss', 'N/A')}")

        return eval_results
