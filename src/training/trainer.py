"""
NeMo trainer for fine-tuning language models.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import torch

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
    from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
    from nemo.core.config import hydra_runner
    from omegaconf import DictConfig, OmegaConf
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    logging.warning("NeMo or PyTorch Lightning not available. Training will be limited.")

from ..models.model_config import ModelConfig
from ..models.model_factory import ModelFactory
from .lora_config import LoRAConfig
from ..data.data_processor import DataProcessor
from ..data.instruction_formatter import InstructionFormatter
from ..data.dataset_builder import DatasetBuilder

logger = logging.getLogger(__name__)


class NeMoTrainer:
    """NeMo trainer for fine-tuning language models with LoRA."""
    
    def __init__(self, 
                 model_type: str,
                 model_size: str,
                 config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the NeMo trainer.
        
        Args:
            model_type: Type of model (llama2, llama3, codellama)
            model_size: Size of model (7b, 8b, 13b, etc.)
            config_path: Optional path to custom configuration
        """
        if not NEMO_AVAILABLE:
            raise RuntimeError("NeMo and PyTorch Lightning are required for training")
        
        self.model_type = model_type.lower()
        self.model_size = model_size.lower()
        
        # Initialize configuration managers
        self.model_config = ModelConfig(model_type, model_size)
        self.lora_config = LoRAConfig(model_type)
        self.model_factory = ModelFactory()
        
        # Load configurations
        self.config = self.model_config.load_config()
        self.lora_config.load_config()
        
        # Training components
        self.model = None
        self.trainer = None
        self.data_module = None
        
        # Apply custom config if provided
        if config_path:
            self._load_custom_config(config_path)
    
    def _load_custom_config(self, config_path: Union[str, Path]) -> None:
        """Load custom configuration overrides."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Custom config not found: {config_path}")
        
        custom_config = OmegaConf.load(config_path)
        self.config = OmegaConf.merge(self.config, custom_config)
        logger.info(f"Applied custom configuration from {config_path}")
    
    def setup_model(self, 
                   base_model_path: Optional[str] = None,
                   checkpoint_path: Optional[str] = None) -> None:
        """
        Setup the model for training.
        
        Args:
            base_model_path: Path to base model or HuggingFace model name
            checkpoint_path: Path to existing checkpoint to resume from
        """
        try:
            # Update model config with LoRA settings
            lora_nemo_config = self.lora_config.create_nemo_lora_config()
            self.config.model.peft = lora_nemo_config
            
            # Set base model path if provided
            if base_model_path:
                self.config.model.tokenizer.type = base_model_path
            
            if checkpoint_path and Path(checkpoint_path).exists():
                # Resume from checkpoint
                self.model = MegatronGPTModel.restore_from(checkpoint_path)
                logger.info(f"Resumed model from checkpoint: {checkpoint_path}")
            else:
                # Create new model or load from pretrained
                if base_model_path:
                    # Load from pretrained model
                    self.model = MegatronGPTModel.from_pretrained(
                        model_name=base_model_path,
                        override_config_path=self.config
                    )
                else:
                    # Create from config
                    self.model = MegatronGPTModel(cfg=self.config.model)
                
                logger.info(f"Setup {self.model_type} {self.model_size} model")
            
        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            raise
    
    def setup_data(self, 
                  train_file: Union[str, Path],
                  val_file: Optional[Union[str, Path]] = None,
                  test_file: Optional[Union[str, Path]] = None) -> None:
        """
        Setup data for training.
        
        Args:
            train_file: Path to training data YAML file
            val_file: Path to validation data YAML file
            test_file: Path to test data YAML file
        """
        try:
            # Initialize data components
            data_processor = DataProcessor(self.model_type)
            formatter = InstructionFormatter(self.model_type)
            dataset_builder = DatasetBuilder(self.model_type)
            
            # Setup tokenizer
            tokenizer_config = self.model_config.get_tokenizer_config()
            tokenizer = dataset_builder.setup_tokenizer(tokenizer_config['type'])
            
            # Process training data
            train_conversations = data_processor.load_yaml_data(train_file)
            train_conversations = data_processor.filter_conversations(train_conversations)
            train_formatted = formatter.format_for_training(train_conversations)
            
            # Create training dataset
            max_length = self.model_config.get_sequence_length()
            train_dataset = dataset_builder.create_dataset(
                train_formatted, 
                max_length=max_length
            )
            
            # Process validation data if provided
            val_dataset = None
            if val_file and Path(val_file).exists():
                val_conversations = data_processor.load_yaml_data(val_file)
                val_conversations = data_processor.filter_conversations(val_conversations)
                val_formatted = formatter.format_for_training(val_conversations)
                val_dataset = dataset_builder.create_dataset(
                    val_formatted,
                    max_length=max_length
                )
            
            # Store datasets
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            
            # Update config with data info
            self.config.model.data.seq_length = max_length
            
            logger.info(f"Setup data: {len(train_dataset)} training examples")
            if val_dataset:
                logger.info(f"Validation examples: {len(val_dataset)}")
                
        except Exception as e:
            logger.error(f"Failed to setup data: {e}")
            raise
    
    def setup_trainer(self, 
                     output_dir: Union[str, Path] = "checkpoints",
                     log_dir: Union[str, Path] = "logs",
                     max_epochs: int = 3,
                     gpus: int = 1,
                     precision: str = "bf16",
                     accumulate_grad_batches: int = 4) -> None:
        """
        Setup PyTorch Lightning trainer.
        
        Args:
            output_dir: Directory to save checkpoints
            log_dir: Directory for logs
            max_epochs: Maximum number of epochs
            gpus: Number of GPUs to use
            precision: Training precision
            accumulate_grad_batches: Gradient accumulation steps
        """
        try:
            output_dir = Path(output_dir)
            log_dir = Path(log_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup callbacks
            callbacks = []
            
            # Checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                dirpath=output_dir,
                filename=f"{self.model_type}_{self.model_size}_{{epoch:02d}}_{{val_loss:.2f}}",
                monitor="val_loss" if self.val_dataset else "train_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
                verbose=True
            )
            callbacks.append(checkpoint_callback)
            
            # Early stopping
            if self.val_dataset:
                early_stopping = EarlyStopping(
                    monitor="val_loss",
                    patience=2,
                    mode="min",
                    verbose=True
                )
                callbacks.append(early_stopping)
            
            # Setup loggers
            loggers = []
            
            # TensorBoard logger
            tb_logger = TensorBoardLogger(
                save_dir=log_dir,
                name=f"{self.model_type}_{self.model_size}",
                version=None
            )
            loggers.append(tb_logger)
            
            # WandB logger (if available)
            try:
                import wandb
                wandb_logger = WandbLogger(
                    project=f"nemo-finetune-{self.model_type}",
                    name=f"{self.model_type}_{self.model_size}",
                    save_dir=log_dir
                )
                loggers.append(wandb_logger)
            except ImportError:
                logger.info("WandB not available, using only TensorBoard")
            
            # Setup strategy for multi-GPU training
            strategy = "auto"
            if gpus > 1:
                strategy = NLPDDPStrategy(
                    no_ddp_communication_hook=True,
                    gradient_as_bucket_view=True,
                    find_unused_parameters=False,
                )
            
            # Create trainer
            self.trainer = pl.Trainer(
                max_epochs=max_epochs,
                devices=gpus,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                strategy=strategy,
                precision=precision,
                accumulate_grad_batches=accumulate_grad_batches,
                gradient_clip_val=1.0,
                callbacks=callbacks,
                logger=loggers,
                enable_checkpointing=True,
                log_every_n_steps=10,
                val_check_interval=0.5,
                limit_val_batches=50 if self.val_dataset else 0,
                enable_progress_bar=True,
                enable_model_summary=True
            )
            
            logger.info(f"Setup trainer with {gpus} GPUs, precision: {precision}")
            
        except Exception as e:
            logger.error(f"Failed to setup trainer: {e}")
            raise
    
    def train(self) -> None:
        """Start training."""
        if self.model is None:
            raise RuntimeError("Model not setup. Call setup_model() first.")
        
        if self.trainer is None:
            raise RuntimeError("Trainer not setup. Call setup_trainer() first.")
        
        if not hasattr(self, 'train_dataset'):
            raise RuntimeError("Data not setup. Call setup_data() first.")
        
        try:
            logger.info("Starting training...")
            
            # Start training
            self.trainer.fit(
                model=self.model,
                train_dataloaders=self.train_dataset,
                val_dataloaders=self.val_dataset
            )
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self, output_path: Union[str, Path]) -> None:
        """
        Save the trained model.
        
        Args:
            output_path: Path to save the model
        """
        if self.model is None:
            raise RuntimeError("No model to save")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.model.save_to(output_path)
            logger.info(f"Saved model to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training setup."""
        info = {
            "model_type": self.model_type,
            "model_size": self.model_size,
            "lora_enabled": True,
            "lora_rank": self.lora_config.get_rank_and_alpha()[0],
            "lora_alpha": self.lora_config.get_rank_and_alpha()[1],
            "max_seq_length": self.model_config.get_sequence_length(),
            "precision": self.model_config.get_precision_config()
        }
        
        if hasattr(self, 'train_dataset'):
            info["train_dataset_size"] = len(self.train_dataset)
        
        if hasattr(self, 'val_dataset') and self.val_dataset:
            info["val_dataset_size"] = len(self.val_dataset)
        
        return info
