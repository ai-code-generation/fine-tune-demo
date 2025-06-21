#!/usr/bin/env python3
"""
Main training script for CodeLlama fine-tuning pipeline.
Orchestrates the complete training process with configurable parameters.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.training import CodeLlamaTrainer
from src.data_handler import validate_yaml_format, ConversationDataHandler


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune CodeLlama models with LoRA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model_configs/codellama_7b.yaml",
        help="Path to model configuration file"
    )
    
    parser.add_argument(
        "--lora-config",
        type=str,
        default="configs/lora_configs/lora_default.yaml",
        help="Path to LoRA configuration file"
    )
    
    # Data paths
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/train.yaml",
        help="Path to training data YAML file"
    )
    
    parser.add_argument(
        "--eval-data",
        type=str,
        default="data/validation.yaml",
        help="Path to evaluation data YAML file"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save fine-tuned model and outputs"
    )
    
    # Training options
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--validate-data",
        action="store_true",
        help="Validate data format before training"
    )
    
    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="Create sample training data and exit"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser.parse_args()


def validate_inputs(args):
    """Validate input arguments and files."""
    logger = logging.getLogger(__name__)
    
    # Check if configuration files exist
    if not os.path.exists(args.model_config):
        raise FileNotFoundError(f"Model config file not found: {args.model_config}")
    
    if not os.path.exists(args.lora_config):
        raise FileNotFoundError(f"LoRA config file not found: {args.lora_config}")
    
    # Check if training data exists
    if not os.path.exists(args.train_data):
        raise FileNotFoundError(f"Training data file not found: {args.train_data}")
    
    # Validate data format if requested
    if args.validate_data:
        logger.info("Validating training data format...")
        if not validate_yaml_format(args.train_data):
            raise ValueError(f"Invalid YAML format in training data: {args.train_data}")
        
        if args.eval_data and os.path.exists(args.eval_data):
            logger.info("Validating evaluation data format...")
            if not validate_yaml_format(args.eval_data):
                raise ValueError(f"Invalid YAML format in evaluation data: {args.eval_data}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Input validation completed successfully")


def create_sample_data():
    """Create sample training data."""
    logger = logging.getLogger(__name__)
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Create a dummy tokenizer for the data handler
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
    
    # Create data handler and generate sample data
    data_handler = ConversationDataHandler(tokenizer)
    
    # Create sample training data
    data_handler.create_sample_data("data/sample_train.yaml")
    
    # Create sample validation data (smaller version)
    sample_validation = [
        {
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a helpful coding assistant.'
                },
                {
                    'role': 'user',
                    'content': 'Write a simple hello world program in Python'
                },
                {
                    'role': 'assistant',
                    'content': '''```python
print("Hello, World!")
```'''
                }
            ]
        }
    ]
    
    import yaml
    with open("data/sample_validation.yaml", 'w', encoding='utf-8') as file:
        yaml.dump_all(sample_validation, file, default_flow_style=False, allow_unicode=True)
    
    logger.info("Sample data created:")
    logger.info("  - data/sample_train.yaml")
    logger.info("  - data/sample_validation.yaml")


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting CodeLlama fine-tuning pipeline")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Create sample data if requested
        if args.create_sample_data:
            create_sample_data()
            return
        
        # Validate inputs
        validate_inputs(args)
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = CodeLlamaTrainer(
            model_config_path=args.model_config,
            lora_config_path=args.lora_config,
            output_dir=args.output_dir
        )
        
        # Run training
        logger.info("Starting training process...")
        results = trainer.train(
            train_data_path=args.train_data,
            eval_data_path=args.eval_data if os.path.exists(args.eval_data) else None,
            resume_from_checkpoint=args.resume_from_checkpoint
        )
        
        # Log results
        logger.info("Training completed successfully!")
        logger.info(f"Results: {results}")
        logger.info(f"Model saved to: {args.output_dir}")
        
        # Run evaluation if eval data is available
        if args.eval_data and os.path.exists(args.eval_data):
            logger.info("Running final evaluation...")
            eval_results = trainer.evaluate(args.eval_data)
            logger.info(f"Evaluation results: {eval_results}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
