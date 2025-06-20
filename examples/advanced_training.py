#!/usr/bin/env python3
"""
Advanced training example for the NeMo Fine-Tuning Pipeline.

This example demonstrates:
1. Custom LoRA configuration
2. Multi-GPU training
3. Advanced evaluation metrics
4. Model comparison
5. Custom data processing
6. Configuration validation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.trainer import NeMoTrainer
from training.lora_config import LoRAConfig
from evaluation.evaluator import ModelEvaluator
from evaluation.metrics import MetricsCalculator
from deployment.deployer import ModelDeployer
from deployment.converter import ModelConverter
from config.config_manager import ConfigManager
from config.config_validator import ConfigValidator
from data.data_processor import DataProcessor
from data.instruction_formatter import InstructionFormatter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_custom_data():
    """Prepare and process custom training data."""
    logger.info("Preparing custom training data...")
    
    # Initialize data processor
    processor = DataProcessor("llama3")
    
    # Load and process data
    train_conversations = processor.load_yaml_data("data/sample_data.yaml")
    
    # Filter conversations
    filtered_conversations = processor.filter_conversations(
        train_conversations,
        min_length=2,  # At least 2 messages
        max_length=10  # At most 10 messages
    )
    
    # Get statistics
    stats = processor.get_statistics(filtered_conversations)
    logger.info(f"Data statistics: {stats}")
    
    # Save processed data
    processor.save_processed_data(
        filtered_conversations,
        "data/processed_train.jsonl",
        format="jsonl"
    )
    
    return filtered_conversations


def setup_custom_lora_config(model_type: str):
    """Setup custom LoRA configuration."""
    logger.info("Setting up custom LoRA configuration...")
    
    lora_config = LoRAConfig(model_type)
    lora_config.load_config()
    
    # Customize LoRA parameters
    custom_lora_settings = {
        "lora.rank": 64,  # Higher rank for better performance
        "lora.alpha": 128,  # Alpha = 2 * rank
        "lora.dropout": 0.05,  # Lower dropout
        "lora.target_modules": [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    }
    
    lora_config.update_config(custom_lora_settings)
    
    # Validate configuration
    if lora_config.validate_config():
        logger.info("LoRA configuration is valid")
    else:
        logger.error("LoRA configuration validation failed")
        return None
    
    # Save custom LoRA config
    lora_config.save_config("configs/lora/custom_lora.yaml")
    
    return lora_config


def create_advanced_config():
    """Create advanced training configuration."""
    logger.info("Creating advanced training configuration...")
    
    config_manager = ConfigManager("llama3")
    
    # Create config with custom overrides
    advanced_overrides = {
        "model": {
            "type": "llama3",
            "size": "8b",
            "base_model_path": "meta-llama/Meta-Llama-3-8B"
        },
        "training": {
            "max_epochs": 5,
            "learning_rate": 1e-4,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 4,
            "precision": "bf16"
        },
        "data": {
            "max_seq_length": 4096,
            "batch_size": 2,
            "num_workers": 4
        },
        "lora": {
            "rank": 64,
            "alpha": 128,
            "dropout": 0.05
        },
        "hardware": {
            "gpus": 2,  # Multi-GPU training
            "nodes": 1
        },
        "evaluation": {
            "eval_steps": 250,
            "save_steps": 500,
            "metrics": ["perplexity", "bleu", "rouge", "exact_match"]
        }
    }
    
    config = config_manager.create_config_from_template(
        "configs/training/advanced_config.yaml",
        overrides=advanced_overrides
    )
    
    # Validate configuration
    validator = ConfigValidator()
    if validator.validate_config(config):
        logger.info("Advanced configuration is valid")
    else:
        report = validator.get_validation_report()
        logger.error(f"Configuration validation failed: {report}")
        return None
    
    return config


def run_advanced_training():
    """Run advanced training with custom configuration."""
    logger.info("Starting advanced training...")
    
    # Prepare data
    conversations = prepare_custom_data()
    
    # Setup custom LoRA config
    lora_config = setup_custom_lora_config("llama3")
    if lora_config is None:
        return None
    
    # Create advanced config
    config = create_advanced_config()
    if config is None:
        return None
    
    # Initialize trainer with custom config
    trainer = NeMoTrainer(
        model_type="llama3",
        model_size="8b",
        config_path="configs/training/advanced_config.yaml"
    )
    
    # Setup model
    trainer.setup_model(
        base_model_path="meta-llama/Meta-Llama-3-8B"
    )
    
    # Setup data
    trainer.setup_data(
        train_file="data/sample_data.yaml",
        val_file="data/sample_data.yaml",  # Using same file for demo
        test_file="data/sample_data.yaml"
    )
    
    # Setup trainer with advanced settings
    trainer.setup_trainer(
        output_dir="checkpoints/advanced",
        log_dir="logs/advanced",
        max_epochs=5,
        gpus=2,  # Multi-GPU
        precision="bf16",
        accumulate_grad_batches=4
    )
    
    # Train model
    trainer.train()
    
    # Save model
    model_path = "checkpoints/advanced/advanced_model.nemo"
    trainer.save_model(model_path)
    
    return model_path


def run_comprehensive_evaluation(model_path: str):
    """Run comprehensive evaluation with multiple metrics."""
    logger.info("Running comprehensive evaluation...")
    
    # Initialize evaluator
    evaluator = ModelEvaluator("llama3", model_path)
    evaluator.load_model(model_path)
    evaluator.load_tokenizer(model_path)
    
    # Run evaluation
    results = evaluator.evaluate_on_file(
        test_file="data/sample_data.yaml",
        output_file="logs/advanced/comprehensive_evaluation.json",
        max_examples=50
    )
    
    # Calculate additional metrics
    metrics_calculator = MetricsCalculator("llama3")
    
    # For demonstration, create some sample predictions and targets
    sample_predictions = ["Sample prediction 1", "Sample prediction 2"]
    sample_targets = ["Sample target 1", "Sample target 2"]
    
    additional_metrics = metrics_calculator.calculate_all_metrics(
        sample_predictions,
        sample_targets
    )
    
    logger.info("Comprehensive evaluation results:")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")
    
    return results


def deploy_with_conversion(model_path: str):
    """Deploy model with format conversion."""
    logger.info("Deploying model with format conversion...")
    
    # Deploy model
    deployer = ModelDeployer("llama3", "8b", "deploy/advanced")
    deployment_path = deployer.deploy_model(
        checkpoint_path=model_path,
        model_name="advanced_llama3_model",
        copy_tokenizer=True,
        create_config=True,
        create_readme=True
    )
    
    # Convert to different formats
    converter = ModelConverter("llama3")
    
    # Create inference script
    inference_script_path = deployment_path / "inference.py"
    converter.create_inference_script(
        model_path=deployment_path,
        output_path=inference_script_path,
        model_format="nemo"
    )
    
    # Convert to HuggingFace format (placeholder)
    hf_path = deployment_path / "huggingface"
    converter.convert_nemo_to_hf(
        nemo_path=model_path,
        output_path=hf_path
    )
    
    # Quantize model (placeholder)
    quantized_path = deployment_path / "quantized_model.pt"
    converter.quantize_model(
        model_path=model_path,
        output_path=quantized_path,
        quantization_type="int8"
    )
    
    logger.info(f"Model deployed with conversions to: {deployment_path}")
    
    return deployment_path


def compare_models():
    """Compare different model configurations."""
    logger.info("Comparing model configurations...")
    
    # Compare configurations
    config_manager = ConfigManager("llama3")
    config_manager.load_config("configs/training/advanced_config.yaml")
    
    comparison = config_manager.compare_configs("configs/training/llama3_training.yaml")
    
    logger.info("Configuration comparison:")
    for path, diff in comparison["differences"].items():
        logger.info(f"  {path}: {diff['current']} vs {diff['other']}")
    
    return comparison


def main():
    """Run advanced training example."""
    logger.info("Starting advanced training example...")
    
    try:
        # Run advanced training
        model_path = run_advanced_training()
        if model_path is None:
            logger.error("Advanced training failed")
            return
        
        # Run comprehensive evaluation
        eval_results = run_comprehensive_evaluation(model_path)
        
        # Deploy with conversion
        deployment_path = deploy_with_conversion(model_path)
        
        # Compare configurations
        comparison = compare_models()
        
        logger.info("Advanced training example completed successfully!")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Deployment path: {deployment_path}")
        
    except Exception as e:
        logger.error(f"Advanced training example failed: {e}")
        raise


if __name__ == "__main__":
    main()
