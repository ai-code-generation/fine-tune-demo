#!/usr/bin/env python3
"""
Basic training example for the NeMo Fine-Tuning Pipeline.

This example demonstrates how to:
1. Load and process training data
2. Configure the model and training parameters
3. Train a model with LoRA
4. Evaluate the trained model
5. Deploy the model
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.trainer import NeMoTrainer
from evaluation.evaluator import ModelEvaluator
from deployment.deployer import ModelDeployer
from config.config_manager import ConfigManager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run basic training example."""
    
    # Configuration
    model_type = "llama2"
    model_size = "7b"
    base_model = "meta-llama/Llama-2-7b-hf"
    
    # Data files
    train_file = "data/sample_data.yaml"
    val_file = None  # Using train file for validation in this example
    test_file = "data/sample_data.yaml"
    
    logger.info("Starting basic training example...")
    
    try:
        # Step 1: Setup configuration
        logger.info("Setting up configuration...")
        config_manager = ConfigManager(model_type)
        config = config_manager.load_config()
        
        # Customize configuration for this example
        config_manager.set_model_path(base_model)
        config_manager.set_data_paths(train_file, val_file, test_file)
        config_manager.set_training_params(
            max_epochs=2,  # Short training for example
            learning_rate=2e-4,
            batch_size=1,  # Small batch for demo
            gpus=1
        )
        
        # Optimize for available hardware (assuming 16GB GPU)
        config_manager.optimize_for_hardware(gpu_memory_gb=16, num_gpus=1)
        
        # Save custom config
        custom_config_path = "configs/example_config.yaml"
        config_manager.save_config(custom_config_path)
        
        logger.info(f"Configuration saved to: {custom_config_path}")
        
        # Step 2: Initialize trainer
        logger.info("Initializing trainer...")
        trainer = NeMoTrainer(
            model_type=model_type,
            model_size=model_size,
            config_path=custom_config_path
        )
        
        # Step 3: Setup model
        logger.info("Setting up model...")
        trainer.setup_model(base_model_path=base_model)
        
        # Step 4: Setup data
        logger.info("Setting up data...")
        trainer.setup_data(
            train_file=train_file,
            val_file=val_file,
            test_file=test_file
        )
        
        # Step 5: Setup trainer
        logger.info("Setting up trainer...")
        trainer.setup_trainer(
            output_dir="checkpoints/example",
            log_dir="logs/example",
            max_epochs=2,
            gpus=1,
            precision="bf16"
        )
        
        # Step 6: Train model
        logger.info("Starting training...")
        trainer.train()
        
        # Step 7: Save model
        model_path = "checkpoints/example/basic_example_model.nemo"
        trainer.save_model(model_path)
        logger.info(f"Model saved to: {model_path}")
        
        # Step 8: Evaluate model (optional)
        if Path(test_file).exists():
            logger.info("Evaluating model...")
            evaluator = ModelEvaluator(model_type, model_path)
            
            # Load model for evaluation
            evaluator.load_model(model_path)
            evaluator.load_tokenizer(model_path)
            
            # Run evaluation
            results = evaluator.evaluate_on_file(
                test_file=test_file,
                output_file="logs/example/evaluation_results.json",
                max_examples=10  # Small number for demo
            )
            
            logger.info("Evaluation results:")
            for metric, value in results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {metric}: {value:.4f}")
        
        # Step 9: Deploy model
        logger.info("Deploying model...")
        deployer = ModelDeployer(model_type, model_size, "deploy/example")
        
        deployment_path = deployer.deploy_model(
            checkpoint_path=model_path,
            model_name="basic_example_model",
            copy_tokenizer=True,
            create_config=True,
            create_readme=True
        )
        
        logger.info(f"Model deployed to: {deployment_path}")
        
        # Step 10: Show training info
        training_info = trainer.get_training_info()
        logger.info("Training completed successfully!")
        logger.info("Training info:")
        for key, value in training_info.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("Basic training example completed successfully!")
        
    except Exception as e:
        logger.error(f"Training example failed: {e}")
        raise


if __name__ == "__main__":
    main()
