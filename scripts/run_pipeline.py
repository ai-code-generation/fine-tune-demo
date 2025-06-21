#!/usr/bin/env python3
"""
Main pipeline script for fine-tuning language models with NeMo.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Setup robust import handling
project_root = Path(__file__).parent.parent
src_path = project_root / "src"

# Ensure src is in Python path
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Also add the project root to handle different import scenarios
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def import_pipeline_modules():
    """Import pipeline modules with multiple fallback strategies."""
    try:
        # Strategy 1: Direct imports from src
        from training.trainer import NeMoTrainer
        from training.lora_config import LoRAConfig
        from models.model_config import ModelConfig
        from models.model_factory import ModelFactory
        from evaluation.evaluator import ModelEvaluator
        from deployment.deployer import ModelDeployer
        from deployment.converter import ModelConverter
        return NeMoTrainer, LoRAConfig, ModelConfig, ModelFactory, ModelEvaluator, ModelDeployer, ModelConverter
    except ImportError as e1:
        print(f"Strategy 1 failed: {e1}")
        try:
            # Strategy 2: Import from src prefix
            from src.training.trainer import NeMoTrainer
            from src.training.lora_config import LoRAConfig
            from src.models.model_config import ModelConfig
            from src.models.model_factory import ModelFactory
            from src.evaluation.evaluator import ModelEvaluator
            from src.deployment.deployer import ModelDeployer
            from src.deployment.converter import ModelConverter
            return NeMoTrainer, LoRAConfig, ModelConfig, ModelFactory, ModelEvaluator, ModelDeployer, ModelConverter
        except ImportError as e2:
            print(f"Strategy 2 failed: {e2}")
            # Strategy 3: Direct file imports
            import importlib.util

            def import_from_file(file_path, module_name):
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                return module

            try:
                # Import modules directly from files
                trainer_module = import_from_file(src_path / "training" / "trainer.py", "trainer")
                lora_module = import_from_file(src_path / "training" / "lora_config.py", "lora_config")
                model_config_module = import_from_file(src_path / "models" / "model_config.py", "model_config")
                model_factory_module = import_from_file(src_path / "models" / "model_factory.py", "model_factory")
                evaluator_module = import_from_file(src_path / "evaluation" / "evaluator.py", "evaluator")
                deployer_module = import_from_file(src_path / "deployment" / "deployer.py", "deployer")
                converter_module = import_from_file(src_path / "deployment" / "converter.py", "converter")

                return (trainer_module.NeMoTrainer, lora_module.LoRAConfig,
                       model_config_module.ModelConfig, model_factory_module.ModelFactory,
                       evaluator_module.ModelEvaluator, deployer_module.ModelDeployer,
                       converter_module.ModelConverter)
            except Exception as e3:
                print(f"Strategy 3 failed: {e3}")
                print(f"Current working directory: {os.getcwd()}")
                print(f"Python path: {sys.path}")
                print(f"Source path: {src_path}")
                print(f"Source path exists: {src_path.exists()}")
                if src_path.exists():
                    print(f"Contents of src: {list(src_path.iterdir())}")
                raise ImportError(f"All import strategies failed. Last error: {e3}")

# Import pipeline components
try:
    NeMoTrainer, LoRAConfig, ModelConfig, ModelFactory, ModelEvaluator, ModelDeployer, ModelConverter = import_pipeline_modules()
except ImportError as e:
    print(f"Failed to import pipeline modules: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class FineTunePipeline:
    """Main fine-tuning pipeline orchestrator."""
    
    def __init__(self, 
                 model_type: str,
                 model_size: str,
                 config_path: Optional[str] = None):
        """
        Initialize the fine-tuning pipeline.
        
        Args:
            model_type: Type of model (llama2, llama3, codellama)
            model_size: Size of model (7b, 8b, 13b, etc.)
            config_path: Optional path to custom configuration
        """
        self.model_type = model_type.lower()
        self.model_size = model_size.lower()
        self.config_path = config_path
        
        # Initialize components
        self.trainer = None
        self.evaluator = None
        self.deployer = None
        
        # Pipeline state
        self.trained_model_path = None
        self.evaluation_results = None
        self.deployment_path = None
        
        logger.info(f"Initialized pipeline for {self.model_type} {self.model_size}")
    
    def setup_pipeline(self, 
                      base_model_path: Optional[str] = None,
                      checkpoint_path: Optional[str] = None) -> None:
        """
        Setup the pipeline components.
        
        Args:
            base_model_path: Path to base model or HuggingFace model name
            checkpoint_path: Path to existing checkpoint to resume from
        """
        try:
            logger.info("Setting up pipeline components...")
            
            # Initialize trainer
            self.trainer = NeMoTrainer(
                model_type=self.model_type,
                model_size=self.model_size,
                config_path=self.config_path
            )
            
            # Setup model
            self.trainer.setup_model(
                base_model_path=base_model_path,
                checkpoint_path=checkpoint_path
            )
            
            # Initialize evaluator
            self.evaluator = ModelEvaluator(
                model_type=self.model_type
            )
            
            # Initialize deployer
            self.deployer = ModelDeployer(
                model_type=self.model_type,
                model_size=self.model_size
            )
            
            logger.info("Pipeline setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup pipeline: {e}")
            raise
    
    def run_training(self, 
                    train_file: str,
                    val_file: Optional[str] = None,
                    test_file: Optional[str] = None,
                    max_epochs: int = 3,
                    gpus: int = 1,
                    output_dir: str = "checkpoints") -> str:
        """
        Run the training phase.
        
        Args:
            train_file: Path to training data YAML file
            val_file: Path to validation data YAML file
            test_file: Path to test data YAML file
            max_epochs: Maximum number of epochs
            gpus: Number of GPUs to use
            output_dir: Directory to save checkpoints
            
        Returns:
            Path to the trained model
        """
        try:
            logger.info("Starting training phase...")
            
            if self.trainer is None:
                raise RuntimeError("Pipeline not setup. Call setup_pipeline() first.")
            
            # Setup data
            self.trainer.setup_data(
                train_file=train_file,
                val_file=val_file,
                test_file=test_file
            )
            
            # Setup trainer
            self.trainer.setup_trainer(
                output_dir=output_dir,
                max_epochs=max_epochs,
                gpus=gpus
            )
            
            # Start training
            start_time = time.time()
            self.trainer.train()
            training_time = time.time() - start_time
            
            # Save final model
            model_name = f"{self.model_type}_{self.model_size}_finetuned"
            model_path = Path(output_dir) / f"{model_name}.nemo"
            self.trainer.save_model(model_path)
            
            self.trained_model_path = str(model_path)
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Model saved to: {self.trained_model_path}")
            
            return self.trained_model_path
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def run_evaluation(self, 
                      test_file: str,
                      model_path: Optional[str] = None,
                      max_examples: Optional[int] = None) -> Dict[str, float]:
        """
        Run the evaluation phase.
        
        Args:
            test_file: Path to test data YAML file
            model_path: Optional path to model (uses trained model if not provided)
            max_examples: Maximum number of examples to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            logger.info("Starting evaluation phase...")
            
            if self.evaluator is None:
                raise RuntimeError("Pipeline not setup. Call setup_pipeline() first.")
            
            # Use trained model if no specific path provided
            eval_model_path = model_path or self.trained_model_path
            if not eval_model_path:
                raise RuntimeError("No model available for evaluation")
            
            # Load model for evaluation
            self.evaluator.load_model(eval_model_path)
            self.evaluator.load_tokenizer(eval_model_path)
            
            # Run evaluation
            results_file = f"logs/evaluation_results_{self.model_type}_{self.model_size}.json"
            self.evaluation_results = self.evaluator.evaluate_on_file(
                test_file=test_file,
                output_file=results_file,
                max_examples=max_examples
            )
            
            logger.info("Evaluation completed")
            logger.info(f"Results saved to: {results_file}")
            
            return self.evaluation_results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def run_deployment(self, 
                      model_path: Optional[str] = None,
                      model_name: Optional[str] = None,
                      create_inference_script: bool = True) -> str:
        """
        Run the deployment phase.
        
        Args:
            model_path: Optional path to model (uses trained model if not provided)
            model_name: Optional custom name for deployed model
            create_inference_script: Whether to create inference script
            
        Returns:
            Path to the deployed model directory
        """
        try:
            logger.info("Starting deployment phase...")
            
            if self.deployer is None:
                raise RuntimeError("Pipeline not setup. Call setup_pipeline() first.")
            
            # Use trained model if no specific path provided
            deploy_model_path = model_path or self.trained_model_path
            if not deploy_model_path:
                raise RuntimeError("No model available for deployment")
            
            # Deploy model
            self.deployment_path = self.deployer.deploy_model(
                checkpoint_path=deploy_model_path,
                model_name=model_name,
                copy_tokenizer=True,
                create_config=True,
                create_readme=True
            )
            
            # Create inference script if requested
            if create_inference_script:
                converter = ModelConverter(self.model_type)
                script_path = self.deployment_path / "inference.py"
                converter.create_inference_script(
                    model_path=self.deployment_path,
                    output_path=script_path,
                    model_format="nemo"
                )
            
            logger.info("Deployment completed")
            logger.info(f"Model deployed to: {self.deployment_path}")
            
            return str(self.deployment_path)
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise
    
    def run_full_pipeline(self, 
                         train_file: str,
                         val_file: Optional[str] = None,
                         test_file: Optional[str] = None,
                         base_model_path: Optional[str] = None,
                         max_epochs: int = 3,
                         gpus: int = 1,
                         evaluate: bool = True,
                         deploy: bool = True) -> Dict[str, Any]:
        """
        Run the complete pipeline from training to deployment.
        
        Args:
            train_file: Path to training data YAML file
            val_file: Path to validation data YAML file
            test_file: Path to test data YAML file
            base_model_path: Path to base model or HuggingFace model name
            max_epochs: Maximum number of epochs
            gpus: Number of GPUs to use
            evaluate: Whether to run evaluation
            deploy: Whether to run deployment
            
        Returns:
            Dictionary with pipeline results
        """
        try:
            logger.info("Starting full pipeline...")
            start_time = time.time()
            
            results = {
                "model_type": self.model_type,
                "model_size": self.model_size,
                "pipeline_start_time": start_time
            }
            
            # Setup pipeline
            self.setup_pipeline(base_model_path=base_model_path)
            
            # Training phase
            trained_model = self.run_training(
                train_file=train_file,
                val_file=val_file,
                test_file=test_file,
                max_epochs=max_epochs,
                gpus=gpus
            )
            results["trained_model_path"] = trained_model
            
            # Evaluation phase
            if evaluate and test_file:
                eval_results = self.run_evaluation(test_file=test_file)
                results["evaluation_results"] = eval_results
            
            # Deployment phase
            if deploy:
                deployment_path = self.run_deployment()
                results["deployment_path"] = deployment_path
            
            # Pipeline completion
            total_time = time.time() - start_time
            results["total_time_seconds"] = total_time
            results["pipeline_status"] = "completed"
            
            logger.info(f"Full pipeline completed in {total_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results["pipeline_status"] = "failed"
            results["error"] = str(e)
            raise
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration."""
        info = {
            "model_type": self.model_type,
            "model_size": self.model_size,
            "config_path": self.config_path,
            "components": {
                "trainer_initialized": self.trainer is not None,
                "evaluator_initialized": self.evaluator is not None,
                "deployer_initialized": self.deployer is not None
            },
            "state": {
                "trained_model_path": self.trained_model_path,
                "evaluation_completed": self.evaluation_results is not None,
                "deployment_completed": self.deployment_path is not None
            }
        }
        
        if self.trainer:
            info["training_info"] = self.trainer.get_training_info()
        
        return info


def main():
    """Main entry point for the pipeline script."""
    parser = argparse.ArgumentParser(description="Fine-tune language models with NeMo")
    
    # Model configuration
    parser.add_argument("--model-type", required=True, 
                       choices=["llama2", "llama3", "codellama"],
                       help="Type of model to fine-tune")
    parser.add_argument("--model-size", required=True,
                       choices=["7b", "8b", "13b", "34b", "70b"],
                       help="Size of the model")
    parser.add_argument("--config", type=str,
                       help="Path to custom configuration file")
    
    # Data files
    parser.add_argument("--train-file", required=True, type=str,
                       help="Path to training data YAML file")
    parser.add_argument("--val-file", type=str,
                       help="Path to validation data YAML file")
    parser.add_argument("--test-file", type=str,
                       help="Path to test data YAML file")
    
    # Model paths
    parser.add_argument("--base-model", type=str,
                       help="Path to base model or HuggingFace model name")
    parser.add_argument("--checkpoint", type=str,
                       help="Path to existing checkpoint to resume from")
    
    # Training parameters
    parser.add_argument("--max-epochs", type=int, default=3,
                       help="Maximum number of training epochs")
    parser.add_argument("--gpus", type=int, default=1,
                       help="Number of GPUs to use")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    
    # Pipeline control
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Skip evaluation phase")
    parser.add_argument("--skip-deployment", action="store_true",
                       help="Skip deployment phase")
    parser.add_argument("--training-only", action="store_true",
                       help="Run only training phase")
    
    # Evaluation parameters
    parser.add_argument("--max-eval-examples", type=int,
                       help="Maximum number of examples for evaluation")
    
    args = parser.parse_args()
    
    try:
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Initialize pipeline
        pipeline = FineTunePipeline(
            model_type=args.model_type,
            model_size=args.model_size,
            config_path=args.config
        )
        
        if args.training_only:
            # Run only training
            pipeline.setup_pipeline(
                base_model_path=args.base_model,
                checkpoint_path=args.checkpoint
            )
            trained_model = pipeline.run_training(
                train_file=args.train_file,
                val_file=args.val_file,
                test_file=args.test_file,
                max_epochs=args.max_epochs,
                gpus=args.gpus,
                output_dir=args.output_dir
            )
            print(f"Training completed. Model saved to: {trained_model}")
        else:
            # Run full pipeline
            results = pipeline.run_full_pipeline(
                train_file=args.train_file,
                val_file=args.val_file,
                test_file=args.test_file,
                base_model_path=args.base_model,
                max_epochs=args.max_epochs,
                gpus=args.gpus,
                evaluate=not args.skip_evaluation and args.test_file is not None,
                deploy=not args.skip_deployment
            )
            
            print("Pipeline completed successfully!")
            print(f"Results: {results}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
