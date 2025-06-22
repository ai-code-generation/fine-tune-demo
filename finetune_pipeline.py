#!/usr/bin/env python3
"""
NeMo 24.07 Fine-tuning Pipeline
Optimized for CodeLlama-13B and Llama3 models (8B-70B)
"""

import os
import sys
import yaml
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Import our configurations
from configs.model_configs import (
    get_model_config,
    get_hardware_requirements,
    list_available_models,
    validate_hardware,
    get_recommended_config,
    get_training_config_template,
    get_nemo_24_07_optimizations
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeMo24FineTuningPipeline:
    """Fine-tuning pipeline optimized for NeMo 24.07"""
    
    def __init__(self, model_name: str, output_dir: str = "./outputs"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.model_config = get_model_config(model_name)
        
        # Create output directories
        self.models_dir = self.output_dir / "models"
        self.data_dir = self.output_dir / "data"
        self.experiments_dir = self.output_dir / "experiments"
        self.logs_dir = self.output_dir / "logs"
        
        self._create_directories()
        self._setup_cache_directories()

        logger.info(f"Initialized NeMo 24.07 pipeline for {model_name}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _create_directories(self):
        """Create necessary output directories."""
        for dir_path in [self.models_dir, self.data_dir, self.experiments_dir, self.logs_dir]:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            except PermissionError:
                logger.error(f"Permission denied creating directory: {dir_path}")
                logger.error("Make sure you have write permissions to the output directory")
                raise

    def _setup_cache_directories(self):
        """Setup cache directories with proper permissions to avoid permission errors."""
        import os

        # Define cache directories in workspace
        workspace_cache = os.path.abspath("./.cache")
        cache_subdirs = ["huggingface", "transformers", "datasets", "torch"]

        # Create cache directories
        for subdir in cache_subdirs:
            cache_dir = os.path.join(workspace_cache, subdir)
            try:
                os.makedirs(cache_dir, exist_ok=True)
                # Try to set permissions
                os.chmod(cache_dir, 0o755)
            except (OSError, PermissionError) as e:
                logger.warning(f"Could not create/set permissions for {cache_dir}: {e}")

        # Set environment variables to use workspace cache
        os.environ["TRANSFORMERS_CACHE"] = f"{workspace_cache}/transformers"
        os.environ["HF_HOME"] = f"{workspace_cache}/huggingface"
        os.environ["HF_DATASETS_CACHE"] = f"{workspace_cache}/datasets"
        os.environ["TORCH_HOME"] = f"{workspace_cache}/torch"
        os.environ["XDG_CACHE_HOME"] = workspace_cache

        # Also try to create user cache directories as fallback
        try:
            user_cache = os.path.expanduser("~/.cache")
            for subdir in cache_subdirs:
                user_cache_dir = os.path.join(user_cache, subdir)
                os.makedirs(user_cache_dir, exist_ok=True)
        except (OSError, PermissionError):
            pass  # Ignore if we can't create user cache

        logger.info(f"Cache directories configured to use: {workspace_cache}")
    
    def download_model(self, hf_token: Optional[str] = None) -> str:
        """Download the base model from Hugging Face using NeMo 24.07 methods."""
        hf_model_name = self.model_config["hf_model_name"]
        model_dir = self.models_dir / f"{self.model_name}_hf"
        
        logger.info(f"Downloading model {hf_model_name} to {model_dir}")
        
        if not hf_token:
            logger.error("Hugging Face token is required for model download")
            logger.error("Get a token from: https://huggingface.co/settings/tokens")
            logger.error("For gated models, make sure you have access approval")
            raise ValueError("Hugging Face token is required")
        
        # Use huggingface_hub for reliable downloads
        try:
            from huggingface_hub import snapshot_download
            logger.info("Using huggingface_hub for model download...")
            
            snapshot_download(
                repo_id=hf_model_name,
                local_dir=str(model_dir),
                token=hf_token,
                local_dir_use_symlinks=False
            )
            
            logger.info(f"Successfully downloaded model to {model_dir}")
            return str(model_dir)
            
        except ImportError:
            logger.error("huggingface_hub library not found. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
            
            # Try again after installation
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=hf_model_name,
                local_dir=str(model_dir),
                token=hf_token,
                local_dir_use_symlinks=False
            )
            
            logger.info(f"Successfully downloaded model to {model_dir}")
            return str(model_dir)
        
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            if "403" in str(e) or "unauthorized" in str(e).lower():
                logger.error("Access denied. Please check:")
                logger.error("1. Your HF token is valid and has read permissions")
                logger.error("2. You have access to the gated model")
                logger.error(f"3. Visit https://huggingface.co/{hf_model_name} to request access")
            raise
    
    def convert_to_nemo(self, hf_model_path: str) -> str:
        """Convert HuggingFace model to NeMo format using NeMo 24.07 tools."""
        nemo_model_path = self.models_dir / f"{self.model_name}_base.nemo"
        
        logger.info(f"Converting {hf_model_path} to NeMo format: {nemo_model_path}")
        
        # NeMo 24.07 conversion script paths
        conversion_scripts = [
            "/opt/NeMo/scripts/checkpoint_converters/convert_hf_llama_to_nemo.py",
            "/opt/NeMo/scripts/nlp_language_modeling/convert_hf_llama_to_nemo.py",
            "/workspace/NeMo/scripts/checkpoint_converters/convert_hf_llama_to_nemo.py",
        ]
        
        convert_script = None
        for script in conversion_scripts:
            if os.path.exists(script):
                convert_script = script
                logger.info(f"Found conversion script: {convert_script}")
                break
        
        if not convert_script:
            logger.warning("No conversion script found, using HuggingFace model directly")
            return hf_model_path
        
        # Build conversion command
        cmd = [
            "python", convert_script,
            f"--input_name_or_path={hf_model_path}",
            f"--output_path={nemo_model_path}",
            f"--precision=bf16",
        ]
        
        # Add model-specific parameters
        if self.model_config.get("tensor_model_parallel_size", 1) > 1:
            cmd.extend([
                f"--tensor_model_parallel_size={self.model_config['tensor_model_parallel_size']}",
                f"--pipeline_model_parallel_size={self.model_config.get('pipeline_model_parallel_size', 1)}",
            ])
        
        try:
            logger.info(f"Running conversion: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=str(self.models_dir))
            logger.info(f"Successfully converted model to {nemo_model_path}")
            return str(nemo_model_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"Model conversion failed: {e}")
            logger.warning("Using HuggingFace model directly")
            return hf_model_path
    
    def prepare_data(self, yaml_data_path: str, validation_split: float = 0.1) -> Tuple[str, str]:
        """Prepare training data from YAML format."""
        logger.info(f"Preparing data from {yaml_data_path}")
        
        # Import data preprocessing
        from data_preprocessing import convert_yaml_to_jsonl
        
        output_base = self.data_dir / f"{self.model_name}_training_data"
        train_file, val_file = convert_yaml_to_jsonl(
            yaml_data_path, 
            str(output_base), 
            validation_split
        )
        
        logger.info(f"Training data: {train_file}")
        logger.info(f"Validation data: {val_file}")
        
        return train_file, val_file
    
    def create_config_file(self, train_file: str, val_file: str, nemo_model_path: str) -> str:
        """Create the training configuration file optimized for NeMo 24.07."""
        
        # Get the appropriate template
        config_template_path = get_training_config_template(self.model_name)
        
        if not os.path.exists(config_template_path):
            logger.warning(f"Template {config_template_path} not found, creating from scratch")
            config = self._create_default_config()
        else:
            # Load template config
            with open(config_template_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Update with model-specific settings
        self._update_config_for_model(config, train_file, val_file, nemo_model_path)
        
        # Add NeMo 24.07 optimizations
        nemo_optimizations = get_nemo_24_07_optimizations(self.model_name)
        config['model'].update(nemo_optimizations)
        
        # Save updated config
        config_file = self.experiments_dir / f"{self.model_name}_training_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Created training config: {config_file}")
        return str(config_file)
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create a default configuration for NeMo 24.07."""
        return {
            "name": f"{self.model_name}_lora_tuning",
            "trainer": {
                "devices": self.model_config["devices"],
                "num_nodes": self.model_config["num_nodes"],
                "accelerator": "gpu",
                "precision": "bf16",
                "logger": False,
                "enable_checkpointing": False,
                "use_distributed_sampler": False,
                "max_epochs": -1,
                "max_steps": 100,
                "log_every_n_steps": 10,
                "val_check_interval": 0.25,
                "limit_val_batches": 50,
                "accumulate_grad_batches": 1,
                "gradient_clip_val": 1.0,
            },
            "model": {
                "restore_from_path": None,
                "tensor_model_parallel_size": self.model_config["tensor_model_parallel_size"],
                "pipeline_model_parallel_size": self.model_config["pipeline_model_parallel_size"],
                "micro_batch_size": self.model_config["micro_batch_size"],
                "global_batch_size": self.model_config["global_batch_size"],
                "sequence_length": self.model_config.get("sequence_length", 4096),
                "tokenizer": {
                    "library": "huggingface",
                    "type": self.model_config["hf_model_name"],
                    "use_fast": True,
                },
                "peft": {
                    "peft_scheme": "lora",
                    "lora_tuning": {
                        "target_modules": ["attention_qkv", "attention_dense", "mlp_fc1", "mlp_fc2"],
                        "adapter_dim": self.model_config["adapter_dim"],
                        "adapter_dropout": 0.0,
                        "column_init_method": "xavier",
                        "row_init_method": "zero",
                    }
                },
                "optim": {
                    "name": "fused_adam",
                    "lr": self.model_config["learning_rate"],
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.95],
                    "sched": {
                        "name": "CosineAnnealing",
                        "warmup_steps": 50,
                        "constant_steps": 0,
                        "min_lr": self.model_config["learning_rate"] / 10,
                    }
                },
                "data": {
                    "train_ds": {
                        "file_names": [],
                        "global_batch_size": self.model_config["global_batch_size"],
                        "micro_batch_size": self.model_config["micro_batch_size"],
                        "shuffle": True,
                        "num_workers": 0,
                        "pin_memory": True,
                        "max_seq_length": self.model_config.get("sequence_length", 4096),
                        "min_seq_length": 1,
                        "drop_last": True,
                        "label_key": "output",
                        "add_eos": True,
                        "add_sep": False,
                        "add_bos": False,
                        "truncation_field": "input",
                        "data_impl": "jsonl",
                    },
                    "validation_ds": {
                        "file_names": [],
                        "global_batch_size": self.model_config["global_batch_size"],
                        "micro_batch_size": self.model_config["micro_batch_size"],
                        "shuffle": False,
                        "num_workers": 0,
                        "pin_memory": True,
                        "max_seq_length": self.model_config.get("sequence_length", 4096),
                        "min_seq_length": 1,
                        "drop_last": False,
                        "label_key": "output",
                        "add_eos": True,
                        "add_sep": False,
                        "add_bos": False,
                        "truncation_field": "input",
                        "data_impl": "jsonl",
                    }
                }
            }
        }
    
    def _update_config_for_model(self, config: Dict[str, Any], train_file: str, val_file: str, nemo_model_path: str):
        """Update configuration with model-specific settings."""
        
        # Set model path
        if nemo_model_path.endswith('.nemo'):
            config['model']['restore_from_path'] = nemo_model_path
        else:
            # Using HuggingFace model directly
            config['model']['restore_from_path'] = None
            config['model']['model_name_or_path'] = self.model_config['hf_model_name']
        
        # Set data files
        config['model']['data']['train_ds']['file_names'] = [train_file]
        config['model']['data']['validation_ds']['file_names'] = [val_file]
        
        # Update model architecture from our config
        model_arch_keys = [
            'num_layers', 'hidden_size', 'ffn_hidden_size', 'num_attention_heads',
            'num_query_groups', 'max_position_embeddings', 'tensor_model_parallel_size',
            'pipeline_model_parallel_size', 'micro_batch_size', 'global_batch_size'
        ]
        
        for key in model_arch_keys:
            if key in self.model_config:
                config['model'][key] = self.model_config[key]
        
        # Update trainer settings
        config['trainer']['devices'] = self.model_config['devices']
        config['trainer']['num_nodes'] = self.model_config['num_nodes']
        
        # Update tokenizer
        config['model']['tokenizer']['type'] = self.model_config['hf_model_name']
        
        # Update LoRA settings
        if 'peft' in config['model'] and 'lora_tuning' in config['model']['peft']:
            config['model']['peft']['lora_tuning']['adapter_dim'] = self.model_config['adapter_dim']
        
        # Update optimizer
        if 'optim' in config['model']:
            config['model']['optim']['lr'] = self.model_config['learning_rate']

    def run_training(self, config_file: str, max_steps: int = 100) -> str:
        """Run the LoRA fine-tuning using NeMo 24.07."""
        logger.info(f"Starting LoRA fine-tuning with config: {config_file}")

        # Update max_steps in config if specified
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        config['trainer']['max_steps'] = max_steps
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Determine training script and command
        training_scripts = [
            "/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py",
            "/workspace/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py",
        ]

        training_script = None
        for script in training_scripts:
            if os.path.exists(script):
                training_script = script
                break

        if not training_script:
            raise FileNotFoundError("NeMo training script not found. Check your NeMo 24.07 installation.")

        # Build training command
        num_gpus = self.model_config['devices']
        if num_gpus > 1:
            cmd = [
                "torchrun", f"--nproc_per_node={num_gpus}",
                training_script,
                f"--config-path={os.path.dirname(config_file)}",
                f"--config-name={os.path.basename(config_file).replace('.yaml', '')}"
            ]
        else:
            cmd = [
                "python", training_script,
                f"--config-path={os.path.dirname(config_file)}",
                f"--config-name={os.path.basename(config_file).replace('.yaml', '')}"
            ]

        try:
            logger.info(f"Running training command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=str(self.experiments_dir))
            logger.info("Training completed successfully")

            # Find the trained adapter model
            adapter_path = self.find_adapter_model()
            return adapter_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            raise

    def find_adapter_model(self) -> str:
        """Find the trained adapter model."""
        # Look for the adapter model in the experiments directory
        for root, _, files in os.walk(self.experiments_dir):
            for file in files:
                if file.endswith('.nemo') and ('peft' in file.lower() or 'lora' in file.lower()):
                    adapter_path = os.path.join(root, file)
                    logger.info(f"Found adapter model: {adapter_path}")
                    return adapter_path

        raise FileNotFoundError("Could not find trained adapter model")

    def merge_weights(self, base_model_path: str, adapter_path: str) -> str:
        """Merge base model with LoRA adapter weights using NeMo 24.07."""
        merged_model_path = self.models_dir / f"{self.model_name}_merged.nemo"

        logger.info(f"Merging weights: {base_model_path} + {adapter_path} -> {merged_model_path}")

        # NeMo 24.07 merge script paths
        merge_scripts = [
            "/opt/NeMo/scripts/nlp_language_modeling/merge_lora_weights/merge.py",
            "/workspace/NeMo/scripts/nlp_language_modeling/merge_lora_weights/merge.py",
        ]

        merge_script = None
        for script in merge_scripts:
            if os.path.exists(script):
                merge_script = script
                break

        if not merge_script:
            logger.warning("Merge script not found, adapter model will be used directly")
            return adapter_path

        cmd = [
            "python", merge_script,
            "trainer.accelerator=gpu",
            f"tensor_model_parallel_size={self.model_config['tensor_model_parallel_size']}",
            f"pipeline_model_parallel_size={self.model_config.get('pipeline_model_parallel_size', 1)}",
            f"gpt_model_file={base_model_path}",
            f"lora_model_path={adapter_path}",
            f"merged_model_path={merged_model_path}"
        ]

        try:
            logger.info(f"Running merge command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully merged weights to {merged_model_path}")
            return str(merged_model_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"Weight merging failed: {e}")
            logger.warning("Using adapter model directly")
            return adapter_path

    def run_full_pipeline(self, yaml_data_path: str, max_steps: int = 100,
                         hf_token: Optional[str] = None, validation_split: float = 0.1) -> str:
        """Run the complete fine-tuning pipeline."""
        logger.info(f"Starting NeMo 24.07 fine-tuning pipeline for {self.model_name}")

        try:
            # Step 1: Download model
            hf_model_path = self.download_model(hf_token)

            # Step 2: Convert to NeMo format (if possible)
            nemo_model_path = self.convert_to_nemo(hf_model_path)

            # Step 3: Prepare data
            train_file, val_file = self.prepare_data(yaml_data_path, validation_split)

            # Step 4: Create config
            config_file = self.create_config_file(train_file, val_file, nemo_model_path)

            # Step 5: Run training
            adapter_path = self.run_training(config_file, max_steps)

            # Step 6: Merge weights (if we have a .nemo base model)
            if nemo_model_path.endswith('.nemo'):
                final_model_path = self.merge_weights(nemo_model_path, adapter_path)
            else:
                final_model_path = adapter_path

            logger.info(f"Pipeline completed successfully! Final model: {final_model_path}")
            return final_model_path

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="NeMo 24.07 Fine-tuning Pipeline for CodeLlama and Llama3")
    parser.add_argument("--model", "-m", required=True, choices=list_available_models(),
                       help="Model to fine-tune")
    parser.add_argument("--data", "-d", required=True, help="Path to YAML training data")
    parser.add_argument("--output-dir", "-o", default="./outputs", help="Output directory")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum training steps")
    parser.add_argument("--hf-token", help="Hugging Face access token (required for gated models)")
    parser.add_argument("--validation-split", type=float, default=0.1,
                       help="Fraction of data for validation")
    parser.add_argument("--check-hardware", action="store_true",
                       help="Check hardware requirements and exit")

    args = parser.parse_args()

    # Check hardware requirements if requested
    if args.check_hardware:
        requirements = get_hardware_requirements(args.model)
        print(f"Hardware requirements for {args.model}:")
        print(f"  Minimum GPUs: {requirements['min_gpus']}")
        print(f"  Minimum GPU memory: {requirements['min_gpu_memory_gb']}GB per GPU")
        print(f"  Recommended GPUs: {requirements['recommended_gpus']}")
        print(f"  Recommended GPU memory: {requirements['recommended_gpu_memory_gb']}GB per GPU")
        print(f"  Minimum system RAM: {requirements['min_system_ram_gb']}GB")
        print(f"  Recommended system RAM: {requirements['recommended_system_ram_gb']}GB")
        return 0

    # Validate input data file
    if not os.path.exists(args.data):
        logger.error(f"Data file {args.data} does not exist")
        return 1

    # Check for HF token for gated models
    if not args.hf_token:
        hf_token = os.environ.get('HF_TOKEN')
        if not hf_token:
            logger.error("Hugging Face token is required for downloading gated models")
            logger.error("Provide it via --hf-token argument or HF_TOKEN environment variable")
            logger.error("Get a token from: https://huggingface.co/settings/tokens")
            return 1
        args.hf_token = hf_token

    try:
        # Initialize pipeline
        pipeline = NeMo24FineTuningPipeline(args.model, args.output_dir)

        # Run the complete pipeline
        final_model = pipeline.run_full_pipeline(
            args.data,
            args.max_steps,
            args.hf_token,
            args.validation_split
        )

        print(f"Fine-tuning completed successfully!")
        print(f"Final model saved to: {final_model}")
        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
