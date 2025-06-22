#!/usr/bin/env python3
"""
Standalone script for merging LoRA adapters with base models.
This script can be used to merge existing trained LoRA models with their base models
to create complete deployable models for production environments.
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add src to path for imports
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir / "src"))

from src.model_setup import ModelSetup


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def validate_paths(lora_model_path: str, output_path: str):
    """Validate input and output paths."""
    lora_path = Path(lora_model_path)

    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA model path not found: {lora_model_path}")

    # Check if it looks like a LoRA model directory
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing_files = []

    for file in required_files:
        if not (lora_path / file).exists():
            # Try alternative file names
            if file == "adapter_model.safetensors" and (lora_path / "adapter_model.bin").exists():
                continue
            missing_files.append(file)

    if missing_files:
        logging.warning(f"Some expected LoRA files not found: {missing_files}")
        logging.warning("This might not be a valid LoRA model directory")

    # Create output directory if it doesn't exist
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)


def find_config_files(lora_model_path: str):
    """
    Try to find the original configuration files used for training.

    Args:
        lora_model_path: Path to the LoRA model directory

    Returns:
        Tuple of (model_config_path, lora_config_path)
    """
    lora_path = Path(lora_model_path)

    # Look for model_info.json which contains the original configuration
    model_info_path = lora_path / "model_info.json"
    if model_info_path.exists():
        import json
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)

        # Extract model name to determine config
        model_name = model_info.get('model_name', '')

        # Try to find matching config files
        config_dir = Path(__file__).parent.parent / "configs"

        # Find model config
        model_config_path = None
        if "7b" in model_name.lower():
            model_config_path = config_dir / "model_configs" / "codellama_7b.yaml"
        elif "13b" in model_name.lower():
            model_config_path = config_dir / "model_configs" / "codellama_13b.yaml"

        # Find LoRA config based on parameters
        lora_config_path = config_dir / "lora_configs" / "lora_default.yaml"

        return str(model_config_path), str(lora_config_path)

    # Fallback to default configs
    config_dir = Path(__file__).parent.parent / "configs"
    return (
        str(config_dir / "model_configs" / "codellama_7b.yaml"),
        str(config_dir / "lora_configs" / "lora_default.yaml")
    )


def merge_model(lora_model_path: str, output_path: str,
                model_config_path: str = None, lora_config_path: str = None):
    """
    Merge LoRA adapter with base model.

    Args:
        lora_model_path: Path to the trained LoRA model
        output_path: Path where merged model should be saved
        model_config_path: Path to model configuration (optional)
        lora_config_path: Path to LoRA configuration (optional)
    """
    logger = logging.getLogger(__name__)

    # Find config files if not provided
    if not model_config_path or not lora_config_path:
        logger.info("Configuration files not provided, attempting to find them...")
        found_model_config, found_lora_config = find_config_files(lora_model_path)

        model_config_path = model_config_path or found_model_config
        lora_config_path = lora_config_path or found_lora_config

        logger.info(f"Using model config: {model_config_path}")
        logger.info(f"Using LoRA config: {lora_config_path}")

    # Validate config files exist
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Model config not found: {model_config_path}")
    if not os.path.exists(lora_config_path):
        raise FileNotFoundError(f"LoRA config not found: {lora_config_path}")

    # Initialize model setup
    logger.info("Initializing model setup...")
    model_setup = ModelSetup(model_config_path, lora_config_path)

    # Perform the merge
    logger.info("Starting model merge process...")
    merged_path = model_setup.merge_lora_with_base_model(lora_model_path, output_path)

    logger.info("=" * 60)
    logger.info("MODEL MERGE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"LoRA Model: {lora_model_path}")
    logger.info(f"Merged Model: {merged_path}")
    logger.info("=" * 60)
    logger.info("The merged model is now ready for production deployment!")

    return merged_path


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model for production deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge a LoRA model with automatic config detection
  python scripts/merge_lora_model.py --lora-model ./output/my_model --output ./merged_models/my_merged_model

  # Merge with specific config files
  python scripts/merge_lora_model.py --lora-model ./output/my_model --output ./merged_models/my_merged_model \\
    --model-config configs/model_configs/codellama_7b.yaml \\
    --lora-config configs/lora_configs/lora_default.yaml
        """
    )

    parser.add_argument(
        "--lora-model",
        type=str,
        required=True,
        help="Path to the trained LoRA model directory"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path where the merged model should be saved"
    )

    parser.add_argument(
        "--model-config",
        type=str,
        help="Path to model configuration YAML file (auto-detected if not provided)"
    )

    parser.add_argument(
        "--lora-config",
        type=str,
        help="Path to LoRA configuration YAML file (auto-detected if not provided)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Validate paths
        validate_paths(args.lora_model, args.output)

        # Perform merge
        merged_path = merge_model(
            lora_model_path=args.lora_model,
            output_path=args.output,
            model_config_path=args.model_config,
            lora_config_path=args.lora_config
        )

        logger.info(f"Merge completed successfully! Merged model saved to: {merged_path}")

    except Exception as e:
        logger.error(f"Merge failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()