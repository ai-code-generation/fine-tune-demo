#!/usr/bin/env python3
"""
Model management script for organizing and managing fine-tuned models.
Handles copying, validation, cleanup, and model information.
"""

import argparse
import logging
import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils import (
    copy_model_to_output,
    validate_model_output,
    get_model_size,
    cleanup_checkpoints,
    create_model_card,
    load_finetuned_model
)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Manage fine-tuned CodeLlama models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Copy model command
    copy_parser = subparsers.add_parser('copy', help='Copy model to output directory')
    copy_parser.add_argument('--source', required=True, help='Source model directory')
    copy_parser.add_argument('--output', required=True, help='Output directory')
    copy_parser.add_argument('--name', help='Model name (optional)')
    
    # Validate model command
    validate_parser = subparsers.add_parser('validate', help='Validate model directory')
    validate_parser.add_argument('--model-dir', required=True, help='Model directory to validate')
    
    # Get model info command
    info_parser = subparsers.add_parser('info', help='Get model information')
    info_parser.add_argument('--model-dir', required=True, help='Model directory')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Cleanup training checkpoints')
    cleanup_parser.add_argument('--output-dir', required=True, help='Training output directory')
    cleanup_parser.add_argument('--keep-best', action='store_true', default=True, help='Keep best checkpoint')
    cleanup_parser.add_argument('--keep-last', action='store_true', default=True, help='Keep last checkpoint')
    
    # Create model card command
    card_parser = subparsers.add_parser('create-card', help='Create model card')
    card_parser.add_argument('--model-dir', required=True, help='Model directory')
    card_parser.add_argument('--training-info', help='Path to training info JSON file')
    
    # Test model command
    test_parser = subparsers.add_parser('test', help='Test model inference')
    test_parser.add_argument('--model-dir', required=True, help='Model directory')
    test_parser.add_argument('--base-model', help='Base model name (auto-detected if not provided)')
    test_parser.add_argument('--prompt', default="Write a Python function to calculate factorial", help='Test prompt')
    
    # List models command
    list_parser = subparsers.add_parser('list', help='List available models')
    list_parser.add_argument('--output-dir', required=True, help='Output directory to scan')
    
    # Global options
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    return parser.parse_args()


def copy_model_command(args):
    """Handle copy model command."""
    logger = logging.getLogger(__name__)
    
    try:
        final_path = copy_model_to_output(args.source, args.output, args.name)
        logger.info(f"Model copied successfully to: {final_path}")
        
        # Validate the copied model
        if validate_model_output(final_path):
            logger.info("Model validation passed")
        else:
            logger.warning("Model validation failed")
            
    except Exception as e:
        logger.error(f"Failed to copy model: {e}")
        sys.exit(1)


def validate_model_command(args):
    """Handle validate model command."""
    logger = logging.getLogger(__name__)
    
    if validate_model_output(args.model_dir):
        logger.info("Model validation passed")
    else:
        logger.error("Model validation failed")
        sys.exit(1)


def info_command(args):
    """Handle model info command."""
    logger = logging.getLogger(__name__)
    
    try:
        size_info = get_model_size(args.model_dir)
        
        print(f"\nModel Information for: {args.model_dir}")
        print("=" * 50)
        print(f"Total Size: {size_info['total_size_gb']:.2f} GB ({size_info['total_size_mb']:.2f} MB)")
        print(f"File Count: {size_info['file_count']}")
        print("\nFile Details:")
        
        for file_name, file_info in size_info['files'].items():
            print(f"  {file_name}: {file_info['size_mb']:.2f} MB")
        
        # Try to load model info if available
        model_info_path = Path(args.model_dir) / "model_info.json"
        if model_info_path.exists():
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            
            print("\nTraining Information:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")
                
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        sys.exit(1)


def cleanup_command(args):
    """Handle cleanup command."""
    logger = logging.getLogger(__name__)
    
    try:
        cleanup_checkpoints(args.output_dir, args.keep_best, args.keep_last)
        logger.info("Cleanup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to cleanup: {e}")
        sys.exit(1)


def create_card_command(args):
    """Handle create model card command."""
    logger = logging.getLogger(__name__)
    
    try:
        # Load training info
        training_info = {}
        
        if args.training_info and Path(args.training_info).exists():
            with open(args.training_info, 'r') as f:
                training_info = json.load(f)
        else:
            # Try to load from model directory
            model_info_path = Path(args.model_dir) / "model_info.json"
            if model_info_path.exists():
                with open(model_info_path, 'r') as f:
                    training_info = json.load(f)
        
        create_model_card(args.model_dir, training_info)
        logger.info("Model card created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create model card: {e}")
        sys.exit(1)


def test_model_command(args):
    """Handle test model command."""
    logger = logging.getLogger(__name__)
    
    try:
        from src.utils import generate_text
        
        logger.info("Loading model for testing...")
        model, tokenizer = load_finetuned_model(args.model_dir, args.base_model)
        
        logger.info(f"Testing with prompt: {args.prompt}")
        response = generate_text(model, tokenizer, args.prompt, max_length=256)
        
        print(f"\nPrompt: {args.prompt}")
        print("=" * 50)
        print(f"Response: {response}")
        
    except Exception as e:
        logger.error(f"Failed to test model: {e}")
        sys.exit(1)


def list_models_command(args):
    """Handle list models command."""
    logger = logging.getLogger(__name__)
    
    try:
        output_path = Path(args.output_dir)
        
        if not output_path.exists():
            logger.error(f"Output directory not found: {args.output_dir}")
            sys.exit(1)
        
        # Find model directories
        model_dirs = []
        for item in output_path.iterdir():
            if item.is_dir():
                # Check if it looks like a model directory
                if (item / "adapter_config.json").exists():
                    model_dirs.append(item)
        
        if not model_dirs:
            print("No models found in the output directory")
            return
        
        print(f"\nFound {len(model_dirs)} model(s) in {args.output_dir}:")
        print("=" * 60)
        
        for model_dir in sorted(model_dirs):
            print(f"\nModel: {model_dir.name}")
            
            # Get size info
            try:
                size_info = get_model_size(str(model_dir))
                print(f"  Size: {size_info['total_size_gb']:.2f} GB")
            except:
                print("  Size: Unknown")
            
            # Get model info if available
            model_info_path = model_dir / "model_info.json"
            if model_info_path.exists():
                try:
                    with open(model_info_path, 'r') as f:
                        model_info = json.load(f)
                    print(f"  Base Model: {model_info.get('model_name', 'Unknown')}")
                    print(f"  Timestamp: {model_info.get('timestamp', 'Unknown')}")
                    print(f"  LoRA Rank: {model_info.get('lora_r', 'Unknown')}")
                except:
                    pass
            
            # Check validation
            if validate_model_output(str(model_dir)):
                print("  Status: ✓ Valid")
            else:
                print("  Status: ✗ Invalid")
                
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        sys.exit(1)


def main():
    """Main function."""
    args = parse_arguments()
    
    if not args.command:
        print("Please specify a command. Use --help for available commands.")
        sys.exit(1)
    
    setup_logging(args.log_level)
    
    # Route to appropriate command handler
    if args.command == 'copy':
        copy_model_command(args)
    elif args.command == 'validate':
        validate_model_command(args)
    elif args.command == 'info':
        info_command(args)
    elif args.command == 'cleanup':
        cleanup_command(args)
    elif args.command == 'create-card':
        create_card_command(args)
    elif args.command == 'test':
        test_model_command(args)
    elif args.command == 'list':
        list_models_command(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
