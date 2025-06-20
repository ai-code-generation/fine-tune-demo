"""
Model deployment system for copying and organizing trained models.
"""

import os
import shutil
import json
import time
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import logging

try:
    from transformers import AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Some deployment features will be limited.")

logger = logging.getLogger(__name__)


class ModelDeployer:
    """Handles deployment of trained models to deployment directory."""
    
    def __init__(self, 
                 model_type: str,
                 model_size: str,
                 deploy_dir: Union[str, Path] = "deploy"):
        """
        Initialize model deployer.
        
        Args:
            model_type: Type of model (llama2, llama3, codellama)
            model_size: Size of model (7b, 8b, 13b, etc.)
            deploy_dir: Base deployment directory
        """
        self.model_type = model_type.lower()
        self.model_size = model_size.lower()
        self.deploy_dir = Path(deploy_dir)
        
        # Create deployment directory structure
        self.deploy_dir.mkdir(parents=True, exist_ok=True)
        
        # Model-specific deployment path
        self.model_deploy_dir = self.deploy_dir / f"{self.model_type}_{self.model_size}"
        
        logger.info(f"Initialized deployer for {self.model_type} {self.model_size}")
    
    def deploy_model(self, 
                    checkpoint_path: Union[str, Path],
                    model_name: Optional[str] = None,
                    copy_tokenizer: bool = True,
                    create_config: bool = True,
                    create_readme: bool = True) -> Path:
        """
        Deploy a trained model to the deployment directory.
        
        Args:
            checkpoint_path: Path to the trained model checkpoint
            model_name: Optional custom name for the deployed model
            copy_tokenizer: Whether to copy tokenizer files
            create_config: Whether to create deployment config
            create_readme: Whether to create README file
            
        Returns:
            Path to the deployed model directory
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Generate model name if not provided
        if model_name is None:
            timestamp = int(time.time())
            model_name = f"{self.model_type}_{self.model_size}_finetuned_{timestamp}"
        
        # Create deployment path
        deploy_path = self.model_deploy_dir / model_name
        deploy_path.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Deploying model to {deploy_path}")
            
            # Copy model files
            self._copy_model_files(checkpoint_path, deploy_path)
            
            # Copy tokenizer if requested
            if copy_tokenizer:
                self._copy_tokenizer_files(checkpoint_path, deploy_path)
            
            # Create deployment config
            if create_config:
                self._create_deployment_config(deploy_path, checkpoint_path, model_name)
            
            # Create README
            if create_readme:
                self._create_readme(deploy_path, model_name)
            
            # Create model info file
            self._create_model_info(deploy_path, checkpoint_path, model_name)
            
            logger.info(f"Model deployed successfully to {deploy_path}")
            return deploy_path
            
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            # Clean up on failure
            if deploy_path.exists():
                shutil.rmtree(deploy_path)
            raise
    
    def _copy_model_files(self, source_path: Path, target_path: Path) -> None:
        """Copy model files from checkpoint to deployment directory."""
        try:
            if source_path.is_file():
                # Single checkpoint file
                if source_path.suffix == '.nemo':
                    # NeMo checkpoint
                    shutil.copy2(source_path, target_path / "model.nemo")
                else:
                    # Other checkpoint format
                    shutil.copy2(source_path, target_path / source_path.name)
            else:
                # Directory with multiple files
                model_files = []
                
                # Common model file patterns
                patterns = [
                    "*.nemo",
                    "*.pt", "*.pth",
                    "*.bin", "*.safetensors",
                    "pytorch_model*.bin",
                    "model*.safetensors"
                ]
                
                for pattern in patterns:
                    model_files.extend(source_path.glob(pattern))
                
                if not model_files:
                    # Copy entire directory if no specific files found
                    shutil.copytree(source_path, target_path / "model", dirs_exist_ok=True)
                else:
                    # Copy specific model files
                    for file_path in model_files:
                        shutil.copy2(file_path, target_path / file_path.name)
            
            logger.info("Model files copied successfully")
            
        except Exception as e:
            logger.error(f"Failed to copy model files: {e}")
            raise
    
    def _copy_tokenizer_files(self, source_path: Path, target_path: Path) -> None:
        """Copy tokenizer files."""
        try:
            # Look for tokenizer files in source directory
            if source_path.is_file():
                source_dir = source_path.parent
            else:
                source_dir = source_path
            
            tokenizer_files = []
            tokenizer_patterns = [
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
                "sentencepiece.bpe.model",
                "tokenizer.model"
            ]
            
            for pattern in tokenizer_patterns:
                tokenizer_files.extend(source_dir.glob(pattern))
            
            if tokenizer_files:
                tokenizer_dir = target_path / "tokenizer"
                tokenizer_dir.mkdir(exist_ok=True)
                
                for file_path in tokenizer_files:
                    shutil.copy2(file_path, tokenizer_dir / file_path.name)
                
                logger.info(f"Copied {len(tokenizer_files)} tokenizer files")
            else:
                logger.warning("No tokenizer files found to copy")
                
        except Exception as e:
            logger.error(f"Failed to copy tokenizer files: {e}")
            # Don't raise - tokenizer copying is optional
    
    def _create_deployment_config(self, 
                                 deploy_path: Path, 
                                 checkpoint_path: Path,
                                 model_name: str) -> None:
        """Create deployment configuration file."""
        try:
            config = {
                "model_info": {
                    "name": model_name,
                    "type": self.model_type,
                    "size": self.model_size,
                    "deployment_time": time.time(),
                    "deployment_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source_checkpoint": str(checkpoint_path)
                },
                "deployment": {
                    "deploy_path": str(deploy_path),
                    "model_files": list(deploy_path.glob("*.nemo")) + list(deploy_path.glob("*.pt")) + list(deploy_path.glob("*.bin")),
                    "tokenizer_available": (deploy_path / "tokenizer").exists(),
                    "format": "nemo" if any(deploy_path.glob("*.nemo")) else "pytorch"
                },
                "usage": {
                    "loading_instructions": self._get_loading_instructions(),
                    "inference_example": self._get_inference_example()
                }
            }
            
            # Convert Path objects to strings for JSON serialization
            config["deployment"]["model_files"] = [str(f) for f in config["deployment"]["model_files"]]
            
            config_file = deploy_path / "deployment_config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info("Created deployment configuration")
            
        except Exception as e:
            logger.error(f"Failed to create deployment config: {e}")
            # Don't raise - config creation is optional
    
    def _create_readme(self, deploy_path: Path, model_name: str) -> None:
        """Create README file for the deployed model."""
        try:
            readme_content = f"""# {model_name}

## Model Information

- **Model Type**: {self.model_type.upper()}
- **Model Size**: {self.model_size.upper()}
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Deployment Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Description

This is a fine-tuned {self.model_type} {self.model_size} model trained using NVIDIA NeMo with LoRA adaptation.
The model has been optimized for instruction-following tasks.

## Files

- `model.nemo` or `*.pt`: The fine-tuned model weights
- `tokenizer/`: Tokenizer files (if available)
- `deployment_config.json`: Deployment configuration
- `model_info.json`: Detailed model information

## Usage

### Loading the Model

```python
# For NeMo models
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

model = MegatronGPTModel.restore_from("model.nemo")

# For PyTorch models
import torch
model = torch.load("model.pt")
```

### Inference Example

```python
# Example inference code
input_text = "Your instruction here"
response = model.generate(input_text, max_length=512)
print(response)
```

## Training Details

This model was fine-tuned using:
- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Instruction-tuning format optimized for {self.model_type}
- YAML-based training data format

## License

Please refer to the original model license and ensure compliance with usage terms.

## Support

For questions or issues, please refer to the NeMo documentation or the fine-tuning pipeline repository.
"""
            
            readme_file = deploy_path / "README.md"
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            logger.info("Created README file")
            
        except Exception as e:
            logger.error(f"Failed to create README: {e}")
            # Don't raise - README creation is optional
    
    def _create_model_info(self, 
                          deploy_path: Path, 
                          checkpoint_path: Path,
                          model_name: str) -> None:
        """Create detailed model information file."""
        try:
            # Get file sizes
            model_files = list(deploy_path.glob("*.nemo")) + list(deploy_path.glob("*.pt")) + list(deploy_path.glob("*.bin"))
            total_size = sum(f.stat().st_size for f in model_files if f.exists())
            
            model_info = {
                "model_name": model_name,
                "model_type": self.model_type,
                "model_size": self.model_size,
                "deployment_info": {
                    "deployment_time": time.time(),
                    "deployment_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source_checkpoint": str(checkpoint_path),
                    "deploy_path": str(deploy_path)
                },
                "file_info": {
                    "model_files": [{"name": f.name, "size_bytes": f.stat().st_size} for f in model_files],
                    "total_size_bytes": total_size,
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "tokenizer_available": (deploy_path / "tokenizer").exists()
                },
                "model_specs": self._get_model_specs(),
                "usage_notes": {
                    "recommended_gpu_memory": self._get_recommended_gpu_memory(),
                    "inference_batch_size": self._get_recommended_batch_size(),
                    "supported_tasks": self._get_supported_tasks()
                }
            }
            
            info_file = deploy_path / "model_info.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            logger.info("Created model information file")
            
        except Exception as e:
            logger.error(f"Failed to create model info: {e}")
            # Don't raise - info creation is optional
    
    def _get_loading_instructions(self) -> str:
        """Get model loading instructions."""
        if self.model_type in ["llama2", "llama3", "codellama"]:
            return "Use NeMo MegatronGPTModel.restore_from() for .nemo files or torch.load() for .pt files"
        return "Use appropriate loading method based on model format"
    
    def _get_inference_example(self) -> str:
        """Get inference example code."""
        return f"""
# Load model
model = MegatronGPTModel.restore_from("model.nemo")

# Generate response
input_text = "Your instruction here"
response = model.generate(input_text, max_length=512, temperature=0.7)
"""
    
    def _get_model_specs(self) -> Dict[str, Any]:
        """Get model specifications."""
        specs = {
            "architecture": f"{self.model_type}_{self.model_size}",
            "fine_tuning_method": "LoRA",
            "instruction_format": f"{self.model_type}_format"
        }
        
        # Add size-specific specs
        if self.model_size == "7b":
            specs.update({"parameters": "~7B", "layers": 32, "hidden_size": 4096})
        elif self.model_size == "8b":
            specs.update({"parameters": "~8B", "layers": 32, "hidden_size": 4096})
        elif self.model_size == "13b":
            specs.update({"parameters": "~13B", "layers": 40, "hidden_size": 5120})
        
        return specs
    
    def _get_recommended_gpu_memory(self) -> str:
        """Get recommended GPU memory."""
        memory_requirements = {
            "7b": "16GB+",
            "8b": "16GB+", 
            "13b": "24GB+",
            "34b": "48GB+",
            "70b": "80GB+"
        }
        return memory_requirements.get(self.model_size, "16GB+")
    
    def _get_recommended_batch_size(self) -> int:
        """Get recommended inference batch size."""
        batch_sizes = {
            "7b": 4,
            "8b": 4,
            "13b": 2,
            "34b": 1,
            "70b": 1
        }
        return batch_sizes.get(self.model_size, 1)
    
    def _get_supported_tasks(self) -> List[str]:
        """Get list of supported tasks."""
        if self.model_type == "codellama":
            return ["code_generation", "code_completion", "code_explanation", "debugging"]
        else:
            return ["instruction_following", "question_answering", "text_generation", "conversation"]
    
    def list_deployed_models(self) -> List[Dict[str, Any]]:
        """List all deployed models."""
        deployed_models = []
        
        try:
            if not self.model_deploy_dir.exists():
                return deployed_models
            
            for model_dir in self.model_deploy_dir.iterdir():
                if model_dir.is_dir():
                    info_file = model_dir / "model_info.json"
                    if info_file.exists():
                        with open(info_file, 'r', encoding='utf-8') as f:
                            model_info = json.load(f)
                        deployed_models.append(model_info)
                    else:
                        # Basic info if no info file
                        deployed_models.append({
                            "model_name": model_dir.name,
                            "deploy_path": str(model_dir),
                            "deployment_date": "unknown"
                        })
            
            return deployed_models
            
        except Exception as e:
            logger.error(f"Failed to list deployed models: {e}")
            return []
    
    def remove_deployed_model(self, model_name: str) -> bool:
        """
        Remove a deployed model.
        
        Args:
            model_name: Name of the model to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = self.model_deploy_dir / model_name
            if model_path.exists():
                shutil.rmtree(model_path)
                logger.info(f"Removed deployed model: {model_name}")
                return True
            else:
                logger.warning(f"Model not found: {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove model {model_name}: {e}")
            return False
