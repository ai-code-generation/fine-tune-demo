# API Reference

This document provides detailed API reference for the NeMo Fine-Tuning Pipeline.

## Core Modules

### Data Processing (`src.data`)

#### DataProcessor

```python
class DataProcessor:
    """Handles loading and processing of YAML training data."""
    
    def __init__(self, model_type: str = "llama2")
    def load_yaml_data(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]
    def validate_conversation(self, conversation: Dict[str, Any]) -> bool
    def filter_conversations(self, conversations: List[Dict[str, Any]], 
                           min_length: int = 1, 
                           max_length: Optional[int] = None) -> List[Dict[str, Any]]
    def save_processed_data(self, conversations: List[Dict[str, Any]], 
                          output_path: Union[str, Path], 
                          format: str = "jsonl") -> None
    def get_statistics(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]
```

**Example:**
```python
processor = DataProcessor("llama2")
conversations = processor.load_yaml_data("data/train.yaml")
stats = processor.get_statistics(conversations)
```

#### InstructionFormatter

```python
class InstructionFormatter:
    """Formats conversations into instruction-tuning format for different models."""
    
    def __init__(self, model_type: str = "llama2")
    def format_conversation(self, conversation: Dict[str, Any]) -> str
    def format_for_training(self, conversations: List[Dict[str, Any]]) -> List[str]
    def get_special_tokens(self) -> Dict[str, str]
```

**Example:**
```python
formatter = InstructionFormatter("llama3")
formatted = formatter.format_conversation(conversation)
special_tokens = formatter.get_special_tokens()
```

#### DatasetBuilder

```python
class DatasetBuilder:
    """Builds datasets for NeMo training."""
    
    def __init__(self, model_type: str = "llama2")
    def setup_tokenizer(self, tokenizer_name_or_path: str) -> AutoTokenizer
    def create_dataset(self, formatted_conversations: List[str],
                      max_length: int = 2048,
                      padding: str = "max_length",
                      truncation: bool = True) -> ConversationDataset
    def save_dataset_info(self, dataset: ConversationDataset,
                         output_path: Union[str, Path]) -> None
```

### Model Management (`src.models`)

#### ModelConfig

```python
class ModelConfig:
    """Handles model configuration loading and management."""
    
    def __init__(self, model_type: str, model_size: str)
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> DictConfig
    def get_tokenizer_config(self) -> Dict[str, Any]
    def get_model_config(self) -> Dict[str, Any]
    def get_lora_config(self) -> Dict[str, Any]
    def get_optimizer_config(self) -> Dict[str, Any]
    def update_config(self, updates: Dict[str, Any]) -> None
    def save_config(self, output_path: Union[str, Path]) -> None
```

**Example:**
```python
model_config = ModelConfig("llama2", "7b")
config = model_config.load_config()
tokenizer_config = model_config.get_tokenizer_config()
```

#### ModelFactory

```python
class ModelFactory:
    """Factory for creating and configuring models."""
    
    def __init__(self)
    def create_model(self, model_type: str, model_size: str,
                    base_model_path: Optional[str] = None,
                    custom_config: Optional[Dict[str, Any]] = None) -> Optional[object]
    def setup_tokenizer(self, model_type: str,
                       tokenizer_name_or_path: str) -> Optional[AutoTokenizer]
    def get_model_info(self, model_type: str, model_size: str) -> Dict[str, Any]
    def validate_model_config(self, model_type: str, model_size: str) -> bool
    def list_available_models(self) -> Dict[str, list]
```

### Training (`src.training`)

#### NeMoTrainer

```python
class NeMoTrainer:
    """NeMo trainer for fine-tuning language models with LoRA."""
    
    def __init__(self, model_type: str, model_size: str,
                 config_path: Optional[Union[str, Path]] = None)
    def setup_model(self, base_model_path: Optional[str] = None,
                   checkpoint_path: Optional[str] = None) -> None
    def setup_data(self, train_file: Union[str, Path],
                  val_file: Optional[Union[str, Path]] = None,
                  test_file: Optional[Union[str, Path]] = None) -> None
    def setup_trainer(self, output_dir: Union[str, Path] = "checkpoints",
                     log_dir: Union[str, Path] = "logs",
                     max_epochs: int = 3,
                     gpus: int = 1,
                     precision: str = "bf16",
                     accumulate_grad_batches: int = 4) -> None
    def train(self) -> None
    def save_model(self, output_path: Union[str, Path]) -> None
    def get_training_info(self) -> Dict[str, Any]
```

**Example:**
```python
trainer = NeMoTrainer("llama2", "7b")
trainer.setup_model("meta-llama/Llama-2-7b-hf")
trainer.setup_data("data/train.yaml", "data/val.yaml")
trainer.setup_trainer(max_epochs=3, gpus=1)
trainer.train()
```

#### LoRAConfig

```python
class LoRAConfig:
    """Manages LoRA configuration for different model types."""
    
    def __init__(self, model_type: str)
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> DictConfig
    def get_lora_params(self) -> Dict[str, Any]
    def get_target_modules(self) -> List[str]
    def get_rank_and_alpha(self) -> tuple[int, int]
    def get_dropout(self) -> float
    def get_layer_selection(self) -> Optional[List[int]]
    def create_nemo_lora_config(self) -> Dict[str, Any]
    def get_recommended_batch_size(self, model_size: str, gpu_memory_gb: int = 24) -> int
    def validate_config(self) -> bool
```

### Evaluation (`src.evaluation`)

#### ModelEvaluator

```python
class ModelEvaluator:
    """Evaluates fine-tuned models on test data."""
    
    def __init__(self, model_type: str,
                 model_path: Optional[str] = None,
                 tokenizer_path: Optional[str] = None)
    def load_model(self, model_path: Optional[str] = None) -> None
    def load_tokenizer(self, tokenizer_path: Optional[str] = None) -> None
    def generate_predictions(self, test_data: List[Dict[str, Any]],
                           max_new_tokens: int = 512,
                           temperature: float = 0.7,
                           top_p: float = 0.9) -> List[str]
    def evaluate_on_file(self, test_file: Union[str, Path],
                        output_file: Optional[Union[str, Path]] = None,
                        max_examples: Optional[int] = None) -> Dict[str, float]
    def compare_models(self, other_results: Dict[str, float],
                      output_file: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, float]]
```

#### MetricsCalculator

```python
class MetricsCalculator:
    """Calculates various evaluation metrics for model outputs."""
    
    def __init__(self, model_type: str = "llama2")
    def calculate_perplexity(self, predictions: List[str], targets: List[str],
                           model=None) -> float
    def calculate_bleu(self, predictions: List[str], targets: List[str]) -> Dict[str, float]
    def calculate_rouge(self, predictions: List[str], targets: List[str]) -> Dict[str, float]
    def calculate_exact_match(self, predictions: List[str], targets: List[str]) -> float
    def calculate_code_metrics(self, predictions: List[str], targets: List[str]) -> Dict[str, float]
    def calculate_all_metrics(self, predictions: List[str], targets: List[str],
                             model=None) -> Dict[str, float]
```

### Deployment (`src.deployment`)

#### ModelDeployer

```python
class ModelDeployer:
    """Handles deployment of trained models to deployment directory."""
    
    def __init__(self, model_type: str, model_size: str,
                 deploy_dir: Union[str, Path] = "deploy")
    def deploy_model(self, checkpoint_path: Union[str, Path],
                    model_name: Optional[str] = None,
                    copy_tokenizer: bool = True,
                    create_config: bool = True,
                    create_readme: bool = True) -> Path
    def list_deployed_models(self) -> List[Dict[str, Any]]
    def remove_deployed_model(self, model_name: str) -> bool
```

**Example:**
```python
deployer = ModelDeployer("llama2", "7b")
deployment_path = deployer.deploy_model(
    "checkpoints/model.nemo",
    model_name="my_finetuned_model"
)
```

#### ModelConverter

```python
class ModelConverter:
    """Converts models between different formats for deployment."""
    
    def __init__(self, model_type: str)
    def convert_nemo_to_hf(self, nemo_path: Union[str, Path],
                          output_path: Union[str, Path],
                          tokenizer_path: Optional[Union[str, Path]] = None) -> bool
    def convert_to_onnx(self, model_path: Union[str, Path],
                       output_path: Union[str, Path],
                       input_shape: tuple = (1, 512)) -> bool
    def quantize_model(self, model_path: Union[str, Path],
                      output_path: Union[str, Path],
                      quantization_type: str = "int8") -> bool
    def create_inference_script(self, model_path: Union[str, Path],
                               output_path: Union[str, Path],
                               model_format: str = "nemo") -> bool
```

### Configuration Management (`src.config`)

#### ConfigManager

```python
class ConfigManager:
    """Manages configuration files and settings for the pipeline."""
    
    def __init__(self, model_type: str = "llama2")
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> DictConfig
    def create_config_from_template(self, output_path: Union[str, Path],
                                   overrides: Optional[Dict[str, Any]] = None) -> DictConfig
    def update_config(self, updates: Dict[str, Any]) -> None
    def save_config(self, output_path: Optional[Union[str, Path]] = None) -> None
    def get_model_config(self) -> Dict[str, Any]
    def get_data_config(self) -> Dict[str, Any]
    def get_training_config(self) -> Dict[str, Any]
    def set_data_paths(self, train_file: str,
                      val_file: Optional[str] = None,
                      test_file: Optional[str] = None) -> None
    def set_training_params(self, max_epochs: Optional[int] = None,
                           learning_rate: Optional[float] = None,
                           batch_size: Optional[int] = None,
                           gpus: Optional[int] = None) -> None
    def optimize_for_hardware(self, gpu_memory_gb: int, num_gpus: int = 1) -> None
    def get_config_summary(self) -> Dict[str, Any]
```

**Example:**
```python
config_manager = ConfigManager("llama2")
config = config_manager.load_config()
config_manager.set_training_params(max_epochs=5, batch_size=2)
config_manager.optimize_for_hardware(gpu_memory_gb=16, num_gpus=1)
```

#### ConfigValidator

```python
class ConfigValidator:
    """Validates configuration files and settings."""
    
    def __init__(self)
    def validate_config(self, config: Dict[str, Any]) -> bool
    def get_validation_report(self) -> Dict[str, Any]
    def validate_file_paths(self, config: Dict[str, Any],
                           base_path: Optional[Path] = None) -> bool
```

## Pipeline Orchestration

### FineTunePipeline

```python
class FineTunePipeline:
    """Main fine-tuning pipeline orchestrator."""
    
    def __init__(self, model_type: str, model_size: str,
                 config_path: Optional[str] = None)
    def setup_pipeline(self, base_model_path: Optional[str] = None,
                      checkpoint_path: Optional[str] = None) -> None
    def run_training(self, train_file: str,
                    val_file: Optional[str] = None,
                    test_file: Optional[str] = None,
                    max_epochs: int = 3,
                    gpus: int = 1,
                    output_dir: str = "checkpoints") -> str
    def run_evaluation(self, test_file: str,
                      model_path: Optional[str] = None,
                      max_examples: Optional[int] = None) -> Dict[str, float]
    def run_deployment(self, model_path: Optional[str] = None,
                      model_name: Optional[str] = None,
                      create_inference_script: bool = True) -> str
    def run_full_pipeline(self, train_file: str,
                         val_file: Optional[str] = None,
                         test_file: Optional[str] = None,
                         base_model_path: Optional[str] = None,
                         max_epochs: int = 3,
                         gpus: int = 1,
                         evaluate: bool = True,
                         deploy: bool = True) -> Dict[str, Any]
    def get_pipeline_info(self) -> Dict[str, Any]
```

**Example:**
```python
pipeline = FineTunePipeline("llama2", "7b")
results = pipeline.run_full_pipeline(
    train_file="data/train.yaml",
    val_file="data/val.yaml",
    test_file="data/test.yaml",
    base_model_path="meta-llama/Llama-2-7b-hf",
    max_epochs=3,
    gpus=1
)
```

## Error Handling

All classes include comprehensive error handling and logging. Common exceptions:

- `FileNotFoundError`: When data files or model paths don't exist
- `ValueError`: When invalid parameters are provided
- `RuntimeError`: When operations fail due to system issues
- `ImportError`: When required dependencies are not available

## Logging

The pipeline uses Python's logging module. Configure logging level:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Log files are saved to the `logs/` directory by default.

## Type Hints

All functions include comprehensive type hints for better IDE support and code clarity. Import types:

```python
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
```
