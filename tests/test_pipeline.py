"""
Integration test script for the complete fine-tuning pipeline.
Tests the end-to-end functionality with small models and sample data.
"""

import pytest
import tempfile
import yaml
import os
import sys
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.training import CodeLlamaTrainer
from src.data_handler import ConversationDataHandler
from src.utils import validate_model_output, get_model_size


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.fixture
    def test_configs(self):
        """Create test configurations for small-scale testing."""
        model_config = {
            'model': {
                'name': 'gpt2',  # Small model for testing
                'torch_dtype': 'float32',
                'device_map': 'cpu',
                'trust_remote_code': True,
                'use_cache': False
            },
            'tokenizer': {
                'name': 'gpt2',
                'padding_side': 'right',
                'truncation_side': 'right',
                'add_eos_token': True,
                'add_bos_token': True
            },
            'training': {
                'max_length': 128,  # Small for testing
                'batch_size': 1,
                'gradient_accumulation_steps': 1,
                'learning_rate': 1e-4,
                'num_epochs': 1,  # Just one epoch for testing
                'warmup_steps': 2,
                'logging_steps': 1,
                'save_steps': 5,
                'eval_steps': 5,
                'save_total_limit': 1,
                'dataloader_num_workers': 0,
                'remove_unused_columns': False,
                'optim': 'adamw_torch',
                'lr_scheduler_type': 'linear',
                'weight_decay': 0.01,
                'max_grad_norm': 1.0,
                'group_by_length': False,
                'ddp_find_unused_parameters': False
            },
            'quantization': {
                'load_in_4bit': False
            },
            'memory_optimization': {
                'gradient_checkpointing': False,
                'dataloader_pin_memory': False,
                'fp16': False,
                'bf16': False
            }
        }
        
        lora_config = {
            'lora': {
                'r': 2,  # Very small rank for testing
                'lora_alpha': 4,
                'lora_dropout': 0.1,
                'target_modules': ['c_attn'],  # Just one module for testing
                'bias': 'none',
                'task_type': 'CAUSAL_LM',
                'use_rslora': False,
                'use_dora': False
            }
        }
        
        return model_config, lora_config
    
    @pytest.fixture
    def test_data(self):
        """Create minimal test data."""
        return [
            {
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are helpful.'
                    },
                    {
                        'role': 'user',
                        'content': 'Say hello.'
                    },
                    {
                        'role': 'assistant',
                        'content': 'Hello there!'
                    }
                ]
            },
            {
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are helpful.'
                    },
                    {
                        'role': 'user',
                        'content': 'Count to three.'
                    },
                    {
                        'role': 'assistant',
                        'content': 'One, two, three.'
                    }
                ]
            }
        ]
    
    @pytest.fixture
    def temp_files(self, test_configs, test_data):
        """Create temporary files for testing."""
        model_config, lora_config = test_configs
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Create config files
        model_config_path = os.path.join(temp_dir, 'model_config.yaml')
        with open(model_config_path, 'w') as f:
            yaml.dump(model_config, f)
        
        lora_config_path = os.path.join(temp_dir, 'lora_config.yaml')
        with open(lora_config_path, 'w') as f:
            yaml.dump(lora_config, f)
        
        # Create data files
        train_data_path = os.path.join(temp_dir, 'train.yaml')
        with open(train_data_path, 'w') as f:
            yaml.dump_all(test_data, f)
        
        eval_data_path = os.path.join(temp_dir, 'eval.yaml')
        with open(eval_data_path, 'w') as f:
            yaml.dump_all(test_data[:1], f)  # Just one sample for eval
        
        # Create output directory
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        yield {
            'temp_dir': temp_dir,
            'model_config_path': model_config_path,
            'lora_config_path': lora_config_path,
            'train_data_path': train_data_path,
            'eval_data_path': eval_data_path,
            'output_dir': output_dir
        }
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_data_processing(self, temp_files):
        """Test data processing pipeline."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        handler = ConversationDataHandler(tokenizer, max_length=128)
        
        # Test loading and processing
        dataset = handler.process_yaml_file(temp_files['train_data_path'])
        
        assert len(dataset) == 2
        assert 'input_ids' in dataset[0]
        assert 'labels' in dataset[0]
        assert 'text' in dataset[0]
        
        # Check that text is properly formatted
        assert '[INST]' in dataset[0]['text']
        assert '[/INST]' in dataset[0]['text']
    
    def test_model_setup_and_training_preparation(self, temp_files):
        """Test model setup without actual training."""
        trainer = CodeLlamaTrainer(
            model_config_path=temp_files['model_config_path'],
            lora_config_path=temp_files['lora_config_path'],
            output_dir=temp_files['output_dir']
        )
        
        # Test model and tokenizer setup
        trainer.setup_model_and_tokenizer()
        
        assert trainer.tokenizer is not None
        assert trainer.model is not None
        
        # Test dataset preparation
        datasets = trainer.prepare_datasets(
            temp_files['train_data_path'],
            temp_files['eval_data_path']
        )
        
        assert 'train' in datasets
        assert 'eval' in datasets
        assert len(datasets['train']) == 2
        assert len(datasets['eval']) == 1
        
        # Test trainer setup
        trainer.setup_trainer(datasets)
        assert trainer.trainer is not None
    
    @pytest.mark.slow
    def test_minimal_training(self, temp_files):
        """Test minimal training run (marked as slow test)."""
        trainer = CodeLlamaTrainer(
            model_config_path=temp_files['model_config_path'],
            lora_config_path=temp_files['lora_config_path'],
            output_dir=temp_files['output_dir']
        )
        
        # Run minimal training
        results = trainer.train(
            train_data_path=temp_files['train_data_path'],
            eval_data_path=temp_files['eval_data_path']
        )
        
        # Check results
        assert 'train_loss' in results
        assert 'output_dir' in results
        assert results['output_dir'] == temp_files['output_dir']
        
        # Check that model files were created
        output_path = Path(temp_files['output_dir'])
        assert (output_path / 'adapter_config.json').exists()
        assert (output_path / 'model_info.json').exists()
        
        # Validate the model
        assert validate_model_output(temp_files['output_dir'])
        
        # Get model size info
        size_info = get_model_size(temp_files['output_dir'])
        assert size_info['total_size_bytes'] > 0
        assert size_info['file_count'] > 0


def run_manual_integration_test():
    """Run manual integration test for debugging."""
    print("Running manual integration test...")
    
    try:
        # Create test directory
        test_dir = tempfile.mkdtemp()
        print(f"Test directory: {test_dir}")
        
        # Create minimal configs
        model_config = {
            'model': {
                'name': 'gpt2',
                'torch_dtype': 'float32',
                'device_map': 'cpu',
                'use_cache': False
            },
            'tokenizer': {
                'name': 'gpt2',
                'padding_side': 'right'
            },
            'training': {
                'max_length': 64,
                'batch_size': 1,
                'gradient_accumulation_steps': 1,
                'learning_rate': 1e-4,
                'num_epochs': 1,
                'warmup_steps': 1,
                'logging_steps': 1,
                'save_steps': 2,
                'eval_steps': 2,
                'save_total_limit': 1,
                'dataloader_num_workers': 0,
                'remove_unused_columns': False,
                'optim': 'adamw_torch',
                'lr_scheduler_type': 'linear',
                'weight_decay': 0.01,
                'max_grad_norm': 1.0,
                'group_by_length': False,
                'ddp_find_unused_parameters': False
            },
            'quantization': {'load_in_4bit': False},
            'memory_optimization': {
                'gradient_checkpointing': False,
                'fp16': False,
                'bf16': False
            }
        }
        
        lora_config = {
            'lora': {
                'r': 1,
                'lora_alpha': 2,
                'lora_dropout': 0.1,
                'target_modules': ['c_attn'],
                'bias': 'none',
                'task_type': 'CAUSAL_LM'
            }
        }
        
        # Save configs
        model_config_path = os.path.join(test_dir, 'model.yaml')
        lora_config_path = os.path.join(test_dir, 'lora.yaml')
        
        with open(model_config_path, 'w') as f:
            yaml.dump(model_config, f)
        with open(lora_config_path, 'w') as f:
            yaml.dump(lora_config, f)
        
        # Create minimal data
        test_data = [
            {
                'messages': [
                    {'role': 'system', 'content': 'You are helpful.'},
                    {'role': 'user', 'content': 'Hi'},
                    {'role': 'assistant', 'content': 'Hello!'}
                ]
            }
        ]
        
        train_data_path = os.path.join(test_dir, 'train.yaml')
        with open(train_data_path, 'w') as f:
            yaml.dump_all(test_data, f)
        
        output_dir = os.path.join(test_dir, 'output')
        
        print("Testing data processing...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        handler = ConversationDataHandler(tokenizer, max_length=64)
        dataset = handler.process_yaml_file(train_data_path)
        print(f"Dataset size: {len(dataset)}")
        
        print("Testing trainer setup...")
        trainer = CodeLlamaTrainer(model_config_path, lora_config_path, output_dir)
        trainer.setup_model_and_tokenizer()
        print("Model and tokenizer setup complete")
        
        datasets = trainer.prepare_datasets(train_data_path)
        print(f"Datasets prepared: {list(datasets.keys())}")
        
        trainer.setup_trainer(datasets)
        print("Trainer setup complete")
        
        print("Manual integration test passed!")
        
        # Cleanup
        shutil.rmtree(test_dir)
        
    except Exception as e:
        print(f"Manual integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run manual test if script is executed directly
    run_manual_integration_test()
