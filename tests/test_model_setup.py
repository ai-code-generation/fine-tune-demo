"""
Test script for model setup functionality.
"""

import pytest
import tempfile
import yaml
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.model_setup import ModelSetup


class TestModelSetup:
    """Test cases for ModelSetup class."""
    
    @pytest.fixture
    def sample_model_config(self):
        """Sample model configuration for testing."""
        return {
            'model': {
                'name': 'gpt2',  # Use small model for testing
                'model_type': 'gpt2',
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
                'max_length': 512,
                'batch_size': 2,
                'gradient_accumulation_steps': 2,
                'learning_rate': 2e-4,
                'num_epochs': 1,
                'warmup_steps': 10,
                'logging_steps': 5,
                'save_steps': 50,
                'eval_steps': 50,
                'save_total_limit': 2,
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
                'load_in_4bit': False,
                'bnb_4bit_compute_dtype': 'float32',
                'bnb_4bit_use_double_quant': False,
                'bnb_4bit_quant_type': 'fp4'
            },
            'memory_optimization': {
                'gradient_checkpointing': False,
                'dataloader_pin_memory': False,
                'fp16': False,
                'bf16': False
            }
        }
    
    @pytest.fixture
    def sample_lora_config(self):
        """Sample LoRA configuration for testing."""
        return {
            'lora': {
                'r': 8,
                'lora_alpha': 16,
                'lora_dropout': 0.1,
                'target_modules': ['c_attn', 'c_proj'],
                'bias': 'none',
                'task_type': 'CAUSAL_LM',
                'use_rslora': False,
                'use_dora': False
            }
        }
    
    @pytest.fixture
    def temp_config_files(self, sample_model_config, sample_lora_config):
        """Create temporary configuration files."""
        # Create model config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_model_config, f)
            model_config_path = f.name
        
        # Create LoRA config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_lora_config, f)
            lora_config_path = f.name
        
        return model_config_path, lora_config_path
    
    def test_load_config(self, temp_config_files):
        """Test configuration loading."""
        model_config_path, lora_config_path = temp_config_files
        
        model_setup = ModelSetup(model_config_path, lora_config_path)
        
        assert 'model' in model_setup.model_config
        assert 'lora' in model_setup.lora_config
        assert model_setup.model_config['model']['name'] == 'gpt2'
        assert model_setup.lora_config['lora']['r'] == 8
        
        # Cleanup
        os.unlink(model_config_path)
        os.unlink(lora_config_path)
    
    def test_setup_tokenizer(self, temp_config_files):
        """Test tokenizer setup."""
        model_config_path, lora_config_path = temp_config_files
        
        model_setup = ModelSetup(model_config_path, lora_config_path)
        tokenizer = model_setup.setup_tokenizer()
        
        assert tokenizer is not None
        assert hasattr(tokenizer, 'encode')
        assert hasattr(tokenizer, 'decode')
        assert tokenizer.padding_side == 'right'
        
        # Cleanup
        os.unlink(model_config_path)
        os.unlink(lora_config_path)
    
    def test_setup_quantization_config(self, temp_config_files):
        """Test quantization configuration setup."""
        model_config_path, lora_config_path = temp_config_files
        
        model_setup = ModelSetup(model_config_path, lora_config_path)
        quant_config = model_setup.setup_quantization_config()
        
        # Should return None since load_in_4bit is False in test config
        assert quant_config is None
        
        # Cleanup
        os.unlink(model_config_path)
        os.unlink(lora_config_path)
    
    def test_setup_lora_config(self, temp_config_files):
        """Test LoRA configuration setup."""
        model_config_path, lora_config_path = temp_config_files
        
        model_setup = ModelSetup(model_config_path, lora_config_path)
        lora_config = model_setup.setup_lora_config()
        
        assert lora_config.r == 8
        assert lora_config.lora_alpha == 16
        assert lora_config.lora_dropout == 0.1
        assert 'c_attn' in lora_config.target_modules
        assert lora_config.task_type == 'CAUSAL_LM'
        
        # Cleanup
        os.unlink(model_config_path)
        os.unlink(lora_config_path)
    
    def test_setup_training_arguments(self, temp_config_files):
        """Test training arguments setup."""
        model_config_path, lora_config_path = temp_config_files
        
        model_setup = ModelSetup(model_config_path, lora_config_path)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            training_args = model_setup.setup_training_arguments(temp_dir)
            
            assert training_args.output_dir == temp_dir
            assert training_args.per_device_train_batch_size == 2
            assert training_args.learning_rate == 2e-4
            assert training_args.num_train_epochs == 1
            assert training_args.warmup_steps == 10
        
        # Cleanup
        os.unlink(model_config_path)
        os.unlink(lora_config_path)
    
    def test_get_model_info(self, temp_config_files):
        """Test model information retrieval."""
        model_config_path, lora_config_path = temp_config_files
        
        model_setup = ModelSetup(model_config_path, lora_config_path)
        model_info = model_setup.get_model_info()
        
        assert 'model_name' in model_info
        assert 'max_length' in model_info
        assert 'lora_r' in model_info
        assert 'lora_alpha' in model_info
        assert 'target_modules' in model_info
        
        assert model_info['model_name'] == 'gpt2'
        assert model_info['max_length'] == 512
        assert model_info['lora_r'] == 8
        
        # Cleanup
        os.unlink(model_config_path)
        os.unlink(lora_config_path)


def run_manual_tests():
    """Run manual tests for debugging."""
    print("Running manual tests for model setup...")
    
    try:
        # Create test configurations
        model_config = {
            'model': {
                'name': 'gpt2',
                'torch_dtype': 'float32',
                'device_map': 'cpu',
                'trust_remote_code': True,
                'use_cache': False
            },
            'tokenizer': {
                'name': 'gpt2',
                'padding_side': 'right'
            },
            'training': {
                'max_length': 256,
                'batch_size': 1,
                'gradient_accumulation_steps': 1,
                'learning_rate': 1e-4,
                'num_epochs': 1,
                'warmup_steps': 5,
                'logging_steps': 1,
                'save_steps': 10,
                'eval_steps': 10,
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
                'fp16': False,
                'bf16': False
            }
        }
        
        lora_config = {
            'lora': {
                'r': 4,
                'lora_alpha': 8,
                'lora_dropout': 0.1,
                'target_modules': ['c_attn'],
                'bias': 'none',
                'task_type': 'CAUSAL_LM'
            }
        }
        
        # Save to temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(model_config, f)
            model_config_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(lora_config, f)
            lora_config_path = f.name
        
        # Test model setup
        print("Testing model setup initialization...")
        model_setup = ModelSetup(model_config_path, lora_config_path)
        
        print("Testing tokenizer setup...")
        tokenizer = model_setup.setup_tokenizer()
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        
        print("Testing LoRA config...")
        lora_cfg = model_setup.setup_lora_config()
        print(f"LoRA rank: {lora_cfg.r}, alpha: {lora_cfg.lora_alpha}")
        
        print("Testing model info...")
        model_info = model_setup.get_model_info()
        print(f"Model info: {model_info}")
        
        # Cleanup
        os.unlink(model_config_path)
        os.unlink(lora_config_path)
        
        print("All manual tests passed!")
        
    except Exception as e:
        print(f"Manual test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run manual tests if script is executed directly
    run_manual_tests()
