"""
Tests for model merging functionality.
Tests the ability to merge LoRA adapters with base models.
"""

import pytest
import tempfile
import os
import yaml
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.model_setup import ModelSetup


class TestModelMerging:
    """Test model merging functionality."""
    
    @pytest.fixture
    def temp_config_files(self):
        """Create temporary configuration files for testing."""
        model_config = {
            'model': {
                'name': 'codellama/CodeLlama-7b-Instruct-hf',
                'model_type': 'llama',
                'torch_dtype': 'bfloat16',
                'device_map': 'auto',
                'trust_remote_code': True,
                'use_cache': False
            },
            'tokenizer': {
                'name': 'codellama/CodeLlama-7b-Instruct-hf',
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
                'num_epochs': 1.0,
                'warmup_steps': 10,
                'logging_steps': 5,
                'save_steps': 50,
                'eval_steps': 50,
                'save_total_limit': 2,
                'dataloader_num_workers': 0,
                'remove_unused_columns': False,
                'optim': 'adamw_torch',
                'lr_scheduler_type': 'cosine',
                'weight_decay': 0.001,
                'max_grad_norm': 0.3,
                'group_by_length': True,
                'ddp_find_unused_parameters': False
            },
            'quantization': {
                'load_in_4bit': False,
                'bnb_4bit_compute_dtype': 'bfloat16',
                'bnb_4bit_use_double_quant': True,
                'bnb_4bit_quant_type': 'nf4'
            },
            'memory_optimization': {
                'gradient_checkpointing': False,
                'dataloader_pin_memory': True,
                'fp16': False,
                'bf16': True
            }
        }
        
        lora_config = {
            'lora': {
                'r': 4,
                'lora_alpha': 8,
                'lora_dropout': 0.1,
                'target_modules': ['q_proj', 'v_proj'],
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
        
        yield model_config_path, lora_config_path
        
        # Cleanup
        os.unlink(model_config_path)
        os.unlink(lora_config_path)
    
    @pytest.fixture
    def mock_lora_model_dir(self):
        """Create a mock LoRA model directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            lora_dir = Path(temp_dir)
            
            # Create mock LoRA files
            (lora_dir / "adapter_config.json").write_text(json.dumps({
                "base_model_name_or_path": "codellama/CodeLlama-7b-Instruct-hf",
                "bias": "none",
                "lora_alpha": 8,
                "lora_dropout": 0.1,
                "r": 4,
                "target_modules": ["q_proj", "v_proj"],
                "task_type": "CAUSAL_LM"
            }))
            
            # Create empty adapter model file
            (lora_dir / "adapter_model.safetensors").write_text("")
            
            # Create model info file
            (lora_dir / "model_info.json").write_text(json.dumps({
                "model_name": "codellama/CodeLlama-7b-Instruct-hf",
                "timestamp": "2024-01-01 12:00:00"
            }))
            
            yield str(lora_dir)
    
    def test_merge_info_creation(self, temp_config_files):
        """Test creation of merge information file."""
        model_config_path, lora_config_path = temp_config_files
        
        model_setup = ModelSetup(model_config_path, lora_config_path)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            lora_path = Path(temp_dir) / "lora"
            merged_path = Path(temp_dir) / "merged"
            merged_path.mkdir(parents=True)
            
            # Test merge info creation
            model_setup._create_merge_info(str(lora_path), str(merged_path))
            
            # Check if merge info file was created
            merge_info_path = merged_path / "merge_info.json"
            assert merge_info_path.exists()
            
            # Check merge info content
            with open(merge_info_path, 'r') as f:
                merge_info = json.load(f)
            
            assert "merge_timestamp" in merge_info
            assert merge_info["base_model"] == "codellama/CodeLlama-7b-Instruct-hf"
            assert merge_info["lora_model_path"] == str(lora_path)
            assert merge_info["merged_model_path"] == str(merged_path)
            assert "lora_config" in merge_info
            assert "model_config" in merge_info
            assert merge_info["merge_method"] == "merge_and_unload"
    
    @patch('src.model_setup.AutoModelForCausalLM')
    @patch('src.model_setup.PeftModel')
    def test_merge_lora_with_base_model_mock(self, mock_peft_model, mock_auto_model, 
                                           temp_config_files, mock_lora_model_dir):
        """Test LoRA merging with mocked models."""
        model_config_path, lora_config_path = temp_config_files
        
        # Setup mocks
        mock_base_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_base_model
        
        mock_lora_model_instance = Mock()
        mock_merged_model = Mock()
        mock_lora_model_instance.merge_and_unload.return_value = mock_merged_model
        mock_peft_model.from_pretrained.return_value = mock_lora_model_instance
        
        # Mock tokenizer setup
        with patch.object(ModelSetup, 'setup_tokenizer') as mock_setup_tokenizer:
            mock_tokenizer = Mock()
            mock_setup_tokenizer.return_value = mock_tokenizer
            
            model_setup = ModelSetup(model_config_path, lora_config_path)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "merged_model"
                
                # Perform merge
                result_path = model_setup.merge_lora_with_base_model(
                    mock_lora_model_dir, str(output_path)
                )
                
                # Verify the process
                assert result_path == str(output_path)
                assert output_path.exists()
                
                # Verify model loading was called
                mock_auto_model.from_pretrained.assert_called_once()
                # Check that PeftModel.from_pretrained was called with the base model and model path
                mock_peft_model.from_pretrained.assert_called_once()
                call_args = mock_peft_model.from_pretrained.call_args
                assert call_args[0][0] == mock_base_model
                assert str(call_args[0][1]) == mock_lora_model_dir
                
                # Verify merge was called
                mock_lora_model_instance.merge_and_unload.assert_called_once()
                
                # Verify model saving was called
                mock_merged_model.save_pretrained.assert_called_once_with(
                    output_path, safe_serialization=True
                )
                
                # Verify tokenizer saving was called
                mock_tokenizer.save_pretrained.assert_called_once_with(output_path)
                
                # Check if merge info file was created
                merge_info_path = output_path / "merge_info.json"
                assert merge_info_path.exists()
    
    def test_merge_nonexistent_model_path(self, temp_config_files):
        """Test merging with non-existent model path."""
        model_config_path, lora_config_path = temp_config_files
        
        model_setup = ModelSetup(model_config_path, lora_config_path)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_path = Path(temp_dir) / "nonexistent"
            output_path = Path(temp_dir) / "output"
            
            with pytest.raises(FileNotFoundError):
                model_setup.merge_lora_with_base_model(
                    str(nonexistent_path), str(output_path)
                )
    
    @patch('src.model_setup.AutoModelForCausalLM')
    def test_merge_model_loading_error(self, mock_auto_model, temp_config_files, mock_lora_model_dir):
        """Test handling of model loading errors during merge."""
        model_config_path, lora_config_path = temp_config_files
        
        # Make model loading fail
        mock_auto_model.from_pretrained.side_effect = Exception("Model loading failed")
        
        model_setup = ModelSetup(model_config_path, lora_config_path)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "merged_model"
            
            with pytest.raises(Exception, match="Model loading failed"):
                model_setup.merge_lora_with_base_model(
                    mock_lora_model_dir, str(output_path)
                )
