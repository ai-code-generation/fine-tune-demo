#!/usr/bin/env python3
"""
Unit tests for configuration management modules.
"""

import unittest
import tempfile
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.config_manager import ConfigManager
from config.config_validator import ConfigValidator


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager("llama2")
        
        # Sample configuration
        self.sample_config = {
            "model": {
                "type": "llama2",
                "size": "7b",
                "base_model_path": "meta-llama/Llama-2-7b-hf"
            },
            "data": {
                "train_file": "data/train.yaml",
                "max_seq_length": 2048,
                "batch_size": 4
            },
            "training": {
                "max_epochs": 3,
                "learning_rate": 2e-4,
                "precision": "bf16"
            },
            "lora": {
                "enabled": True,
                "rank": 32,
                "alpha": 64,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"]
            },
            "optimizer": {
                "name": "adamw",
                "lr": 2e-4,
                "betas": [0.9, 0.95]
            },
            "scheduler": {
                "name": "cosine",
                "warmup_steps": 100
            },
            "evaluation": {
                "eval_steps": 500,
                "metrics": ["perplexity", "bleu"]
            },
            "logging": {
                "log_dir": "logs"
            },
            "checkpointing": {
                "checkpoint_dir": "checkpoints"
            },
            "hardware": {
                "gpus": 1,
                "nodes": 1
            }
        }
    
    def test_initialization(self):
        """Test ConfigManager initialization."""
        # Test valid model type
        cm = ConfigManager("llama2")
        self.assertEqual(cm.model_type, "llama2")
        
        # Test invalid model type
        with self.assertRaises(ValueError):
            ConfigManager("invalid_model")
    
    def test_create_config_from_template(self):
        """Test creating config from template with overrides."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.sample_config, f)
            temp_file = f.name
        
        try:
            # Mock the template path
            original_templates = ConfigManager.CONFIG_TEMPLATES
            ConfigManager.CONFIG_TEMPLATES = {"llama2": temp_file}
            
            cm = ConfigManager("llama2")
            
            overrides = {
                "training.max_epochs": 5,
                "lora.rank": 64
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as output_f:
                output_file = output_f.name
            
            config = cm.create_config_from_template(output_file, overrides)
            
            # Verify overrides were applied
            self.assertEqual(config.training.max_epochs, 5)
            self.assertEqual(config.lora.rank, 64)
            
            # Restore original templates
            ConfigManager.CONFIG_TEMPLATES = original_templates
            
        finally:
            Path(temp_file).unlink()
            if Path(output_file).exists():
                Path(output_file).unlink()
    
    def test_get_config_sections(self):
        """Test getting specific configuration sections."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.sample_config, f)
            temp_file = f.name
        
        try:
            cm = ConfigManager("llama2")
            cm.config = cm._load_config_from_dict(self.sample_config)
            
            # Test getting model config
            model_config = cm.get_model_config()
            self.assertEqual(model_config["type"], "llama2")
            self.assertEqual(model_config["size"], "7b")
            
            # Test getting training config
            training_config = cm.get_training_config()
            self.assertEqual(training_config["max_epochs"], 3)
            self.assertEqual(training_config["learning_rate"], 2e-4)
            
            # Test getting LoRA config
            lora_config = cm.get_lora_config()
            self.assertEqual(lora_config["rank"], 32)
            self.assertEqual(lora_config["alpha"], 64)
            
        finally:
            Path(temp_file).unlink()
    
    def test_set_training_params(self):
        """Test setting training parameters."""
        cm = ConfigManager("llama2")
        cm.config = cm._load_config_from_dict(self.sample_config)
        
        cm.set_training_params(
            max_epochs=10,
            learning_rate=1e-4,
            batch_size=8,
            gpus=2
        )
        
        self.assertEqual(cm.config.training.max_epochs, 10)
        self.assertEqual(cm.config.training.learning_rate, 1e-4)
        self.assertEqual(cm.config.data.batch_size, 8)
        self.assertEqual(cm.config.hardware.gpus, 2)
    
    def test_optimize_for_hardware(self):
        """Test hardware optimization."""
        cm = ConfigManager("llama2")
        cm.config = cm._load_config_from_dict(self.sample_config)
        
        # Test optimization for low memory
        cm.optimize_for_hardware(gpu_memory_gb=12, num_gpus=1)
        self.assertEqual(cm.config.data.batch_size, 1)
        self.assertGreaterEqual(cm.config.training.accumulate_grad_batches, 8)
        
        # Test optimization for high memory
        cm.optimize_for_hardware(gpu_memory_gb=32, num_gpus=2)
        self.assertGreaterEqual(cm.config.data.batch_size, 4)
        self.assertEqual(cm.config.hardware.gpus, 2)
    
    def test_get_config_summary(self):
        """Test configuration summary."""
        cm = ConfigManager("llama2")
        cm.config = cm._load_config_from_dict(self.sample_config)
        
        summary = cm.get_config_summary()
        
        self.assertIn("model_type", summary)
        self.assertIn("model_size", summary)
        self.assertIn("max_epochs", summary)
        self.assertIn("learning_rate", summary)
        self.assertIn("lora_rank", summary)
        
        self.assertEqual(summary["model_type"], "llama2")
        self.assertEqual(summary["model_size"], "7b")
    
    def _load_config_from_dict(self, config_dict):
        """Helper method to load config from dictionary."""
        from omegaconf import OmegaConf
        return OmegaConf.create(config_dict)


class TestConfigValidator(unittest.TestCase):
    """Test cases for ConfigValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = ConfigValidator()
        
        # Valid configuration
        self.valid_config = {
            "model": {
                "type": "llama2",
                "size": "7b",
                "base_model_path": "meta-llama/Llama-2-7b-hf"
            },
            "data": {
                "train_file": "data/train.yaml",
                "max_seq_length": 2048,
                "batch_size": 4
            },
            "training": {
                "max_epochs": 3,
                "learning_rate": 2e-4,
                "precision": "bf16"
            },
            "lora": {
                "enabled": True,
                "rank": 32,
                "alpha": 64,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"]
            },
            "optimizer": {
                "name": "adamw",
                "lr": 2e-4,
                "betas": [0.9, 0.95]
            },
            "scheduler": {
                "name": "cosine",
                "warmup_steps": 100
            },
            "evaluation": {
                "eval_steps": 500
            },
            "logging": {
                "log_dir": "logs"
            },
            "checkpointing": {
                "checkpoint_dir": "checkpoints"
            },
            "hardware": {
                "gpus": 1,
                "nodes": 1
            }
        }
    
    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        is_valid = self.validator.validate_config(self.valid_config)
        self.assertTrue(is_valid)
        
        report = self.validator.get_validation_report()
        self.assertEqual(report["error_count"], 0)
        self.assertTrue(report["valid"])
    
    def test_validate_missing_sections(self):
        """Test validation with missing sections."""
        incomplete_config = {
            "model": self.valid_config["model"],
            "data": self.valid_config["data"]
            # Missing other required sections
        }
        
        is_valid = self.validator.validate_config(incomplete_config)
        self.assertFalse(is_valid)
        
        report = self.validator.get_validation_report()
        self.assertGreater(report["error_count"], 0)
        self.assertFalse(report["valid"])
    
    def test_validate_invalid_model_type(self):
        """Test validation with invalid model type."""
        invalid_config = self.valid_config.copy()
        invalid_config["model"]["type"] = "invalid_model"
        
        is_valid = self.validator.validate_config(invalid_config)
        self.assertFalse(is_valid)
        
        report = self.validator.get_validation_report()
        self.assertIn("Invalid model type", str(report["errors"]))
    
    def test_validate_invalid_model_size(self):
        """Test validation with invalid model size."""
        invalid_config = self.valid_config.copy()
        invalid_config["model"]["size"] = "invalid_size"
        
        is_valid = self.validator.validate_config(invalid_config)
        self.assertFalse(is_valid)
        
        report = self.validator.get_validation_report()
        self.assertIn("Invalid model size", str(report["errors"]))
    
    def test_validate_invalid_precision(self):
        """Test validation with invalid precision."""
        invalid_config = self.valid_config.copy()
        invalid_config["training"]["precision"] = "invalid_precision"
        
        is_valid = self.validator.validate_config(invalid_config)
        self.assertFalse(is_valid)
        
        report = self.validator.get_validation_report()
        self.assertIn("Invalid precision", str(report["errors"]))
    
    def test_validate_invalid_learning_rate(self):
        """Test validation with invalid learning rate."""
        invalid_config = self.valid_config.copy()
        invalid_config["training"]["learning_rate"] = -0.001  # Negative LR
        
        is_valid = self.validator.validate_config(invalid_config)
        self.assertFalse(is_valid)
        
        report = self.validator.get_validation_report()
        self.assertIn("Learning rate must be a positive number", str(report["errors"]))
    
    def test_validate_invalid_batch_size(self):
        """Test validation with invalid batch size."""
        invalid_config = self.valid_config.copy()
        invalid_config["data"]["batch_size"] = 0  # Zero batch size
        
        is_valid = self.validator.validate_config(invalid_config)
        self.assertFalse(is_valid)
        
        report = self.validator.get_validation_report()
        self.assertIn("Batch size must be a positive integer", str(report["errors"]))
    
    def test_validate_invalid_lora_params(self):
        """Test validation with invalid LoRA parameters."""
        # Test invalid rank
        invalid_config = self.valid_config.copy()
        invalid_config["lora"]["rank"] = -1
        
        is_valid = self.validator.validate_config(invalid_config)
        self.assertFalse(is_valid)
        
        # Test invalid dropout
        invalid_config = self.valid_config.copy()
        invalid_config["lora"]["dropout"] = 1.5  # > 1.0
        
        is_valid = self.validator.validate_config(invalid_config)
        self.assertFalse(is_valid)
    
    def test_validate_cross_dependencies(self):
        """Test cross-dependency validation."""
        # Test LoRA alpha < rank warning
        config_with_warning = self.valid_config.copy()
        config_with_warning["lora"]["alpha"] = 16  # Less than rank (32)
        
        is_valid = self.validator.validate_config(config_with_warning)
        self.assertTrue(is_valid)  # Should be valid but with warnings
        
        report = self.validator.get_validation_report()
        self.assertGreater(report["warning_count"], 0)
    
    def test_get_validation_report(self):
        """Test validation report generation."""
        # Test with invalid config
        invalid_config = {"model": {"type": "invalid"}}
        
        self.validator.validate_config(invalid_config)
        report = self.validator.get_validation_report()
        
        self.assertIn("valid", report)
        self.assertIn("errors", report)
        self.assertIn("warnings", report)
        self.assertIn("error_count", report)
        self.assertIn("warning_count", report)
        
        self.assertFalse(report["valid"])
        self.assertGreater(report["error_count"], 0)


class TestConfigIntegration(unittest.TestCase):
    """Integration tests for configuration management."""
    
    def test_manager_validator_integration(self):
        """Test integration between ConfigManager and ConfigValidator."""
        # Create a config with ConfigManager
        cm = ConfigManager("llama2")
        
        # Create sample config
        sample_config = {
            "model": {"type": "llama2", "size": "7b", "base_model_path": "test"},
            "data": {"train_file": "test.yaml", "max_seq_length": 2048, "batch_size": 4},
            "training": {"max_epochs": 3, "learning_rate": 2e-4, "precision": "bf16"},
            "lora": {"enabled": True, "rank": 32, "alpha": 64, "dropout": 0.1, "target_modules": ["q_proj"]},
            "optimizer": {"name": "adamw", "lr": 2e-4, "betas": [0.9, 0.95]},
            "scheduler": {"name": "cosine", "warmup_steps": 100},
            "evaluation": {"eval_steps": 500},
            "logging": {"log_dir": "logs"},
            "checkpointing": {"checkpoint_dir": "checkpoints"},
            "hardware": {"gpus": 1, "nodes": 1}
        }
        
        # Load config
        from omegaconf import OmegaConf
        cm.config = OmegaConf.create(sample_config)
        
        # Validate with ConfigValidator
        validator = ConfigValidator()
        config_dict = OmegaConf.to_container(cm.config, resolve=True)
        is_valid = validator.validate_config(config_dict)
        
        self.assertTrue(is_valid)
    
    def test_config_modification_and_validation(self):
        """Test modifying config and validating changes."""
        cm = ConfigManager("llama2")
        
        # Create and load sample config
        sample_config = {
            "model": {"type": "llama2", "size": "7b", "base_model_path": "test"},
            "data": {"train_file": "test.yaml", "max_seq_length": 2048, "batch_size": 4},
            "training": {"max_epochs": 3, "learning_rate": 2e-4, "precision": "bf16"},
            "lora": {"enabled": True, "rank": 32, "alpha": 64, "dropout": 0.1, "target_modules": ["q_proj"]},
            "optimizer": {"name": "adamw", "lr": 2e-4, "betas": [0.9, 0.95]},
            "scheduler": {"name": "cosine", "warmup_steps": 100},
            "evaluation": {"eval_steps": 500},
            "logging": {"log_dir": "logs"},
            "checkpointing": {"checkpoint_dir": "checkpoints"},
            "hardware": {"gpus": 1, "nodes": 1}
        }
        
        from omegaconf import OmegaConf
        cm.config = OmegaConf.create(sample_config)
        
        # Modify config
        cm.set_training_params(learning_rate=1e-3, batch_size=8)
        cm.set_lora_params(rank=64, alpha=128)
        
        # Validate modified config
        validator = ConfigValidator()
        config_dict = OmegaConf.to_container(cm.config, resolve=True)
        is_valid = validator.validate_config(config_dict)
        
        self.assertTrue(is_valid)
        
        # Verify modifications
        self.assertEqual(cm.config.training.learning_rate, 1e-3)
        self.assertEqual(cm.config.data.batch_size, 8)
        self.assertEqual(cm.config.lora.rank, 64)
        self.assertEqual(cm.config.lora.alpha, 128)


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestConfigManager))
    test_suite.addTest(unittest.makeSuite(TestConfigValidator))
    test_suite.addTest(unittest.makeSuite(TestConfigIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
