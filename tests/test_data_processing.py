#!/usr/bin/env python3
"""
Unit tests for data processing modules.
"""

import unittest
import tempfile
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.data_processor import DataProcessor
from data.instruction_formatter import InstructionFormatter
from data.dataset_builder import DatasetBuilder


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DataProcessor("llama2")
        
        # Sample conversation data
        self.sample_conversations = [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing well, thank you!"}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "2+2 equals 4."}
                ]
            }
        ]
    
    def test_validate_conversation_valid(self):
        """Test validation of valid conversations."""
        for conv in self.sample_conversations:
            self.assertTrue(self.processor.validate_conversation(conv))
    
    def test_validate_conversation_invalid(self):
        """Test validation of invalid conversations."""
        invalid_conversations = [
            {},  # Empty dict
            {"messages": []},  # Empty messages
            {"messages": [{"role": "invalid", "content": "test"}]},  # Invalid role
            {"messages": [{"role": "user"}]},  # Missing content
        ]
        
        for conv in invalid_conversations:
            self.assertFalse(self.processor.validate_conversation(conv))
    
    def test_filter_conversations(self):
        """Test conversation filtering."""
        # Test minimum length filter
        filtered = self.processor.filter_conversations(
            self.sample_conversations, 
            min_length=3
        )
        self.assertEqual(len(filtered), 1)  # Only first conversation has 3+ messages
        
        # Test maximum length filter
        filtered = self.processor.filter_conversations(
            self.sample_conversations,
            max_length=2
        )
        self.assertEqual(len(filtered), 1)  # Only second conversation has ≤2 messages
    
    def test_get_statistics(self):
        """Test statistics calculation."""
        stats = self.processor.get_statistics(self.sample_conversations)
        
        self.assertEqual(stats["total_conversations"], 2)
        self.assertEqual(stats["total_messages"], 5)
        self.assertEqual(stats["role_distribution"]["user"], 2)
        self.assertEqual(stats["role_distribution"]["assistant"], 2)
        self.assertEqual(stats["role_distribution"]["system"], 1)
    
    def test_load_yaml_data(self):
        """Test loading YAML data from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump_all(self.sample_conversations, f)
            temp_file = f.name
        
        try:
            loaded_conversations = self.processor.load_yaml_data(temp_file)
            self.assertEqual(len(loaded_conversations), 2)
            self.assertEqual(loaded_conversations, self.sample_conversations)
        finally:
            Path(temp_file).unlink()


class TestInstructionFormatter(unittest.TestCase):
    """Test cases for InstructionFormatter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_conversation = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }
    
    def test_llama2_formatting(self):
        """Test LLaMA2 instruction formatting."""
        formatter = InstructionFormatter("llama2")
        formatted = formatter.format_conversation(self.sample_conversation)
        
        # Check that LLaMA2 format tokens are present
        self.assertIn("<s>", formatted)
        self.assertIn("[INST]", formatted)
        self.assertIn("[/INST]", formatted)
        self.assertIn("<<SYS>>", formatted)
        self.assertIn("<</SYS>>", formatted)
    
    def test_llama3_formatting(self):
        """Test LLaMA3 instruction formatting."""
        formatter = InstructionFormatter("llama3")
        formatted = formatter.format_conversation(self.sample_conversation)
        
        # Check that LLaMA3 format tokens are present
        self.assertIn("<|begin_of_text|>", formatted)
        self.assertIn("<|start_header_id|>", formatted)
        self.assertIn("<|end_header_id|>", formatted)
        self.assertIn("<|eot_id|>", formatted)
    
    def test_codellama_formatting(self):
        """Test CodeLlama instruction formatting."""
        formatter = InstructionFormatter("codellama")
        formatted = formatter.format_conversation(self.sample_conversation)
        
        # CodeLlama uses similar format to LLaMA2
        self.assertIn("<s>", formatted)
        self.assertIn("[INST]", formatted)
        self.assertIn("[/INST]", formatted)
    
    def test_get_special_tokens(self):
        """Test special tokens retrieval."""
        # Test LLaMA2 tokens
        formatter = InstructionFormatter("llama2")
        tokens = formatter.get_special_tokens()
        self.assertIn("bos_token", tokens)
        self.assertIn("eos_token", tokens)
        self.assertEqual(tokens["bos_token"], "<s>")
        self.assertEqual(tokens["eos_token"], "</s>")
        
        # Test LLaMA3 tokens
        formatter = InstructionFormatter("llama3")
        tokens = formatter.get_special_tokens()
        self.assertEqual(tokens["bos_token"], "<|begin_of_text|>")
        self.assertEqual(tokens["eos_token"], "<|end_of_text|>")
    
    def test_format_for_training(self):
        """Test formatting multiple conversations for training."""
        conversations = [self.sample_conversation] * 3
        formatter = InstructionFormatter("llama2")
        
        formatted = formatter.format_for_training(conversations)
        self.assertEqual(len(formatted), 3)
        
        for formatted_conv in formatted:
            self.assertIsInstance(formatted_conv, str)
            self.assertIn("[INST]", formatted_conv)


class TestDatasetBuilder(unittest.TestCase):
    """Test cases for DatasetBuilder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.builder = DatasetBuilder("llama2")
        self.sample_formatted_conversations = [
            "<s>[INST] Hello! [/INST] Hi there! </s>",
            "<s>[INST] How are you? [/INST] I'm doing well! </s>"
        ]
    
    def test_get_special_tokens(self):
        """Test special tokens configuration."""
        special_tokens = self.builder._get_special_tokens()
        
        self.assertIn("bos_token", special_tokens)
        self.assertIn("eos_token", special_tokens)
        self.assertEqual(special_tokens["bos_token"], "<s>")
        self.assertEqual(special_tokens["eos_token"], "</s>")
    
    def test_create_dataset_info(self):
        """Test dataset info creation."""
        # This test would require a mock tokenizer
        # For now, just test that the method exists
        self.assertTrue(hasattr(self.builder, 'save_dataset_info'))


class TestIntegration(unittest.TestCase):
    """Integration tests for data processing pipeline."""
    
    def test_full_pipeline(self):
        """Test complete data processing pipeline."""
        # Sample data
        conversations = [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a programming language."}
                ]
            }
        ]
        
        # Process data
        processor = DataProcessor("llama2")
        filtered_conversations = processor.filter_conversations(conversations)
        
        # Format for training
        formatter = InstructionFormatter("llama2")
        formatted_conversations = formatter.format_for_training(filtered_conversations)
        
        # Verify pipeline
        self.assertEqual(len(filtered_conversations), 1)
        self.assertEqual(len(formatted_conversations), 1)
        self.assertIsInstance(formatted_conversations[0], str)
        self.assertIn("[INST]", formatted_conversations[0])
    
    def test_model_type_consistency(self):
        """Test that model type is consistent across components."""
        model_type = "llama3"
        
        processor = DataProcessor(model_type)
        formatter = InstructionFormatter(model_type)
        builder = DatasetBuilder(model_type)
        
        self.assertEqual(processor.model_type, model_type)
        self.assertEqual(formatter.model_type, model_type)
        self.assertEqual(builder.model_type, model_type)


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDataProcessor))
    test_suite.addTest(unittest.makeSuite(TestInstructionFormatter))
    test_suite.addTest(unittest.makeSuite(TestDatasetBuilder))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
