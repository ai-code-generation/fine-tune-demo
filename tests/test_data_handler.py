"""
Test script for data handler functionality.
"""

import pytest
import tempfile
import yaml
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data_handler import ConversationDataHandler, validate_yaml_format
from transformers import AutoTokenizer


class TestConversationDataHandler:
    """Test cases for ConversationDataHandler."""
    
    @pytest.fixture
    def sample_conversations(self):
        """Sample conversation data for testing."""
        return [
            {
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant.'
                    },
                    {
                        'role': 'user',
                        'content': 'Write a Python function to add two numbers.'
                    },
                    {
                        'role': 'assistant',
                        'content': '''```python
def add_numbers(a, b):
    return a + b
```'''
                    }
                ]
            },
            {
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a coding expert.'
                    },
                    {
                        'role': 'user',
                        'content': 'How do I create a list in Python?'
                    },
                    {
                        'role': 'assistant',
                        'content': '''```python
# Create an empty list
my_list = []

# Create a list with items
my_list = [1, 2, 3, 4, 5]
```'''
                    }
                ]
            }
        ]
    
    @pytest.fixture
    def temp_yaml_file(self, sample_conversations):
        """Create a temporary YAML file with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump_all(sample_conversations, f, default_flow_style=False)
            return f.name
    
    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
        # Use a small tokenizer for testing
        return AutoTokenizer.from_pretrained("gpt2")
    
    def test_load_yaml_conversations(self, temp_yaml_file, tokenizer):
        """Test loading conversations from YAML file."""
        handler = ConversationDataHandler(tokenizer)
        conversations = handler.load_yaml_conversations(temp_yaml_file)
        
        assert len(conversations) == 2
        assert 'messages' in conversations[0]
        assert len(conversations[0]['messages']) == 3
        
        # Cleanup
        os.unlink(temp_yaml_file)
    
    def test_format_conversation_for_training(self, sample_conversations, tokenizer):
        """Test conversation formatting for training."""
        handler = ConversationDataHandler(tokenizer)
        
        formatted = handler.format_conversation_for_training(sample_conversations[0])
        
        assert '[INST]' in formatted
        assert '[/INST]' in formatted
        assert 'You are a helpful assistant.' in formatted
        assert 'Write a Python function' in formatted
        assert 'def add_numbers' in formatted
    
    def test_tokenize_conversation(self, tokenizer):
        """Test conversation tokenization."""
        handler = ConversationDataHandler(tokenizer, max_length=512)
        
        test_text = "[INST] Write a function [/INST] def test(): pass"
        tokenized = handler.tokenize_conversation(test_text)
        
        assert 'input_ids' in tokenized
        assert 'attention_mask' in tokenized
        assert 'labels' in tokenized
        assert len(tokenized['input_ids']) <= 512
    
    def test_process_yaml_file(self, temp_yaml_file, tokenizer):
        """Test complete YAML file processing."""
        handler = ConversationDataHandler(tokenizer, max_length=512)
        
        dataset = handler.process_yaml_file(temp_yaml_file)
        
        assert len(dataset) == 2
        assert 'input_ids' in dataset[0]
        assert 'labels' in dataset[0]
        assert 'text' in dataset[0]
        
        # Cleanup
        os.unlink(temp_yaml_file)
    
    def test_create_sample_data(self, tokenizer):
        """Test sample data creation."""
        handler = ConversationDataHandler(tokenizer)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            sample_file = f.name
        
        handler.create_sample_data(sample_file)
        
        # Verify the file was created and is valid
        assert os.path.exists(sample_file)
        assert validate_yaml_format(sample_file)
        
        # Load and check content
        conversations = handler.load_yaml_conversations(sample_file)
        assert len(conversations) >= 1
        
        # Cleanup
        os.unlink(sample_file)


class TestYAMLValidation:
    """Test cases for YAML format validation."""
    
    def test_valid_yaml_format(self):
        """Test validation of valid YAML format."""
        valid_data = [
            {
                'messages': [
                    {'role': 'system', 'content': 'You are helpful.'},
                    {'role': 'user', 'content': 'Hello'},
                    {'role': 'assistant', 'content': 'Hi there!'}
                ]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump_all(valid_data, f)
            temp_file = f.name
        
        assert validate_yaml_format(temp_file) == True
        
        # Cleanup
        os.unlink(temp_file)
    
    def test_invalid_yaml_format_missing_messages(self):
        """Test validation of invalid YAML (missing messages key)."""
        invalid_data = [
            {
                'conversations': [  # Wrong key name
                    {'role': 'user', 'content': 'Hello'}
                ]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump_all(invalid_data, f)
            temp_file = f.name
        
        assert validate_yaml_format(temp_file) == False
        
        # Cleanup
        os.unlink(temp_file)
    
    def test_invalid_yaml_format_missing_role(self):
        """Test validation of invalid YAML (missing role in message)."""
        invalid_data = [
            {
                'messages': [
                    {'content': 'Hello'}  # Missing role
                ]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump_all(invalid_data, f)
            temp_file = f.name
        
        assert validate_yaml_format(temp_file) == False
        
        # Cleanup
        os.unlink(temp_file)


def run_manual_tests():
    """Run manual tests for debugging."""
    print("Running manual tests for data handler...")
    
    # Test with actual tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        handler = ConversationDataHandler(tokenizer, max_length=512)
        
        # Test sample data creation
        print("Testing sample data creation...")
        handler.create_sample_data("test_sample.yaml")
        
        # Test validation
        print("Testing YAML validation...")
        is_valid = validate_yaml_format("test_sample.yaml")
        print(f"Validation result: {is_valid}")
        
        # Test processing
        print("Testing YAML processing...")
        dataset = handler.process_yaml_file("test_sample.yaml")
        print(f"Dataset size: {len(dataset)}")
        print(f"First item keys: {list(dataset[0].keys())}")
        
        # Test formatting
        print("Testing conversation formatting...")
        conversations = handler.load_yaml_conversations("test_sample.yaml")
        formatted = handler.format_conversation_for_training(conversations[0])
        print(f"Formatted conversation preview: {formatted[:200]}...")
        
        # Cleanup
        os.unlink("test_sample.yaml")
        
        print("All manual tests passed!")
        
    except Exception as e:
        print(f"Manual test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run manual tests if script is executed directly
    run_manual_tests()
