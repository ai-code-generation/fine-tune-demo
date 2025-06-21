"""
Data handler for processing YAML training data with conversation format.
Converts multi-turn conversations to instruction-tuning format for CodeLlama.
"""

import yaml
import logging
from typing import List, Dict, Any, Optional
from datasets import Dataset
from pathlib import Path

logger = logging.getLogger(__name__)


class ConversationDataHandler:
    """Handles loading and processing of conversation data from YAML files."""
    
    def __init__(self, tokenizer, max_length: int = 2048):
        """
        Initialize the data handler.
        
        Args:
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_yaml_conversations(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load conversations from YAML file.
        
        Args:
            file_path: Path to YAML file containing conversations
            
        Returns:
            List of conversation dictionaries
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                # Load all documents from YAML file (separated by ---)
                conversations = list(yaml.safe_load_all(file))
                logger.info(f"Loaded {len(conversations)} conversations from {file_path}")
                return conversations
        except Exception as e:
            logger.error(f"Error loading YAML file {file_path}: {e}")
            raise
    
    def format_conversation_for_training(self, conversation: Dict[str, Any]) -> str:
        """
        Convert conversation to instruction-tuning format for CodeLlama.
        
        Format: [INST] system_prompt + user_message [/INST] assistant_response
        
        Args:
            conversation: Dictionary containing 'messages' list
            
        Returns:
            Formatted string ready for training
        """
        messages = conversation.get('messages', [])
        if not messages:
            return ""
        
        # Extract system, user, and assistant messages
        system_content = ""
        user_content = ""
        assistant_content = ""
        
        for message in messages:
            role = message.get('role', '').lower()
            content = message.get('content', '').strip()
            
            if role == 'system':
                system_content = content
            elif role == 'user':
                user_content = content
            elif role == 'assistant':
                assistant_content = content
        
        # Format for CodeLlama instruction tuning
        if system_content:
            instruction = f"{system_content}\n\n{user_content}"
        else:
            instruction = user_content
            
        formatted_text = f"[INST] {instruction} [/INST] {assistant_content}"
        return formatted_text
    
    def tokenize_conversation(self, conversation_text: str) -> Dict[str, Any]:
        """
        Tokenize a formatted conversation.

        Args:
            conversation_text: Formatted conversation string

        Returns:
            Dictionary with tokenized inputs
        """
        # Tokenize the text with proper settings for training
        tokenized = self.tokenizer(
            conversation_text,
            truncation=True,
            padding=False,  # Let data collator handle padding
            max_length=self.max_length,
            return_tensors=None,
            add_special_tokens=True
        )

        # For causal language modeling, labels are the same as input_ids
        # Convert to list to ensure consistent data types
        input_ids = tokenized['input_ids']
        if isinstance(input_ids, list):
            tokenized['labels'] = input_ids.copy()
        else:
            tokenized['labels'] = input_ids.tolist()
            tokenized['input_ids'] = input_ids.tolist()

        # Ensure attention_mask is also a list
        if 'attention_mask' in tokenized:
            if not isinstance(tokenized['attention_mask'], list):
                tokenized['attention_mask'] = tokenized['attention_mask'].tolist()

        return tokenized
    
    def process_yaml_file(self, file_path: str) -> Dataset:
        """
        Process a YAML file and return a Hugging Face Dataset.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Hugging Face Dataset ready for training
        """
        # Load conversations
        conversations = self.load_yaml_conversations(file_path)
        
        # Format and tokenize conversations
        processed_data = []
        for i, conv in enumerate(conversations):
            try:
                formatted_text = self.format_conversation_for_training(conv)
                if formatted_text:  # Skip empty conversations
                    tokenized = self.tokenize_conversation(formatted_text)
                    tokenized['text'] = formatted_text  # Keep original text for reference

                    # Log sequence length for debugging
                    seq_length = len(tokenized['input_ids'])
                    if seq_length > self.max_length:
                        logger.warning(f"Conversation {i} length {seq_length} exceeds max_length {self.max_length}")

                    processed_data.append(tokenized)
            except Exception as e:
                logger.warning(f"Error processing conversation {i}: {e}")
                continue

        if processed_data:
            lengths = [len(item['input_ids']) for item in processed_data]
            logger.info(f"Successfully processed {len(processed_data)} conversations")
            logger.info(f"Sequence lengths - Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.1f}")
        else:
            logger.warning("No conversations were successfully processed")
        
        # Create Hugging Face Dataset
        dataset = Dataset.from_list(processed_data)
        return dataset
    
    def create_sample_data(self, output_path: str):
        """
        Create sample training data in the expected YAML format.
        
        Args:
            output_path: Path where to save the sample data
        """
        sample_conversations = [
            {
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are an expert test automation assistant using SWTBot.'
                    },
                    {
                        'role': 'user',
                        'content': 'Task: Create and build an S32K144 project in Eclipse IDE'
                    },
                    {
                        'role': 'assistant',
                        'content': '''```java
bot.menu("File").menu("New").menu("Project...").click();
bot.shell("New Project").activate();
bot.tree().expandNode("C/C++").select("C Project");
bot.button("Next >").click();
bot.textWithLabel("Project name:").setText("S32K144_Project");
bot.button("Finish").click();
```'''
                    }
                ]
            },
            {
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a helpful coding assistant.'
                    },
                    {
                        'role': 'user',
                        'content': 'How do I implement a binary search algorithm in Python?'
                    },
                    {
                        'role': 'assistant',
                        'content': '''```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Target not found
```'''
                    }
                ]
            }
        ]
        
        with open(output_path, 'w', encoding='utf-8') as file:
            yaml.dump_all(sample_conversations, file, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Sample data created at {output_path}")


def validate_yaml_format(file_path: str) -> bool:
    """
    Validate that a YAML file follows the expected conversation format.
    
    Args:
        file_path: Path to YAML file to validate
        
    Returns:
        True if format is valid, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            conversations = list(yaml.safe_load_all(file))
        
        for i, conv in enumerate(conversations):
            if not isinstance(conv, dict) or 'messages' not in conv:
                logger.error(f"Conversation {i} missing 'messages' key")
                return False
            
            messages = conv['messages']
            if not isinstance(messages, list):
                logger.error(f"Conversation {i} 'messages' is not a list")
                return False
            
            for j, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    logger.error(f"Message {j} in conversation {i} is not a dict")
                    return False
                
                if 'role' not in msg or 'content' not in msg:
                    logger.error(f"Message {j} in conversation {i} missing 'role' or 'content'")
                    return False
        
        logger.info(f"YAML format validation passed for {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error validating YAML format: {e}")
        return False
