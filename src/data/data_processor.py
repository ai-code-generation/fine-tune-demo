"""
Data processing utilities for handling YAML training data format.
"""

import yaml
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles loading and processing of YAML training data."""
    
    def __init__(self, model_type: str = "llama2"):
        """
        Initialize the data processor.
        
        Args:
            model_type: Type of model (llama2, llama3, codellama)
        """
        self.model_type = model_type.lower()
        self.supported_models = ["llama2", "llama3", "codellama"]
        
        if self.model_type not in self.supported_models:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: {self.supported_models}")
    
    def load_yaml_data(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load training data from YAML file.
        
        Args:
            file_path: Path to the YAML file
            
        Returns:
            List of conversation dictionaries
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Load all documents from YAML file
                documents = list(yaml.safe_load_all(f))
                
            conversations = []
            for doc in documents:
                if doc and 'messages' in doc:
                    conversations.append(doc)
                    
            logger.info(f"Loaded {len(conversations)} conversations from {file_path}")
            return conversations
            
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {file_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading data from {file_path}: {e}")
    
    def validate_conversation(self, conversation: Dict[str, Any]) -> bool:
        """
        Validate a conversation structure.
        
        Args:
            conversation: Conversation dictionary
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(conversation, dict):
            return False
            
        if 'messages' not in conversation:
            return False
            
        messages = conversation['messages']
        if not isinstance(messages, list) or len(messages) == 0:
            return False
            
        # Check message structure
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if 'role' not in msg or 'content' not in msg:
                return False
            if msg['role'] not in ['system', 'user', 'assistant']:
                return False
                
        return True
    
    def filter_conversations(self, conversations: List[Dict[str, Any]], 
                           min_length: int = 1, 
                           max_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Filter conversations based on length and validity.
        
        Args:
            conversations: List of conversations
            min_length: Minimum number of messages
            max_length: Maximum number of messages (None for no limit)
            
        Returns:
            Filtered list of conversations
        """
        filtered = []
        
        for conv in conversations:
            if not self.validate_conversation(conv):
                logger.warning("Skipping invalid conversation")
                continue
                
            messages = conv['messages']
            msg_count = len(messages)
            
            if msg_count < min_length:
                continue
                
            if max_length and msg_count > max_length:
                continue
                
            filtered.append(conv)
            
        logger.info(f"Filtered {len(filtered)} valid conversations from {len(conversations)}")
        return filtered
    
    def save_processed_data(self, conversations: List[Dict[str, Any]], 
                          output_path: Union[str, Path], 
                          format: str = "jsonl") -> None:
        """
        Save processed conversations to file.
        
        Args:
            conversations: List of conversations
            output_path: Output file path
            format: Output format (jsonl, json, yaml)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for conv in conversations:
                    f.write(json.dumps(conv, ensure_ascii=False) + '\n')
                    
        elif format.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, ensure_ascii=False, indent=2)
                
        elif format.lower() == "yaml":
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump_all(conversations, f, default_flow_style=False, 
                            allow_unicode=True)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Saved {len(conversations)} conversations to {output_path}")
    
    def get_statistics(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the conversations.
        
        Args:
            conversations: List of conversations
            
        Returns:
            Dictionary with statistics
        """
        if not conversations:
            return {"total_conversations": 0}
            
        total_conversations = len(conversations)
        total_messages = sum(len(conv['messages']) for conv in conversations)
        
        # Count by role
        role_counts = {"system": 0, "user": 0, "assistant": 0}
        message_lengths = []
        
        for conv in conversations:
            for msg in conv['messages']:
                role = msg['role']
                if role in role_counts:
                    role_counts[role] += 1
                message_lengths.append(len(msg['content']))
        
        avg_msg_length = sum(message_lengths) / len(message_lengths) if message_lengths else 0
        
        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "avg_messages_per_conversation": total_messages / total_conversations,
            "role_distribution": role_counts,
            "avg_message_length": avg_msg_length,
            "min_message_length": min(message_lengths) if message_lengths else 0,
            "max_message_length": max(message_lengths) if message_lengths else 0
        }
