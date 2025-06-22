#!/usr/bin/env python3
"""
Data preprocessing script for converting YAML training data to JSONL format
compatible with NeMo framework for fine-tuning CodeLlama and Llama3 models.

This script handles the conversion of conversation data in YAML format to
the input/output pairs required by NeMo's PEFT training pipeline.
"""

import yaml
import json
import argparse
import os
from typing import List, Dict, Any
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_yaml_data(yaml_file: str) -> List[Dict[str, Any]]:
    """Load conversation data from YAML file."""
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            # Use safe_load_all to handle multiple YAML documents separated by ---
            documents = list(yaml.safe_load_all(f))

        conversations = []
        for doc in documents:
            if doc is None:  # Skip empty documents
                continue

            if isinstance(doc, dict) and 'messages' in doc:
                # Validate that messages is a list
                if isinstance(doc['messages'], list) and len(doc['messages']) > 0:
                    conversations.append(doc)
                else:
                    logger.warning(f"Skipping document with invalid messages format")
            elif isinstance(doc, list):
                # If document is a list, extend conversations
                for item in doc:
                    if isinstance(item, dict) and 'messages' in item:
                        conversations.append(item)
                    else:
                        logger.warning(f"Skipping invalid item in list: {type(item)}")
            else:
                logger.warning(f"Skipping invalid document format: {type(doc)}")
                continue

        if not conversations:
            raise ValueError("No valid conversations found in YAML file. Expected documents with 'messages' key.")

        logger.info(f"Loaded {len(conversations)} conversations from {len(documents)} YAML documents")
        return conversations

    except Exception as e:
        logger.error(f"Error loading YAML file {yaml_file}: {e}")
        raise


def format_conversation_for_training(messages: List[Dict[str, str]]) -> str:
    """
    Format conversation messages into a single prompt string.
    Follows the chat template format commonly used for instruction tuning.
    """
    formatted_parts = []
    
    for message in messages:
        role = message.get('role', '')
        content = message.get('content', '').strip()
        
        if role == 'system':
            formatted_parts.append(f"<|system|>\n{content}")
        elif role == 'user':
            formatted_parts.append(f"<|user|>\n{content}")
        elif role == 'assistant':
            formatted_parts.append(f"<|assistant|>\n{content}")
    
    return '\n'.join(formatted_parts)


def extract_input_output_pairs(conversation: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract input/output pairs from a conversation for training.
    Each user message + assistant response becomes a training pair.
    """
    messages = conversation.get('messages', [])
    pairs = []
    
    # Find system message if present
    system_message = None
    for msg in messages:
        if msg.get('role') == 'system':
            system_message = msg.get('content', '').strip()
            break
    
    # Extract user-assistant pairs
    current_context = []
    if system_message:
        current_context.append({'role': 'system', 'content': system_message})
    
    for i, message in enumerate(messages):
        role = message.get('role')
        content = message.get('content', '').strip()
        
        if role == 'user':
            # Add user message to context
            user_context = current_context + [{'role': 'user', 'content': content}]
            
            # Look for the next assistant message
            for j in range(i + 1, len(messages)):
                if messages[j].get('role') == 'assistant':
                    assistant_content = messages[j].get('content', '').strip()
                    
                    # Create training pair
                    input_text = format_conversation_for_training(user_context)
                    output_text = assistant_content
                    
                    pairs.append({
                        'input': input_text,
                        'output': output_text
                    })
                    
                    # Update context for next iteration
                    current_context = user_context + [{'role': 'assistant', 'content': assistant_content}]
                    break
    
    return pairs


def convert_yaml_to_jsonl(yaml_file: str, output_file: str, validation_split: float = 0.1):
    """
    Convert YAML conversation data to JSONL format for NeMo training.
    
    Args:
        yaml_file: Path to input YAML file
        output_file: Base path for output JSONL files (will create .train.jsonl and .val.jsonl)
        validation_split: Fraction of data to use for validation
    """
    logger.info(f"Loading data from {yaml_file}")
    conversations = load_yaml_data(yaml_file)
    
    # Extract all training pairs
    all_pairs = []
    for conversation in conversations:
        pairs = extract_input_output_pairs(conversation)
        all_pairs.extend(pairs)
    
    logger.info(f"Extracted {len(all_pairs)} training pairs from {len(conversations)} conversations")
    
    # Split into train and validation
    import random
    random.seed(42)  # For reproducibility
    random.shuffle(all_pairs)
    
    split_idx = int(len(all_pairs) * (1 - validation_split))
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    
    # Write training data
    train_file = f"{output_file}.train.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for pair in train_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    # Write validation data
    val_file = f"{output_file}.val.jsonl"
    with open(val_file, 'w', encoding='utf-8') as f:
        for pair in val_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    logger.info(f"Created training file: {train_file} ({len(train_pairs)} samples)")
    logger.info(f"Created validation file: {val_file} ({len(val_pairs)} samples)")
    
    return train_file, val_file


def main():
    parser = argparse.ArgumentParser(description="Convert YAML conversation data to JSONL for NeMo training")
    parser.add_argument("--input", "-i", required=True, help="Input YAML file path")
    parser.add_argument("--output", "-o", required=True, help="Output base path (without extension)")
    parser.add_argument("--validation-split", "-v", type=float, default=0.1, 
                       help="Fraction of data for validation (default: 0.1)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file {args.input} does not exist")
        return 1
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        train_file, val_file = convert_yaml_to_jsonl(args.input, args.output, args.validation_split)
        logger.info("Data preprocessing completed successfully!")
        logger.info(f"Training data: {train_file}")
        logger.info(f"Validation data: {val_file}")
        return 0
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
