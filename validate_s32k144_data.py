#!/usr/bin/env python3
"""
Simple validation script for S32K144 training data.
"""

import yaml
import os
from pathlib import Path

def validate_yaml_file(file_path):
    """Validate a YAML file and return statistics."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            documents = list(yaml.safe_load_all(f))
        
        conversations = 0
        total_messages = 0
        s32k144_mentions = 0
        swtbot_mentions = 0
        
        for doc in documents:
            if doc and 'messages' in doc:
                conversations += 1
                messages = doc['messages']
                total_messages += len(messages)
                
                # Check for S32K144 and SWTBot content
                for msg in messages:
                    content = msg.get('content', '').lower()
                    if 's32k144' in content:
                        s32k144_mentions += 1
                    if 'swtbot' in content or 'bot.' in content:
                        swtbot_mentions += 1
        
        return {
            'valid': True,
            'conversations': conversations,
            'total_messages': total_messages,
            's32k144_mentions': s32k144_mentions,
            'swtbot_mentions': swtbot_mentions,
            'file_size': os.path.getsize(file_path)
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }

def main():
    """Validate all S32K144 training data files."""
    print("=" * 60)
    print("S32K144 Training Data Validation")
    print("=" * 60)
    
    files_to_check = [
        'data/train.yaml',
        'data/val.yaml', 
        'data/test.yaml'
    ]
    
    total_conversations = 0
    total_messages = 0
    total_size = 0
    all_valid = True
    
    for file_path in files_to_check:
        print(f"\nValidating: {file_path}")
        print("-" * 40)
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            all_valid = False
            continue
        
        stats = validate_yaml_file(file_path)
        
        if stats['valid']:
            print(f"✅ Valid YAML file")
            print(f"📊 Conversations: {stats['conversations']}")
            print(f"💬 Total messages: {stats['total_messages']}")
            print(f"🔧 S32K144 mentions: {stats['s32k144_mentions']}")
            print(f"🤖 SWTBot mentions: {stats['swtbot_mentions']}")
            print(f"📁 File size: {stats['file_size']:,} bytes")
            
            total_conversations += stats['conversations']
            total_messages += stats['total_messages']
            total_size += stats['file_size']
        else:
            print(f"❌ Invalid file: {stats['error']}")
            all_valid = False
    
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"📊 Total conversations: {total_conversations}")
    print(f"💬 Total messages: {total_messages}")
    print(f"📁 Total data size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    
    # Check if data is sufficient for fine-tuning
    if total_conversations >= 10:
        print(f"✅ Sufficient training examples ({total_conversations} conversations)")
    else:
        print(f"⚠️  Limited training examples ({total_conversations} conversations)")
    
    if total_size >= 30000:  # At least 30KB
        print(f"✅ Adequate data size for fine-tuning")
    else:
        print(f"⚠️  Small data size, may need more examples")
    
    if all_valid:
        print(f"\n🎉 All files are valid and ready for fine-tuning!")
        print(f"\nYou can now run:")
        print(f"python scripts/run_pipeline.py \\")
        print(f"  --model-type llama2 \\")
        print(f"  --model-size 7b \\")
        print(f"  --train-file data/train.yaml \\")
        print(f"  --val-file data/val.yaml \\")
        print(f"  --test-file data/test.yaml \\")
        print(f"  --max-epochs 3")
        return True
    else:
        print(f"\n❌ Some files have issues. Please fix them before training.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
