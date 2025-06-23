#!/usr/bin/env python3
"""
Debug script to analyze JSONL files and fix memmap indexing issues.
This script helps identify problems with line endings, JSON formatting, and file structure.
"""

import json
import os
import sys

def analyze_jsonl_file(file_path):
    """Analyze a JSONL file for potential issues."""
    print(f"\n🔍 Analyzing {file_path}")
    print("=" * 50)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    # Check file size
    file_size = os.path.getsize(file_path)
    print(f"📁 File size: {file_size:,} bytes")
    
    # Read file in binary mode to check line endings
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # Count different line ending types
    crlf_count = content.count(b'\r\n')
    lf_count = content.count(b'\n') - crlf_count  # Subtract CRLF to avoid double counting
    cr_count = content.count(b'\r') - crlf_count
    
    print(f"📋 Line endings:")
    print(f"   - CRLF (\\r\\n): {crlf_count}")
    print(f"   - LF (\\n): {lf_count}")
    print(f"   - CR (\\r): {cr_count}")
    
    # Analyze JSON lines
    valid_lines = 0
    invalid_lines = 0
    total_lines = 0
    max_line_length = 0
    min_line_length = float('inf')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                line = line.strip()
                
                if not line:  # Skip empty lines
                    continue
                
                line_length = len(line)
                max_line_length = max(max_line_length, line_length)
                min_line_length = min(min_line_length, line_length)
                
                try:
                    json.loads(line)
                    valid_lines += 1
                except json.JSONDecodeError as e:
                    invalid_lines += 1
                    print(f"❌ Invalid JSON on line {line_num}: {e}")
                    print(f"   Line content (first 100 chars): {line[:100]}...")
                    
                    # Try to identify the issue
                    if line.count('"') % 2 != 0:
                        print(f"   🔍 Unmatched quotes detected")
                    if '\\' in line and not line.endswith('\\'):
                        print(f"   🔍 Potential escape sequence issues")
                    
                    # Show where the error occurs
                    try:
                        # Try to find where JSON parsing fails
                        for i in range(len(line)):
                            try:
                                json.loads(line[:i+1])
                            except json.JSONDecodeError:
                                if i > 50:  # Only show if we got reasonably far
                                    print(f"   🔍 JSON parsing fails around character {i}: '{line[max(0,i-10):i+10]}'")
                                break
                    except:
                        pass
    
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    print(f"\n📊 Analysis Results:")
    print(f"   - Total lines: {total_lines}")
    print(f"   - Valid JSON lines: {valid_lines}")
    print(f"   - Invalid JSON lines: {invalid_lines}")
    print(f"   - Max line length: {max_line_length:,} characters")
    print(f"   - Min line length: {min_line_length:,} characters")
    
    if invalid_lines > 0:
        print(f"\n⚠️  Found {invalid_lines} invalid JSON lines!")
        return False
    else:
        print(f"\n✅ All JSON lines are valid!")
        return True

def fix_jsonl_file(input_file, output_file):
    """Fix common JSONL file issues."""
    print(f"\n🔧 Fixing {input_file} -> {output_file}")
    
    fixed_lines = 0
    skipped_lines = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            
            if not line:  # Skip empty lines
                continue
            
            try:
                # Try to parse the JSON
                data = json.loads(line)
                
                # Re-serialize with consistent formatting
                fixed_line = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
                outfile.write(fixed_line + '\n')
                fixed_lines += 1
                
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping invalid line {line_num}: {e}")
                skipped_lines += 1
                continue
    
    print(f"✅ Fixed {fixed_lines} lines, skipped {skipped_lines} invalid lines")
    return fixed_lines > 0

def main():
    """Main function to debug JSONL files."""
    print("🔍 JSONL File Debugger for NeMo Training")
    print("=" * 50)
    
    # Check current directory for JSONL files
    jsonl_files = [f for f in os.listdir('.') if f.endswith('.jsonl')]
    
    if not jsonl_files:
        print("❌ No JSONL files found in current directory")
        return
    
    print(f"📁 Found JSONL files: {jsonl_files}")
    
    all_valid = True
    
    # Analyze each file
    for file_path in jsonl_files:
        is_valid = analyze_jsonl_file(file_path)
        if not is_valid:
            all_valid = False
            
            # Offer to fix the file
            response = input(f"\n🔧 Fix {file_path}? (y/n): ").lower().strip()
            if response == 'y':
                backup_file = f"{file_path}.backup"
                fixed_file = f"{file_path}.fixed"
                
                # Create backup
                os.rename(file_path, backup_file)
                print(f"📋 Created backup: {backup_file}")
                
                # Fix the file
                if fix_jsonl_file(backup_file, fixed_file):
                    # Replace original with fixed version
                    os.rename(fixed_file, file_path)
                    print(f"✅ Fixed file saved as: {file_path}")
                    
                    # Re-analyze to confirm fix
                    print(f"\n🔍 Re-analyzing fixed file...")
                    if analyze_jsonl_file(file_path):
                        print(f"✅ File {file_path} is now valid!")
                    else:
                        print(f"❌ File {file_path} still has issues")
                else:
                    # Restore backup if fix failed
                    os.rename(backup_file, file_path)
                    print(f"❌ Fix failed, restored original file")
    
    if all_valid:
        print(f"\n🎉 All JSONL files are valid and ready for NeMo training!")
    else:
        print(f"\n⚠️  Some JSONL files have issues that need to be resolved")
    
    # Additional recommendations
    print(f"\n💡 Recommendations for NeMo training:")
    print(f"   1. Ensure all lines end with \\n (Unix line endings)")
    print(f"   2. Each line should be valid JSON with 'input' and 'output' fields")
    print(f"   3. Avoid extremely long lines (>10,000 characters)")
    print(f"   4. Use UTF-8 encoding consistently")
    print(f"   5. Remove any BOM (Byte Order Mark) from files")

if __name__ == "__main__":
    main()
