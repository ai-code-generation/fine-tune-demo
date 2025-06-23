#!/usr/bin/env python3
"""
Simple NeMo 24.07 Fine-tuning Script for CodeLlama-7B
Optimized for smaller model with reduced resource requirements.
"""

import os
import sys
import json
import yaml
import random
import argparse
import subprocess
import re
import glob

def clean_text_for_json(text: str) -> str:
    """Clean text content to ensure valid JSON serialization."""
    if not text:
        return ""

    # Convert to string if not already
    text = str(text)

    # Remove or replace problematic characters
    # Replace control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

    # Normalize whitespace but preserve structure
    text = re.sub(r'\r\n', '\n', text)  # Normalize line endings
    text = re.sub(r'\r', '\n', text)    # Convert remaining \r to \n

    # Limit extremely long lines that might cause issues
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if len(line) > 2000:  # Truncate very long lines
            line = line[:2000] + "..."
        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # Remove excessive whitespace
    text = re.sub(r'\n{4,}', '\n\n\n', text)  # Max 3 consecutive newlines

    # Trim whitespace
    text = text.strip()

    return text

def download_model(model_name: str, hf_token: str):
    """Download model from HuggingFace."""
    print(f"📥 Downloading {model_name}...")
    
    if model_name == "codellama-7b":
        repo_id = "meta-llama/CodeLlama-7b-hf"
        local_dir = "./codellama-7b-hf"
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Check if already downloaded
    if os.path.exists(local_dir):
        print(f"✅ Model already exists at {local_dir}")
        return local_dir
    
    # Set up cache directories
    cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(f"{cache_dir}/huggingface", exist_ok=True)
    os.makedirs(f"{cache_dir}/transformers", exist_ok=True)
    
    # Set environment variables for cache
    os.environ["HF_HOME"] = f"{cache_dir}/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = f"{cache_dir}/transformers"
    os.environ["HF_DATASETS_CACHE"] = f"{cache_dir}/datasets"
    
    # Download using huggingface_hub
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=hf_token
        )
        print(f"✅ Downloaded to {local_dir}")
        return local_dir
    except Exception as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)

def convert_to_nemo(hf_path: str, model_name: str):
    """Convert HuggingFace model to .nemo format."""
    nemo_path = f"./{model_name}.nemo"
    
    # Check if already converted
    if os.path.exists(nemo_path):
        print(f"✅ NeMo model already exists at {nemo_path}")
        return nemo_path
    
    print(f"🔄 Converting {hf_path} to {nemo_path}...")
    
    # Check GPU availability
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode != 0:
            print("⚠️  Warning: nvidia-smi failed, but continuing conversion...")
    except FileNotFoundError:
        print("⚠️  Warning: nvidia-smi not found, but continuing conversion...")
    
    # Set environment variables for conversion
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
    env["CUDA_LAUNCH_BLOCKING"] = "1"  # For better error reporting
    
    cmd = [
        "python", "/opt/NeMo/scripts/checkpoint_converters/convert_llama_hf_to_nemo.py",
        f"--input_name_or_path={hf_path}",
        f"--output_path={nemo_path}",
        "--precision=bf16"  # Add precision to avoid some config issues
    ]
    
    try:
        print(f"Running conversion command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)
        print(f"✅ Converted to {nemo_path}")
        return nemo_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")
        print("💡 Troubleshooting tips:")
        print("   1. Make sure you're running inside the NeMo container with GPU access")
        print("   2. Check if GPUs are visible: nvidia-smi")
        print("   3. Try running the container with: docker run --gpus all ...")
        sys.exit(1)

def prepare_data(yaml_file: str):
    """Convert YAML training data to JSONL format and split."""
    print(f"📝 Preparing data from {yaml_file}...")

    # Load YAML data (handle multiple documents)
    jsonl_data = []
    with open(yaml_file, 'r', encoding='utf-8') as f:
        # Load all YAML documents in the file
        documents = list(yaml.safe_load_all(f))

    # Process each document
    for doc_idx, doc in enumerate(documents):
        if doc is None:
            continue

        # Convert to JSONL format
        messages = doc.get('messages', [])
        if len(messages) >= 2:
            # Find user and assistant messages
            user_msg = None
            assistant_msg = None

            for msg in messages:
                if msg.get('role') == 'user':
                    user_msg = msg.get('content', '')
                elif msg.get('role') == 'assistant':
                    assistant_msg = msg.get('content', '')

            if user_msg and assistant_msg:
                # Clean and validate the content
                user_msg = clean_text_for_json(user_msg)
                assistant_msg = clean_text_for_json(assistant_msg)

                # Skip if content is too long (NeMo has sequence length limits)
                if len(user_msg) > 8000 or len(assistant_msg) > 8000:
                    print(f"⚠️  Skipping document {doc_idx}: content too long")
                    continue

                jsonl_data.append({
                    'input': user_msg,
                    'output': assistant_msg
                })

    if not jsonl_data:
        print("❌ No valid training data found")
        print("💡 Make sure your YAML file contains 'messages' with 'user' and 'assistant' roles")
        sys.exit(1)
    
    # Shuffle data
    random.shuffle(jsonl_data)
    
    # Split 90% train, 10% validation
    split_idx = int(len(jsonl_data) * 0.9)
    train_data = jsonl_data[:split_idx]
    val_data = jsonl_data[split_idx:]
    
    # Write train file with proper JSON encoding and Unix line endings
    train_file = "train.jsonl"
    with open(train_file, 'w', encoding='utf-8', newline='\n') as f:
        for item in train_data:
            try:
                # Ensure consistent JSON formatting
                json_line = json.dumps(item, ensure_ascii=False, separators=(',', ':'))
                # Validate the JSON can be parsed back
                json.loads(json_line)
                f.write(json_line + '\n')
            except (UnicodeEncodeError, TypeError, json.JSONDecodeError) as e:
                print(f"⚠️  Skipping invalid training item: {e}")
                continue

    # Write validation file with proper JSON encoding and Unix line endings
    val_file = "validation.jsonl"
    with open(val_file, 'w', encoding='utf-8', newline='\n') as f:
        for item in val_data:
            try:
                # Ensure consistent JSON formatting
                json_line = json.dumps(item, ensure_ascii=False, separators=(',', ':'))
                # Validate the JSON can be parsed back
                json.loads(json_line)
                f.write(json_line + '\n')
            except (UnicodeEncodeError, TypeError, json.JSONDecodeError) as e:
                print(f"⚠️  Skipping invalid validation item: {e}")
                continue
    
    # Validate the generated JSONL files
    print("🔍 Validating generated JSONL files...")
    train_valid = validate_jsonl_file(train_file)
    val_valid = validate_jsonl_file(val_file)

    if not train_valid or not val_valid:
        print("❌ Generated JSONL files contain invalid JSON")
        sys.exit(1)

    print(f"✅ Created {len(train_data)} training and {len(val_data)} validation examples")
    print(f"✅ JSONL files validated successfully")
    return train_file, val_file

def validate_jsonl_file(file_path: str) -> bool:
    """Validate that a JSONL file contains valid JSON lines."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"❌ Invalid JSON in {file_path} line {line_num}: {e}")
                        print(f"   Line content: {line[:100]}...")
                        return False
        return True
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False

def cleanup_nemo_index_files():
    """Clean up any existing NeMo index files that might be corrupted."""
    print("🧹 Cleaning up existing NeMo index files...")

    # Find and remove .idx files
    import glob
    idx_files = glob.glob("*.jsonl.idx.*")

    if idx_files:
        for idx_file in idx_files:
            try:
                os.remove(idx_file)
                print(f"   Removed: {idx_file}")
            except Exception as e:
                print(f"   Failed to remove {idx_file}: {e}")
        print(f"✅ Cleaned up {len(idx_files)} index files")
    else:
        print("   No index files found to clean up")

def run_finetuning(nemo_model: str, train_file: str, val_file: str, max_steps: int = 50):
    """Run NeMo fine-tuning optimized for CodeLlama-7B."""
    print(f"🚀 Starting fine-tuning for {max_steps} steps...")
    
    # Prepare paths
    model_path = os.path.abspath(nemo_model)
    train_path = os.path.abspath(train_file)
    val_path = os.path.abspath(val_file)
    
    # CodeLlama-7B optimized settings (smaller model, less resources needed)
    cmd = [
        "torchrun", "--nproc_per_node=2",  # Use 2 GPUs instead of 4
        "/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
        "trainer.precision=bf16",
        "trainer.devices=2",  # 2 GPUs for 7B model
        "trainer.num_nodes=1",
        "trainer.val_check_interval=0.1",
        f"trainer.max_steps={max_steps}",
        f"model.restore_from_path={model_path}",
        "model.micro_batch_size=2",  # Larger micro batch for 7B
        "model.global_batch_size=16",  # Larger global batch
        "model.tensor_model_parallel_size=1",  # No TP for 7B
        "model.pipeline_model_parallel_size=1",
        "model.megatron_amp_O2=True",
        "model.sequence_parallel=False",
        "model.activations_checkpoint_granularity=selective",
        "model.optim.name=fused_adam",
        "model.optim.lr=2e-5",  # Slightly higher LR for smaller model
        "model.answer_only_loss=True",
        "model.peft.peft_scheme=lora",
        f"model.data.train_ds.file_names=[{train_path}]",
        f"model.data.validation_ds.file_names=[{val_path}]",
        "model.data.train_ds.concat_sampling_probabilities=[1.0]",
        "model.data.train_ds.max_seq_length=2048",
        "model.data.validation_ds.max_seq_length=2048",
        "model.data.train_ds.micro_batch_size=2",
        "model.data.train_ds.global_batch_size=16",
        "model.data.validation_ds.micro_batch_size=2",
        "model.data.validation_ds.global_batch_size=16",
        "model.data.train_ds.num_workers=0",
        "model.data.validation_ds.num_workers=0",
        "exp_manager.create_wandb_logger=False",
        "exp_manager.explicit_log_dir=/results",
        "exp_manager.resume_if_exists=True",
        "exp_manager.resume_ignore_no_checkpoint=True",
        "exp_manager.create_checkpoint_callback=True",
        "exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Fine-tuning completed successfully!")
        return "/results/checkpoints/megatron_gpt_peft_lora_tuning.nemo"
    except subprocess.CalledProcessError as e:
        print(f"❌ Fine-tuning failed: {e}")
        sys.exit(1)

def check_gpu_access():
    """Check if GPUs are accessible."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU access confirmed")
            # Count GPUs
            gpu_count = result.stdout.count("GeForce") + result.stdout.count("Tesla") + result.stdout.count("A100") + result.stdout.count("V100")
            if gpu_count == 0:
                # Try a different way to count GPUs
                gpu_lines = [line for line in result.stdout.split('\n') if 'MiB' in line and '%' in line]
                gpu_count = len(gpu_lines)
            print(f"📊 Detected {gpu_count} GPU(s)")
            return True
        else:
            print("❌ nvidia-smi failed")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        return False

def main():
    parser = argparse.ArgumentParser(description="Simple NeMo 24.07 Fine-tuning for CodeLlama-7B")
    parser.add_argument("--data", required=True, help="Path to YAML training data")
    parser.add_argument("--hf-token", help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum training steps")
    parser.add_argument("--model", default="codellama-7b", help="Model to fine-tune")
    
    args = parser.parse_args()
    
    # Get HF token
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')
    if not hf_token:
        print("❌ HuggingFace token required. Use --hf-token or set HF_TOKEN env var")
        sys.exit(1)
    
    # Check if we're in NeMo container
    if not os.path.exists('/opt/NeMo'):
        print("❌ This script must be run inside NeMo 24.07 container")
        print("Run: docker run --gpus all --shm-size=2g --net=host --ulimit memlock=-1 --rm -it \\")
        print("  -v ${PWD}:/workspace -w /workspace -v ${PWD}/results:/results \\")
        print("  nvcr.io/nvidia/nemo:24.07 bash")
        sys.exit(1)
    
    # Check GPU access
    if not check_gpu_access():
        print("❌ GPU access required for NeMo model conversion and training")
        print("💡 Make sure to run container with: docker run --gpus all ...")
        sys.exit(1)
    
    print("🎯 Simple NeMo 24.07 Fine-tuning Pipeline for CodeLlama-7B")
    print("=" * 60)
    
    # Step 1: Download model
    hf_path = download_model(args.model, hf_token)
    
    # Step 2: Convert to .nemo
    nemo_path = convert_to_nemo(hf_path, args.model)
    
    # Step 3: Clean up any existing corrupted index files
    cleanup_nemo_index_files()

    # Step 4: Prepare data
    train_file, val_file = prepare_data(args.data)
    
    # Step 5: Run fine-tuning
    trained_model = run_finetuning(nemo_path, train_file, val_file, args.max_steps)
    
    print("🎉 Fine-tuning pipeline completed!")
    print(f"📁 Trained model: {trained_model}")
    print(f"📁 Results directory: /results")

if __name__ == "__main__":
    main()
