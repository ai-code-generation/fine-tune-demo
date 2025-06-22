#!/usr/bin/env python3
"""
Simple NeMo 24.07 Fine-tuning Script for CodeLlama-13B
Based on official NeMo documentation and best practices.
"""

import os
import sys
import json
import yaml
import random
import argparse
import subprocess
from pathlib import Path

def download_model(model_name: str, hf_token: str):
    """Download model from HuggingFace."""
    print(f"📥 Downloading {model_name}...")

    if model_name == "codellama-13b":
        repo_id = "meta-llama/CodeLlama-13b-hf"
        local_dir = "./codellama-13b-hf"
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
        "--precision=bf16",  # Add precision to avoid some config issues
        "--tensor_model_parallel_size=1",  # Single GPU conversion
        "--pipeline_model_parallel_size=1"
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
    
    # Load YAML data
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    # Convert to JSONL format
    jsonl_data = []
    for item in data.get('messages', []):
        if len(item) >= 2:
            # Find user and assistant messages
            user_msg = None
            assistant_msg = None
            
            for msg in item:
                if msg.get('role') == 'user':
                    user_msg = msg.get('content', '')
                elif msg.get('role') == 'assistant':
                    assistant_msg = msg.get('content', '')
            
            if user_msg and assistant_msg:
                jsonl_data.append({
                    'input': user_msg,
                    'output': assistant_msg
                })
    
    if not jsonl_data:
        print("❌ No valid training data found")
        sys.exit(1)
    
    # Shuffle data
    random.shuffle(jsonl_data)
    
    # Split 90% train, 10% validation
    split_idx = int(len(jsonl_data) * 0.9)
    train_data = jsonl_data[:split_idx]
    val_data = jsonl_data[split_idx:]
    
    # Write train file
    train_file = "train.jsonl"
    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    # Write validation file
    val_file = "validation.jsonl"
    with open(val_file, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Created {len(train_data)} training and {len(val_data)} validation examples")
    return train_file, val_file

def run_finetuning(nemo_model: str, train_file: str, val_file: str, max_steps: int = 50):
    """Run NeMo fine-tuning."""
    print(f"🚀 Starting fine-tuning for {max_steps} steps...")
    
    # Prepare paths
    model_path = os.path.abspath(nemo_model)
    train_path = os.path.abspath(train_file)
    val_path = os.path.abspath(val_file)
    
    cmd = [
        "torchrun", "--nproc_per_node=4",
        "/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
        "trainer.precision=bf16",
        "trainer.devices=4",
        "trainer.num_nodes=1",
        "trainer.val_check_interval=0.1",
        f"trainer.max_steps={max_steps}",
        f"model.restore_from_path={model_path}",
        "model.micro_batch_size=1",
        "model.global_batch_size=8",
        "model.tensor_model_parallel_size=2",
        "model.pipeline_model_parallel_size=1",
        "model.megatron_amp_O2=True",
        "model.sequence_parallel=False",
        "model.activations_checkpoint_granularity=selective",
        "model.optim.name=distributed_fused_adam",
        "model.optim.lr=1e-5",
        "model.answer_only_loss=True",
        "model.peft.peft_scheme=lora",
        f"model.data.train_ds.file_names=[{train_path}]",
        f"model.data.validation_ds.file_names=[{val_path}]",
        "model.data.train_ds.concat_sampling_probabilities=[1.0]",
        "model.data.train_ds.max_seq_length=2048",
        "model.data.validation_ds.max_seq_length=2048",
        "model.data.train_ds.micro_batch_size=1",
        "model.data.train_ds.global_batch_size=8",
        "model.data.validation_ds.micro_batch_size=1",
        "model.data.validation_ds.global_batch_size=8",
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
    parser = argparse.ArgumentParser(description="Simple NeMo 24.07 Fine-tuning for CodeLlama-13B")
    parser.add_argument("--data", required=True, help="Path to YAML training data")
    parser.add_argument("--hf-token", help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum training steps")
    parser.add_argument("--model", default="codellama-13b", help="Model to fine-tune")

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
    
    print("🎯 Simple NeMo 24.07 Fine-tuning Pipeline")
    print("=" * 50)
    
    # Step 1: Download model
    hf_path = download_model(args.model, hf_token)
    
    # Step 2: Convert to .nemo
    nemo_path = convert_to_nemo(hf_path, args.model)
    
    # Step 3: Prepare data
    train_file, val_file = prepare_data(args.data)
    
    # Step 4: Run fine-tuning
    trained_model = run_finetuning(nemo_path, train_file, val_file, args.max_steps)
    
    print("🎉 Fine-tuning pipeline completed!")
    print(f"📁 Trained model: {trained_model}")
    print(f"📁 Results directory: /results")

if __name__ == "__main__":
    main()
