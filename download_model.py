#!/usr/bin/env python3
"""
Download Llama models to local directory
Supports downloading models from HuggingFace Hub
"""

import argparse
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def download_model(model_name: str, local_dir: str, use_safetensors: bool = True):
    """
    Download model to local directory
    
    Args:
        model_name: Model name on HuggingFace Hub (e.g., "meta-llama/Meta-Llama-3.1-8B")
        local_dir: Local save directory
        use_safetensors: Whether to use safetensors format (safer, recommended)
    """
    print(f"Downloading model: {model_name}")
    print(f"Saving to: {local_dir}")
    
    # Create directory
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        # Download tokenizer
        print("\n1. Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=local_dir,
            local_files_only=False
        )
        tokenizer.save_pretrained(local_dir)
        print(f"   ✓ Tokenizer saved to: {local_dir}")
        
        # Download model
        print("\n2. Downloading model (this may take some time depending on model size and network speed)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=local_dir,
            local_files_only=False,
            torch_dtype=torch.float16,  # Use float16 to save space
            low_cpu_mem_usage=True,
            use_safetensors=use_safetensors
        )
        model.save_pretrained(local_dir)
        print(f"   ✓ Model saved to: {local_dir}")
        
        print(f"\n✓ Model download complete!")
        print(f"  Local path: {os.path.abspath(local_dir)}")
        print(f"\nUsage:")
        print(f"  In training script: --model-name {os.path.abspath(local_dir)}")
        print(f"  Or in train_8x4090.sh: MODEL_NAME={os.path.abspath(local_dir)}")
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nTips:")
        print("1. Ensure transformers is installed: pip install transformers")
        print("2. Ensure sufficient disk space (Llama-3.1-8B requires ~16GB)")
        print("3. For Llama models, you may need to visit https://huggingface.co/meta-llama and accept the license agreement")
        print("4. For slow networks, consider using mirror sites")
        raise


def main():
    parser = argparse.ArgumentParser(description='Download Llama model to local directory')
    parser.add_argument(
        '--model-name',
        type=str,
        default='meta-llama/Meta-Llama-3.1-8B',
        help='Model name on HuggingFace Hub (default: meta-llama/Meta-Llama-3.1-8B)'
    )
    parser.add_argument(
        '--local-dir',
        type=str,
        default='./models/Meta-Llama-3.1-8B',
        help='Local save directory (default: ./models/Meta-Llama-3.1-8B)'
    )
    parser.add_argument(
        '--no-safetensors',
        action='store_true',
        help='Do not use safetensors format (default: use safetensors)'
    )
    
    args = parser.parse_args()
    
    # Convert relative path to absolute path
    local_dir = os.path.abspath(args.local_dir)
    
    download_model(
        model_name=args.model_name,
        local_dir=local_dir,
        use_safetensors=not args.no_safetensors
    )


if __name__ == "__main__":
    main()
