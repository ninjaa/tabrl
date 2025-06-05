#!/usr/bin/env python3
"""
Download trained models from Modal volume using Modal CLI

Usage:
    # List models
    python download_modal_model.py --list
    
    # Download a specific model
    python download_modal_model.py --download model_name.pkl
    
    # Download using Modal CLI directly
    modal volume get tabrl-models model_name.pkl ./models/
"""
import argparse
import subprocess
import sys
from pathlib import Path

def list_models():
    """List models in Modal volume using CLI"""
    cmd = ["modal", "volume", "ls", "tabrl-models"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Models in Modal volume:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error listing models: {e.stderr}")
        sys.exit(1)

def download_model(model_name: str, output_dir: str = "./models"):
    """Download a model using Modal CLI"""
    Path(output_dir).mkdir(exist_ok=True)
    
    cmd = ["modal", "volume", "get", "tabrl-models", model_name, f"{output_dir}/"]
    print(f"Downloading {model_name}...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully downloaded to {output_dir}/{model_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model: {e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models from Modal volume")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--download", type=str, help="Model name to download")
    parser.add_argument("--output", type=str, default="./models", help="Output directory")
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
    elif args.download:
        download_model(args.download, args.output)
    else:
        parser.print_help()
