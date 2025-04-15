#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Upload AAC Conversations Dataset to Hugging Face

This script uploads the prepared AAC conversations dataset to Hugging Face.
You need to have a Hugging Face account and be logged in via the Hugging Face CLI.

Usage:
    python upload_to_huggingface.py [--input_dir INPUT_DIR] [--repo_id REPO_ID]

Example:
    python upload_to_huggingface.py --input_dir ../data --repo_id username/aac-conversations
"""

import argparse
import os
from pathlib import Path
from datasets import load_from_disk
from huggingface_hub import HfApi, create_repo

def upload_dataset(input_dir, repo_id, private=False):
    """Upload the dataset to Hugging Face."""
    # Load the dataset
    print(f"Loading dataset from {input_dir}")
    dataset = load_from_disk(input_dir)
    
    # Create or get the repository
    print(f"Creating/accessing repository: {repo_id}")
    api = HfApi()
    
    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")
    
    # Push the dataset to the hub
    print(f"Pushing dataset to {repo_id}")
    dataset.push_to_hub(
        repo_id,
        private=private,
    )
    
    print(f"Dataset uploaded successfully to https://huggingface.co/datasets/{repo_id}")

def main():
    parser = argparse.ArgumentParser(
        description="Upload AAC Conversations Dataset to Hugging Face"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="../data",
        help="Input directory containing the prepared Hugging Face dataset",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face repository ID (username/repo-name)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Whether to make the repository private",
    )
    args = parser.parse_args()
    
    # Upload the dataset
    upload_dataset(args.input_dir, args.repo_id, args.private)

if __name__ == "__main__":
    main()
