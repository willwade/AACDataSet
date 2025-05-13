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
from pathlib import Path
from datasets import load_from_disk
from huggingface_hub import create_repo

def upload_dataset(input_dir, repo_id, private=False, token=None):
    """Upload the dataset to Hugging Face.

    Args:
        input_dir: Directory containing the dataset
        repo_id: Hugging Face repository ID (username/repo-name)
        private: Whether to make the repository private
        token: Hugging Face API token. If None, will use the token from huggingface-cli login
    """
    # Load the dataset
    print(f"Loading dataset from {input_dir}")
    try:
        # Check for train/test subdirectories (DatasetDict)
        input_path = Path(input_dir)
        train_dir = input_path / "train"
        test_dir = input_path / "test"
        if train_dir.exists() and test_dir.exists():
            print("Detected train/ and test/ splits. Loading as DatasetDict.")
            from datasets import DatasetDict, load_from_disk
            dataset = DatasetDict({
                "train": load_from_disk(str(train_dir)),
                "test": load_from_disk(str(test_dir)),
            })
        else:
            print("No train/test splits detected. Loading as single dataset.")
            dataset = load_from_disk(input_dir)
    except Exception as e:
        print(f"Error loading dataset from {input_dir}: {e}")
        return False

    # Create or get the repository
    print(f"Creating/accessing repository: {repo_id}")
    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
    except Exception as e:
        print(f"Note: {e}")

    # Push the dataset to the hub
    print(f"Pushing dataset to {repo_id}")
    try:
        dataset.push_to_hub(
            repo_id,
            private=private,
            token=token
        )
        print(f"Dataset uploaded successfully to https://huggingface.co/datasets/{repo_id}")
        return True
    except Exception as e:
        print(f"Error uploading dataset to {repo_id}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Upload AAC Conversations Dataset to Hugging Face"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="huggingface/data/aac_dataset",
        help="Input directory containing the prepared Hugging Face dataset",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        default="willwade/AACConversations",
        help="Hugging Face repository ID (username/repo-name)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Whether to make the repository private",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token. If not provided, will use the token from huggingface-cli login.",
    )
    args = parser.parse_args()

    # Check if the input directory exists
    input_path = Path(args.input_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Input directory {args.input_dir} does not exist or is not a directory.")
        return

    # Upload the dataset
    success = upload_dataset(input_path, args.repo_id, args.private, args.token)

    if success:
        print(f"\nSuccessfully uploaded the AAC dataset to {args.repo_id}.")
        print(f"View your dataset at: https://huggingface.co/datasets/{args.repo_id}")
    else:
        print(f"\nFailed to upload the AAC dataset to {args.repo_id}.")
        print("Make sure you have the correct permissions and are authenticated with Hugging Face.")
        print("You can authenticate using 'huggingface-cli login' or by providing a token with --token.")

if __name__ == "__main__":
    main()
