#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example Usage of AAC Conversations Dataset

This script demonstrates how to use the AAC Conversations dataset for various tasks:
1. AAC utterance correction (noisy to clean)
2. AAC utterance expansion (telegraphic to full)
3. AAC response prediction

Usage:
    python example_usage.py [--data_dir DATA_DIR]

Example:
    python example_usage.py --data_dir ../data
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import load_from_disk
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

def load_dataset(data_dir):
    """Load the dataset from disk."""
    print(f"Loading dataset from {data_dir}")
    dataset = load_from_disk(data_dir)
    return dataset

def example_utterance_correction(dataset):
    """Example of using the dataset for AAC utterance correction."""
    print("\n=== Example: AAC Utterance Correction ===")
    
    # Get a few examples from the test set
    examples = dataset['test'].select(range(5))
    
    print("Examples of noisy utterances and their corrections:")
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"Original: {example['utterance']}")
        print(f"Noisy (QWERTY severe): {example['noisy_qwerty_severe']}")
        print(f"Noisy (ABC severe): {example['noisy_abc_severe']}")
        print(f"Minimally corrected: {example['minimally_corrected']}")
        print(f"Fully corrected: {example['fully_corrected']}")

def example_utterance_expansion(dataset):
    """Example of using the dataset for AAC utterance expansion."""
    print("\n=== Example: AAC Utterance Expansion ===")
    
    # Get a few examples from the test set
    examples = dataset['test'].select(range(5))
    
    print("Examples of telegraphic utterances and their expansions:")
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"Telegraphic: {example['utterance']}")
        print(f"Expanded: {example['utterance_intended']}")

def example_response_prediction(dataset):
    """Example of using the dataset for AAC response prediction."""
    print("\n=== Example: AAC Response Prediction ===")
    
    # Get a few examples from the test set
    examples = dataset['test'].select(range(5))
    
    print("Examples of conversation context and responses:")
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}:")
        print("Context:")
        for turn in example['context']:
            print(f"  {turn['speaker']}: {turn['utterance']}")
        print(f"AAC User: {example['utterance']}")
        if example['next_turn']:
            print(f"Response: {example['next_turn']['speaker']}: {example['next_turn']['utterance']}")

def main():
    parser = argparse.ArgumentParser(
        description="Example Usage of AAC Conversations Dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data",
        help="Directory containing the prepared Hugging Face dataset",
    )
    args = parser.parse_args()
    
    # Load the dataset
    dataset = load_dataset(args.data_dir)
    
    # Run examples
    example_utterance_correction(dataset)
    example_utterance_expansion(dataset)
    example_response_prediction(dataset)

if __name__ == "__main__":
    main()
