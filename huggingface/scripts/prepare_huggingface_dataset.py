#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare AAC Conversations Dataset for Hugging Face

This script converts the augmented AAC conversations from JSONL format to a format
suitable for uploading to Hugging Face Datasets. It splits the data into train,
validation, and test sets.

Usage:
    python prepare_huggingface_dataset.py [--input INPUT_FILE] [--output_dir OUTPUT_DIR] [--split_ratio SPLIT_RATIO]

Example:
    python prepare_huggingface_dataset.py --input ../../output/augmented_aac_conversations_en.jsonl --output_dir ../data --split_ratio 0.8,0.1,0.1
"""

import json
import argparse
import random
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import datasets
from datasets import Dataset, DatasetDict

def load_jsonl(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def flatten_conversation_data(conversation_data):
    """
    Flatten the conversation data to create a dataset with one row per AAC utterance.
    Each row contains the utterance, its variations, and context information.
    """
    flattened_data = []
    
    for conv in conversation_data:
        scene = conv.get('scene', '')
        template_id = conv.get('metadata', {}).get('template_id', -1)
        
        # Extract conversation turns
        conversation = conv.get('conversation', [])
        
        # Process each turn in the conversation
        for i, turn in enumerate(conversation):
            # Only process AAC user turns
            if 'speaker' in turn and '(AAC)' in turn.get('speaker', ''):
                # Get context (previous turns)
                context = []
                for j in range(max(0, i-3), i):
                    if j >= 0 and j < len(conversation):
                        context.append({
                            'speaker': conversation[j].get('speaker', ''),
                            'utterance': conversation[j].get('utterance', '')
                        })
                
                # Get next turn (if available) for potential response prediction tasks
                next_turn = None
                if i + 1 < len(conversation):
                    next_turn = {
                        'speaker': conversation[i+1].get('speaker', ''),
                        'utterance': conversation[i+1].get('utterance', '')
                    }
                
                # Create a flattened entry
                entry = {
                    'template_id': template_id,
                    'scene': scene,
                    'context': context,
                    'speaker': turn.get('speaker', ''),
                    'utterance': turn.get('utterance', ''),
                    'utterance_intended': turn.get('utterance_intended', ''),
                    'next_turn': next_turn,
                }
                
                # Add all the noisy variations
                for key in turn:
                    if key.startswith('noisy_') or key in ['minimally_corrected', 'fully_corrected']:
                        entry[key] = turn[key]
                
                flattened_data.append(entry)
    
    return flattened_data

def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """Split data into train, validation, and test sets."""
    # Ensure ratios sum to 1
    total = train_ratio + val_ratio + test_ratio
    train_ratio = train_ratio / total
    val_ratio = val_ratio / total
    test_ratio = test_ratio / total
    
    # First split: train and temp (val+test)
    train_data, temp_data = train_test_split(
        data, train_size=train_ratio, random_state=random_seed
    )
    
    # Second split: val and test from temp
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_data, test_data = train_test_split(
        temp_data, train_size=val_ratio_adjusted, random_state=random_seed
    )
    
    return {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }

def save_to_huggingface_format(data_splits, output_dir):
    """Save the data splits in Hugging Face Datasets format."""
    # Convert to Hugging Face Dataset objects
    dataset_dict = DatasetDict({
        split: Dataset.from_pandas(pd.DataFrame(data))
        for split, data in data_splits.items()
    })
    
    # Save the dataset
    dataset_dict.save_to_disk(output_dir)
    
    # Also save as CSV for easy inspection
    for split, data in data_splits.items():
        df = pd.DataFrame(data)
        csv_path = os.path.join(output_dir, f"{split}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {split} set to {csv_path}")
    
    return dataset_dict

def main():
    parser = argparse.ArgumentParser(
        description="Prepare AAC Conversations Dataset for Hugging Face"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../../output/augmented_aac_conversations_en.jsonl",
        help="Input JSONL file with augmented AAC conversations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data",
        help="Output directory for the Hugging Face dataset",
    )
    parser.add_argument(
        "--split_ratio",
        type=str,
        default="0.8,0.1,0.1",
        help="Ratio for train,validation,test split (comma-separated)",
    )
    args = parser.parse_args()
    
    # Parse split ratio
    train_ratio, val_ratio, test_ratio = map(float, args.split_ratio.split(','))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and process data
    print(f"Loading data from {args.input}")
    conversation_data = load_jsonl(args.input)
    print(f"Loaded {len(conversation_data)} conversations")
    
    # Flatten the data
    print("Flattening conversation data...")
    flattened_data = flatten_conversation_data(conversation_data)
    print(f"Created {len(flattened_data)} flattened entries")
    
    # Split the data
    print(f"Splitting data with ratio {train_ratio}:{val_ratio}:{test_ratio}")
    data_splits = split_data(
        flattened_data, 
        train_ratio=train_ratio, 
        val_ratio=val_ratio, 
        test_ratio=test_ratio
    )
    
    # Save in Hugging Face format
    print(f"Saving data to {output_dir}")
    dataset = save_to_huggingface_format(data_splits, output_dir)
    
    # Print dataset info
    print("\nDataset statistics:")
    for split, data in data_splits.items():
        print(f"  {split}: {len(data)} examples")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
