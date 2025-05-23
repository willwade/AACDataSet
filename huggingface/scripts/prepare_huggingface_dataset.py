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
import os
from pathlib import Path
import pandas as pd
from datasets import Dataset

def load_jsonl(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def flatten_conversation_data(conversation_data, lang_code="en"):
    """
    Flatten the conversation data to create a dataset with one row per AAC utterance.
    Each row contains the utterance, its variations, and context information.

    Args:
        conversation_data: The conversation data to flatten
        lang_code: The language code to add to each entry
    """
    flattened_data = []
    conversation_id = 0

    for conv in conversation_data:
        conversation_id += 1
        scene = conv.get('scene', '')
        template_id = conv.get('metadata', {}).get('template_id', -1)

        # Extract conversation turns
        conversation = conv.get('conversation', [])

        # Process each turn in the conversation
        for i, turn in enumerate(conversation):
            # Use the explicit is_aac_user field if present
            if turn.get('is_aac_user', False):
                # Get context (previous turns)
                context_speakers = []
                context_utterances = []
                for j in range(max(0, i-3), i):
                    if j >= 0 and j < len(conversation):
                        context_speakers.append(conversation[j].get('speaker', ''))
                        context_utterances.append(conversation[j].get('utterance', ''))

                # Get next turn (if available) for potential response prediction tasks
                next_turn_speaker = ""
                next_turn_utterance = ""
                if i + 1 < len(conversation):
                    next_turn_speaker = conversation[i+1].get('speaker', '')
                    next_turn_utterance = conversation[i+1].get('utterance', '')

                # Extract model and provider information from metadata
                model = conv.get('metadata', {}).get('model', 'unknown')
                provider = conv.get('metadata', {}).get('provider', 'unknown')

                # Create a flattened entry
                entry = {
                    'conversation_id': conversation_id,  # Add conversation ID
                    'turn_number': i,                   # Add turn number
                    'language_code': lang_code,         # Add language code
                    'template_id': template_id,
                    'scene': scene,
                    'context_speakers': context_speakers,
                    'context_utterances': context_utterances,
                    'speaker': turn.get('speaker', ''),
                    'utterance': turn.get('utterance', ''),
                    'utterance_intended': turn.get('utterance_intended', turn.get('utterance', '')),
                    'next_turn_speaker': next_turn_speaker,
                    'next_turn_utterance': next_turn_utterance,
                    'model': model,                    # Add model information
                    'provider': provider,              # Add provider information
                }

                # Add all the noisy variations
                for key in turn:
                    if key.startswith('noisy_') or key in ['minimally_corrected', 'fully_corrected']:
                        entry[key] = turn[key]

                flattened_data.append(entry)

    return flattened_data

def prepare_dataset(data):
    """Prepare the data as a single dataset without splits."""
    return {'dataset': data}

def save_to_huggingface_format(data, output_dir):
    """Save the data in Hugging Face Datasets format (Arrow + CSV)."""
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the dataset in Arrow format
    dataset.save_to_disk(output_dir)
    print(f"Saved Hugging Face Arrow dataset to {output_dir}")

    # Also save as CSV for easy inspection
    csv_path = os.path.join(output_dir, "dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved dataset to {csv_path}")

    return dataset

def find_augmented_files(input_dir="data/output"):
    """Find all augmented conversation files in the specified directory."""
    directory_path = Path(input_dir)
    if not directory_path.exists() or not directory_path.is_dir():
        print(f"Warning: Directory {input_dir} does not exist or is not a directory.")
        return []

    # Look for files matching the pattern augmented_aac_conversations_*.jsonl
    augmented_files = list(directory_path.glob("augmented_aac_conversations_*.jsonl"))
    return augmented_files

def extract_lang_code(filename):
    """Extract language code from filename."""
    lang_code = "en"  # Default language code

    # Try to extract language code from filename
    if "_" in filename and "." in filename:
        # Handle both formats: augmented_aac_conversations_en.jsonl and augmented_aac_conversations_en-GB.jsonl
        parts = filename.split("_")
        if len(parts) > 2:
            lang_code = parts[-1].split(".")[0]  # Get 'en' or 'en-GB'

    return lang_code

def process_file(input_path, all_data=None):
    """Process a single input file and add its data to the combined dataset.

    Args:
        input_path: Path to the input file
        all_data: Dictionary to store all flattened data (modified in-place)
        split_ratio: Ratio for train/validation/test split

    Returns:
        Number of flattened entries processed
    """
    # Extract language code from input filename
    input_filename = input_path.name
    lang_code = extract_lang_code(input_filename)

    # Load and process data
    print(f"Loading data from {input_path}")
    conversation_data = load_jsonl(input_path)
    print(f"Loaded {len(conversation_data)} conversations")

    # Flatten the data with language code
    print(f"Flattening conversation data for language: {lang_code}...")
    flattened_data = flatten_conversation_data(conversation_data, lang_code)
    print(f"Created {len(flattened_data)} flattened entries")

    # Check if we have any data to process
    if len(flattened_data) == 0:
        print(f"Warning: No AAC utterances found in {input_path}. Skipping this file.")
        return 0

    # Add the flattened data to the combined dataset
    if all_data is not None:
        if 'all' not in all_data:
            all_data['all'] = []
        all_data['all'].extend(flattened_data)

    return len(flattened_data)

def main():
    parser = argparse.ArgumentParser(
        description="Prepare AAC Conversations Dataset for Hugging Face"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input JSONL file with augmented AAC conversations. If not provided, will process all files in the output directory.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/output",
        help="Directory to search for augmented conversation files when processing all languages.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="huggingface/data",
        help="Output directory for the combined Hugging Face dataset.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="all",
        help="Language code to process (e.g., 'en-GB', 'fr-FR'). Use 'all' to process all available languages.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="aac_dataset",
        help="Name of the dataset (used for the output directory).",
    )
    parser.add_argument(
        "--split_ratio",
        type=str,
        default="0.8,0.2",
        help="Comma-separated split ratio for train and test sets (default: 0.8,0.2)",
    )
    args = parser.parse_args()

    # Create the output directory
    output_dir = Path(args.output_dir) / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to store all flattened data
    all_data = {'all': []}

    # Process all files if no input is provided or if lang=all
    if args.input is None or args.lang == "all":
        # Find all augmented conversation files
        augmented_files = find_augmented_files(args.input_dir)
        if not augmented_files:
            print(f"No augmented conversation files found in {args.input_dir}. Please check the directory or provide a specific input file.")
            return

        print(f"Found {len(augmented_files)} augmented conversation files to process:")
        for file in augmented_files:
            print(f"  - {file.name}")

        total_entries = 0
        for input_file in augmented_files:
            print(f"\n{'='*50}\nProcessing file: {input_file}\n{'='*50}")
            entries_count = process_file(input_file, all_data)
            total_entries += entries_count

        print(f"\nCompleted processing all files. Total entries: {total_entries}.")
    else:
        # Process a single file
        input_path = Path(args.input) if args.input else None

        # If lang is specified but input is not, look for a file with that language code
        if input_path is None and args.lang != "all":
            lang_pattern = f"augmented_aac_conversations_{args.lang}.jsonl"
            potential_files = list(Path(args.input_dir).glob(lang_pattern))
            if potential_files:
                input_path = potential_files[0]
                print(f"Found file for language {args.lang}: {input_path}")
            else:
                print(f"Error: No file found for language {args.lang} in {args.input_dir}.")
                return

        if not input_path or not input_path.exists():
            print(f"Error: Input file {args.input} does not exist.")
            return

        process_file(input_path, all_data)

    # Check if we have any data to process
    if not all_data['all']:
        print("No data to process. Exiting.")
        return

    print(f"\nCombined dataset has {len(all_data['all'])} entries.")

    # Split the dataset
    split_ratio = [float(x) for x in args.split_ratio.split(",")]
    if len(split_ratio) != 2 or not 0 < sum(split_ratio) <= 1.0:
        raise ValueError("split_ratio must be two floats adding up to 1.0 or less (e.g., 0.8,0.2)")
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(all_data['all'], test_size=split_ratio[1], random_state=42)

    print(f"\nSplit: {len(train_data)} train, {len(test_data)} test")

    # Save in Hugging Face format
    print(f"Saving train split to {output_dir}/train")
    save_to_huggingface_format(train_data, output_dir / "train")
    print(f"Saving test split to {output_dir}/test")
    save_to_huggingface_format(test_data, output_dir / "test")

    print(f"\nDataset statistics: {len(all_data['all'])} examples total")
    print("\nDone!")

if __name__ == "__main__":
    main()
