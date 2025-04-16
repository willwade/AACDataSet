#!/usr/bin/env python
"""
Calculate statistics for the AAC Conversations Dataset.

This script analyzes the augmented conversation files and calculates various statistics:
- Number of conversations per language
- Average conversation length (turns)
- Mean length of utterance (MLU) for AAC users and non-AAC users
- Total utterances by speaker type
- Overall totals across all languages

Usage:
    python calculate_dataset_stats.py --input_dir ../../output
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import re
import numpy as np


def load_jsonl(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def is_aac_user(speaker):
    """Check if the speaker is an AAC user."""
    if not isinstance(speaker, str):
        print(f"Warning: Expected string for speaker but got {type(speaker)}. Assuming not an AAC user.")
        return False
    return any(marker in speaker for marker in ["(AAC)", "AAC", "Speaker-AAC"])


def count_words(text):
    """Count the number of words in a text."""
    if not text:
        return 0

    # Handle case where text is not a string (e.g., it's a dictionary)
    if not isinstance(text, str):
        print(f"Warning: Expected string but got {type(text)}. Using empty string instead.")
        return 0

    # Split by whitespace and count non-empty strings
    return len([word for word in re.split(r'\s+', text) if word])


def calculate_stats_for_file(file_path):
    """Calculate statistics for a single file."""
    print(f"Processing {file_path.name}...")

    # Extract language code from filename
    lang_code = "unknown"
    if "_" in file_path.name and "." in file_path.name:
        parts = file_path.name.split("_")
        if len(parts) > 1:
            lang_code = parts[-1].split(".")[0]

    # Load the data
    try:
        conversations = load_jsonl(file_path)
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}")
        return None

    # Initialize statistics
    stats = {
        "language_code": lang_code,
        "num_conversations": len(conversations),
        "total_turns": 0,
        "aac_utterances": 0,
        "non_aac_utterances": 0,
        "aac_word_count": 0,
        "non_aac_word_count": 0,
        "conversation_lengths": []
    }

    # Process each conversation
    for conv_idx, conv in enumerate(conversations):
        try:
            # Check if conversation has the expected structure
            if not isinstance(conv, dict):
                print(f"Warning: Conversation {conv_idx} in {file_path.name} is not a dictionary. Skipping.")
                continue

            turns = conv.get("conversation", [])
            if not isinstance(turns, list):
                print(f"Warning: Conversation {conv_idx} in {file_path.name} has invalid 'conversation' field. Skipping.")
                continue

            stats["total_turns"] += len(turns)
            stats["conversation_lengths"].append(len(turns))

            # Process each turn
            for turn_idx, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    print(f"Warning: Turn {turn_idx} in conversation {conv_idx} in {file_path.name} is not a dictionary. Skipping.")
                    continue

                speaker = turn.get("speaker", "")
                utterance = turn.get("utterance", "")

                # Count words
                word_count = count_words(utterance)

                # Update statistics based on speaker type
                if is_aac_user(speaker):
                    stats["aac_utterances"] += 1
                    stats["aac_word_count"] += word_count
                else:
                    stats["non_aac_utterances"] += 1
                    stats["non_aac_word_count"] += word_count
        except Exception as e:
            print(f"Error processing conversation {conv_idx} in {file_path.name}: {e}")
            continue

    # Calculate averages
    stats["avg_conversation_length"] = np.mean(stats["conversation_lengths"]) if stats["conversation_lengths"] else 0
    stats["aac_mlu"] = stats["aac_word_count"] / stats["aac_utterances"] if stats["aac_utterances"] > 0 else 0
    stats["non_aac_mlu"] = stats["non_aac_word_count"] / stats["non_aac_utterances"] if stats["non_aac_utterances"] > 0 else 0

    return stats


def find_augmented_files(input_dir):
    """Find all augmented conversation files in the specified directory."""
    directory_path = Path(input_dir)
    if not directory_path.exists() or not directory_path.is_dir():
        print(f"Warning: Directory {input_dir} does not exist or is not a directory.")
        return []

    # Look for files matching the pattern augmented_aac_conversations_*.jsonl
    augmented_files = list(directory_path.glob("augmented_aac_conversations_*.jsonl"))
    return augmented_files


def calculate_all_stats(input_dir):
    """Calculate statistics for all files in the input directory."""
    # Find all augmented files
    files = find_augmented_files(input_dir)
    if not files:
        print(f"No augmented conversation files found in {input_dir}.")
        return None

    print(f"Found {len(files)} files to analyze.")

    # Calculate statistics for each file
    all_stats = []
    for file_path in files:
        stats = calculate_stats_for_file(file_path)
        if stats is not None:
            all_stats.append(stats)

    if not all_stats:
        print("No valid statistics were calculated for any file.")
        return None

    # Create a DataFrame
    stats_df = pd.DataFrame(all_stats)

    # Calculate totals
    totals = {
        "language_code": "TOTAL",
        "num_conversations": stats_df["num_conversations"].sum(),
        "total_turns": stats_df["total_turns"].sum(),
        "aac_utterances": stats_df["aac_utterances"].sum(),
        "non_aac_utterances": stats_df["non_aac_utterances"].sum(),
        "aac_word_count": stats_df["aac_word_count"].sum(),
        "non_aac_word_count": stats_df["non_aac_word_count"].sum(),
        "avg_conversation_length": stats_df["total_turns"].sum() / stats_df["num_conversations"].sum() if stats_df["num_conversations"].sum() > 0 else 0,
        "aac_mlu": stats_df["aac_word_count"].sum() / stats_df["aac_utterances"].sum() if stats_df["aac_utterances"].sum() > 0 else 0,
        "non_aac_mlu": stats_df["non_aac_word_count"].sum() / stats_df["non_aac_utterances"].sum() if stats_df["non_aac_utterances"].sum() > 0 else 0
    }

    # Add totals to the DataFrame
    stats_df = pd.concat([stats_df, pd.DataFrame([totals])], ignore_index=True)

    return stats_df


def format_stats_table(stats_df):
    """Format the statistics DataFrame for display."""
    # Round floating point columns
    for col in ["avg_conversation_length", "aac_mlu", "non_aac_mlu"]:
        stats_df[col] = stats_df[col].round(2)

    # Rename columns for better readability
    stats_df = stats_df.rename(columns={
        "language_code": "Language",
        "num_conversations": "Conversations",
        "total_turns": "Total Turns",
        "aac_utterances": "AAC Utterances",
        "non_aac_utterances": "Non-AAC Utterances",
        "aac_word_count": "AAC Words",
        "non_aac_word_count": "Non-AAC Words",
        "avg_conversation_length": "Avg Turns/Conv",
        "aac_mlu": "AAC MLU",
        "non_aac_mlu": "Non-AAC MLU"
    })

    return stats_df


def save_stats(stats_df, output_dir):
    """Save the statistics to CSV and markdown files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    csv_path = output_path / "dataset_statistics.csv"
    stats_df.to_csv(csv_path, index=False)
    print(f"Statistics saved to {csv_path}")

    # Save as markdown
    md_path = output_path / "dataset_statistics.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# AAC Conversations Dataset Statistics\n\n")
        f.write(stats_df.to_markdown(index=False))
        f.write("\n\n## Notes\n\n")
        f.write("- **MLU**: Mean Length of Utterance (average number of words per utterance)\n")
        f.write("- **AAC Utterances**: Utterances from AAC users\n")
        f.write("- **Non-AAC Utterances**: Utterances from communication partners\n")

    print(f"Statistics saved to {md_path}")

    return csv_path, md_path


def main():
    parser = argparse.ArgumentParser(
        description="Calculate statistics for the AAC Conversations Dataset"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="../../output",
        help="Directory containing the augmented conversation files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../stats",
        help="Directory to save the statistics",
    )
    args = parser.parse_args()

    # Calculate statistics
    stats_df = calculate_all_stats(args.input_dir)

    if stats_df is not None:
        # Format the statistics
        formatted_stats = format_stats_table(stats_df)

        # Display the statistics
        print("\nDataset Statistics:")
        print(formatted_stats)

        # Save the statistics
        save_stats(formatted_stats, args.output_dir)

        print(f"\nStatistics have been saved to {args.output_dir}")
    else:
        print("No statistics were calculated.")


if __name__ == "__main__":
    main()
