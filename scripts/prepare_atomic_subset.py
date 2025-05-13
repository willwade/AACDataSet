#!/usr/bin/env python3
"""
Prepare a subset of the ATOMIC2020 dataset for AAC conversations.
This script filters the ATOMIC2020 dataset to create a subset relevant to AAC users with ALS.
"""
import csv
import json
import argparse
from pathlib import Path

# --- Settings ---
INPUT_FILES = [
    Path("atomic2020_data-feb2021/train.tsv"),
    Path("atomic2020_data-feb2021/dev.tsv"),
    Path("atomic2020_data-feb2021/test.tsv"),
]
OUTPUT_FILE = Path("templates/atomic10x/atomic10x_als_subset.json")

ALLOWED_RELATIONS = {
    "xNeed",
    "xIntent",
    "xEffect",
    "xReact",
    "oEffect",
    "oReact",
    "isAfter",
    "isBefore",
    "HinderedBy",
}

ALLOWED_KEYWORDS = [
    "eat",
    "drink",
    "pain",
    "sleep",
    "help",
    "talk",
    "communicate",
    "call",
    "move",
    "comfort",
    "care",
    "rest",
    "doctor",
    "nurse",
    "hospital",
    "medicine",
    "treatment",
    "support",
    "assist",
    "feel",
    "need",
    "want",
    "ask",
    "tell",
    "say",
    "speak",
    "listen",
    "hear",
    "see",
    "look",
    "watch",
    "read",
    "write",
    "type",
    "text",
    "message",
    "email",
    "phone",
    "call",
    "visit",
    "meet",
    "greet",
    "thank",
    "apologize",
    "complain",
    "request",
    "order",
    "buy",
    "pay",
    "spend",
    "save",
    "work",
    "play",
    "relax",
    "rest",
    "sleep",
    "wake",
    "dress",
    "undress",
    "wash",
    "clean",
    "cook",
    "bake",
    "prepare",
    "serve",
    "share",
    "give",
    "take",
    "receive",
    "accept",
    "reject",
    "refuse",
    "agree",
    "disagree",
    "argue",
    "fight",
    "make up",
    "forgive",
    "forget",
    "remember",
    "think",
    "believe",
    "know",
    "understand",
    "learn",
    "teach",
    "study",
    "practice",
    "improve",
    "succeed",
    "fail",
    "try",
    "attempt",
    "start",
    "stop",
    "continue",
    "finish",
    "complete",
    "begin",
    "end",
]


def event_matches(event_text):
    """
    Check if an event text contains any of the allowed keywords.

    Args:
        event_text: The event text to check

    Returns:
        True if the event text contains any of the allowed keywords, False otherwise
    """
    event_text = event_text.lower()
    return any(keyword in event_text for keyword in ALLOWED_KEYWORDS)


def load_tsv_entries(file_path):
    """
    Load entries from a TSV file.

    Args:
        file_path: Path to the TSV file

    Returns:
        List of entries
    """
    entries = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) == 3:
                head, relation, tail = row
                entries.append(
                    {
                        "head": head.strip(),
                        "relation": relation.strip(),
                        "tail": tail.strip(),
                    }
                )
    return entries


def prepare_atomic_subset(entries, limit=None):
    """
    Prepare a subset of the ATOMIC2020 dataset for AAC conversations.

    Args:
        entries: List of entries from the ATOMIC2020 dataset
        limit: Optional limit on the number of entries to include

    Returns:
        List of filtered entries
    """
    output = []
    for entry in entries:
        if entry["relation"] not in ALLOWED_RELATIONS:
            continue
        if not event_matches(entry["head"]):
            continue
        if entry["tail"].lower() == "none":
            continue  # skip useless entries
        output.append(
            {
                "topic": entry["head"],
                "relation": entry["relation"],
                "aac_user": "Alex",
                "partner": "Taylor",
                "aac_user_role": "person with ALS",
                "which": entry["tail"],
            }
        )

        # Break if we've reached the limit
        if limit and len(output) >= limit:
            break

    return output


def save_output(entries, path):
    """
    Save entries to a JSON file.

    Args:
        entries: List of entries to save
        path: Path to the output file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    print(f"Saved {len(entries)} ALS-relevant entries to {path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Prepare ATOMIC subset for AAC conversations"
    )
    parser.add_argument(
        "--limit", type=int, help="Limit the number of entries to include"
    )
    parser.add_argument("--output", type=str, help="Output file path")
    args = parser.parse_args()

    # Determine output file
    output_file = Path(args.output) if args.output else OUTPUT_FILE

    # Load entries
    all_entries = []
    for input_file in INPUT_FILES:
        if input_file.exists():
            print(f"Loading {input_file}...")
            all_entries.extend(load_tsv_entries(input_file))
        else:
            print(f"Warning: {input_file} not found.")
    print(f"Loaded {len(all_entries)} total entries.")

    # Filter entries
    filtered_entries = prepare_atomic_subset(all_entries, args.limit)

    # Save output
    save_output(filtered_entries, output_file)


if __name__ == "__main__":
    main()
