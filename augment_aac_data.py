#!/usr/bin/env python3
"""
Augment AAC Conversation Data

This script reads AAC conversation data from a JSONL file and augments each AAC utterance with:
- Noisy utterances for different keyboard layouts (QWERTY, ABC, frequency)
- Minimally and fully corrected versions

Usage:
    python augment_aac_data.py [--input INPUT_FILE] [--output OUTPUT_FILE]
"""

import json
import argparse
import random
import numpy as np
import re
from pathlib import Path
import string

# Import scanning library functions
try:
    from scanning_library import (
        create_abc_grid,
        create_qwerty_grid,
        create_frequency_grid,
    )
except ImportError:
    print("Warning: scanning_library.py not found. Using simplified grid functions.")

    def create_abc_grid(rows, cols, fillers=None):
        """Simplified ABC grid creation"""
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ_")
        return np.array(letters + [""] * (rows * cols - len(letters))).reshape(
            rows, cols
        )

    def create_qwerty_grid(rows, cols, fillers=None):
        """Simplified QWERTY grid creation"""
        qwerty_layout = list("QWERTYUIOPASDFGHJKLZXCVBNM_")
        return np.array(
            qwerty_layout + [""] * (rows * cols - len(qwerty_layout))
        ).reshape(rows, cols)

    def create_frequency_grid(rows, cols, letter_frequencies, fillers=None):
        """Simplified frequency grid creation"""
        sorted_letters = sorted(
            letter_frequencies.keys(), key=lambda x: -letter_frequencies[x]
        )
        sorted_letters = [char if char != " " else "_" for char in sorted_letters]
        return np.array(
            sorted_letters + [""] * (rows * cols - len(sorted_letters))
        ).reshape(rows, cols)


# Define English letter frequencies
ENGLISH_LETTER_FREQUENCIES = {
    "e": 0.1202,
    "t": 0.0910,
    "a": 0.0812,
    "o": 0.0768,
    "i": 0.0731,
    "n": 0.0695,
    "s": 0.0628,
    "r": 0.0602,
    "h": 0.0592,
    "d": 0.0432,
    "l": 0.0398,
    "u": 0.0288,
    "c": 0.0271,
    "m": 0.0261,
    "f": 0.0230,
    "y": 0.0211,
    "w": 0.0209,
    "g": 0.0203,
    "p": 0.0182,
    "b": 0.0149,
    "v": 0.0111,
    "k": 0.0069,
    "x": 0.0017,
    "q": 0.0011,
    "j": 0.0010,
    "z": 0.0007,
    " ": 0.1800,  # Space is the most frequent
}

# Step 1: Define letter frequencies
letter_frequencies = {
    "E": 12.49,
    "T": 9.28,
    "A": 8.04,
    "O": 7.64,
    "I": 7.57,
    "N": 7.23,
    "S": 6.51,
    "R": 6.28,
    "H": 5.05,
    "L": 4.07,
    "D": 3.82,
    "C": 3.34,
    "U": 2.73,
    "M": 2.51,
    "F": 2.40,
    "P": 2.14,
    "G": 1.87,
    "W": 1.68,
    "Y": 1.66,
    "B": 1.48,
    "V": 1.05,
    "K": 0.54,
    "X": 0.23,
    "J": 0.16,
    "Q": 0.12,
    "Z": 0.09,
    "_": 15.00,
}

# Step 2: Create grids
rows, cols = 5, 6
abc_grid = create_abc_grid(rows, cols)
rows, cols = 6, 6
frequency_grid = create_frequency_grid(rows, cols, letter_frequencies)
rows, cols = 4, 10
qwerty_grid = create_qwerty_grid(rows, cols)

# Create keyboard grids
QWERTY_GRID = create_qwerty_grid(3, 10)
ABC_GRID = create_abc_grid(3, 10)
FREQUENCY_GRID = create_frequency_grid(3, 10, ENGLISH_LETTER_FREQUENCIES)


def get_adjacent_keys(grid, char):
    """Get adjacent keys for a character in a grid"""
    char = char.upper() if char.isalpha() else "_" if char == " " else char

    # Find the position of the character in the grid
    positions = np.where(grid == char)
    if len(positions[0]) == 0:
        return []

    row, col = positions[0][0], positions[1][0]
    rows, cols = grid.shape

    # Get adjacent positions
    adjacent_positions = []
    for r in range(max(0, row - 1), min(rows, row + 2)):
        for c in range(max(0, col - 1), min(cols, col + 2)):
            if (r != row or c != col) and grid[r, c] != "":
                adjacent_positions.append((r, c))

    # Return adjacent characters
    return [grid[r, c] for r, c in adjacent_positions]


def generate_noisy_utterance(text, grid, error_rate=0.2, error_types=None):
    """
    Generate a noisy version of the text based on the keyboard grid

    Error types:
    - 'adjacent': Replace with an adjacent key
    - 'deletion': Delete a character
    - 'insertion': Insert a random character
    - 'transposition': Swap adjacent characters
    """
    if error_types is None:
        error_types = ["adjacent", "deletion", "insertion", "transposition"]

    result = ""
    i = 0
    while i < len(text):
        char = text[i]

        # Decide whether to introduce an error
        if random.random() < error_rate:
            error_type = random.choice(error_types)

            if error_type == "adjacent" and char.isalnum() or char == " ":
                # Replace with an adjacent key
                adjacent_keys = get_adjacent_keys(grid, char)
                if adjacent_keys:
                    result += random.choice(adjacent_keys).lower()
                else:
                    result += char

            elif error_type == "deletion":
                # Skip this character (delete)
                pass

            elif error_type == "insertion":
                # Insert a random character
                if char.isalpha():
                    # Get a random character from the grid
                    random_char = random.choice(grid.flatten())
                    if random_char != "":
                        result += random_char.lower()
                result += char

            elif error_type == "transposition" and i < len(text) - 1:
                # Swap with next character
                result += text[i + 1] + char
                i += 1  # Skip the next character since we've used it

            else:
                result += char
        else:
            result += char

        i += 1

    return result


def generate_minimally_corrected(text):
    """Generate a minimally corrected version of the text"""
    # Add basic capitalization
    if text and text[0].isalpha():
        text = text[0].upper() + text[1:]

    # Add periods at the end if missing
    if text and not text.endswith((".", "!", "?")):
        text += "."

    return text


def generate_fully_corrected(text, intended_text):
    """Generate a fully corrected version based on the intended text"""
    # Start with the intended text
    corrected = intended_text

    # Ensure proper capitalization
    if corrected and corrected[0].isalpha():
        corrected = corrected[0].upper() + corrected[1:]

    # Ensure proper punctuation
    if corrected and not corrected.endswith((".", "!", "?")):
        corrected += "."

    return corrected


def process_conversation(conversation_data):
    """Process a conversation to augment AAC utterances"""
    conversation = conversation_data.get("conversation", [])

    for i, turn in enumerate(conversation):
        # Check if this is an AAC user's turn
        is_aac_user = False

        # Check different possible structures
        if "utterance_intended" in turn:
            is_aac_user = True
        elif (
            isinstance(turn.get("utterance"), dict)
            and "utterance_intended" in turn["utterance"]
        ):
            # Handle nested structure
            intended = turn["utterance"]["utterance_intended"]
            actual = turn["utterance"]["utterance"]
            # Restructure to flat format
            turn["utterance_intended"] = intended
            turn["utterance"] = actual
            is_aac_user = True
        elif "Speaker-AAC" in turn.get("speaker", "") or "AAC" in turn.get(
            "speaker", ""
        ):
            is_aac_user = True

        if is_aac_user and "utterance" in turn:
            # Get the utterance and intended text
            utterance = turn["utterance"]
            intended = turn.get("utterance_intended", utterance)

            # Generate noisy utterances
            turn["noisy_utterance"] = generate_noisy_utterance(
                utterance,
                QWERTY_GRID,
                error_rate=0.3,
                error_types=["adjacent", "deletion"],
            )

            turn["noisy_utterance_qwerty"] = generate_noisy_utterance(
                utterance, QWERTY_GRID, error_rate=0.25
            )

            turn["noisy_utterance_abc"] = generate_noisy_utterance(
                utterance, ABC_GRID, error_rate=0.25
            )

            turn["noisy_utterance_frequency"] = generate_noisy_utterance(
                utterance, FREQUENCY_GRID, error_rate=0.25
            )

            # Generate corrected versions
            turn["minimally_corrected"] = generate_minimally_corrected(utterance)
            turn["fully_corrected"] = generate_fully_corrected(utterance, intended)

    return conversation_data


def main():
    parser = argparse.ArgumentParser(
        description="Augment AAC conversation data with noisy utterances"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="output/aac_conversations_en.jsonl",
        help="Input JSONL file with AAC conversations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/augmented_aac_conversations_en.jsonl",
        help="Output JSONL file for augmented data",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process the input file
    print(f"Reading conversations from {input_path}")
    augmented_count = 0

    with open(output_path, "w", encoding="utf-8") as out_file:
        with open(input_path, "r", encoding="utf-8") as in_file:
            for line_num, line in enumerate(in_file, 1):
                try:
                    # Parse the JSON data
                    data = json.loads(line)

                    # Process the conversation
                    augmented_data = process_conversation(data)

                    # Write the augmented data
                    out_file.write(json.dumps(augmented_data) + "\n")
                    augmented_count += 1

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")

    print(f"Augmented {augmented_count} conversations and saved to {output_path}")


if __name__ == "__main__":
    main()
