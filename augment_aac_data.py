# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas"
# ]
# ///
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

# Import language-specific keyboard layouts
try:
    print("Attempting to import language_keyboards...")
    from language_keyboards import (
        get_keyboard_layout,
        get_letter_frequencies,
        create_language_abc_grid,
        create_language_qwerty_grid,
        create_language_frequency_grid,
        LANGUAGE_NAMES
    )
    print("Successfully imported language_keyboards")
    LANGUAGE_SUPPORT = True
except ImportError as e:
    print(f"Import error details: {e}")
    print("Warning: language_keyboards.py not found. Using English keyboard layouts only.")
    LANGUAGE_SUPPORT = False


# Import scanning library functions
try:
    print("Attempting to import scanning_library...")
    from scanning_library import (
        create_abc_grid,
        create_qwerty_grid,
        create_frequency_grid,
    )
    print("Successfully imported scanning_library")
except ImportError as e:
    print(f"Import error details: {e}")
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

# Default keyboard grids (will be replaced with language-specific grids)
QWERTY_GRID = create_qwerty_grid(3, 10)
ABC_GRID = create_abc_grid(3, 10)
FREQUENCY_GRID = create_frequency_grid(3, 10, ENGLISH_LETTER_FREQUENCIES)

# Function to create language-specific keyboard grids
def create_language_grids(lang_code):
    """Create keyboard grids for a specific language."""
    if LANGUAGE_SUPPORT:
        # Use language-specific grid creation functions
        qwerty_grid = create_language_qwerty_grid(lang_code, 3, 10)
        abc_grid = create_language_abc_grid(lang_code, 3, 10)
        frequency_grid = create_language_frequency_grid(lang_code, 3, 10)
    else:
        # Fall back to default English grids
        qwerty_grid = create_qwerty_grid(3, 10)
        abc_grid = create_abc_grid(3, 10)
        frequency_grid = create_frequency_grid(3, 10, ENGLISH_LETTER_FREQUENCIES)

    return qwerty_grid, abc_grid, frequency_grid


def get_adjacent_keys(grid, char):
    """Get adjacent keys for a character in a grid"""
    # Handle different character types
    if char.isalpha():
        char = char.upper()
    elif char == " ":
        char = "_"

    # Find the position of the character in the grid
    try:
        positions = np.where(grid == char)
        if len(positions[0]) == 0:
            # Character not found in grid, try to find a similar character
            # For non-Latin scripts, this is important as there might be variations
            # of the same character (e.g., different forms in Arabic)
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
    except Exception as e:
        # If there's any error, return an empty list
        # This is safer than crashing when processing non-Latin scripts
        print(f"Warning: Error finding adjacent keys for '{char}': {e}")
        return []


def generate_noisy_utterance(text, grid, error_rate=0.2, error_types=None, preserve_case=False):
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

    # Safety check for empty text
    if not text:
        return text

    try:
        result = ""
        i = 0
        while i < len(text):
            char = text[i]
            is_upper = char.isupper() if hasattr(char, 'isupper') else False

            # Decide whether to introduce an error
            if random.random() < error_rate:
                error_type = random.choice(error_types)

                if error_type == "adjacent":
                    # Replace with an adjacent key
                    adjacent_keys = get_adjacent_keys(grid, char)
                    if adjacent_keys:
                        replacement = random.choice(adjacent_keys)
                        # Preserve case if requested and possible
                        if preserve_case and is_upper and hasattr(replacement, 'isalpha') and replacement.isalpha():
                            result += replacement
                        elif hasattr(replacement, 'lower') and replacement.isalpha():
                            result += replacement.lower()
                        else:
                            result += replacement
                    else:
                        result += char

                elif error_type == "deletion":
                    # Skip this character (delete)
                    pass

                elif error_type == "insertion":
                    # Insert a random character
                    # Get a random character from the grid
                    non_empty_cells = [c for c in grid.flatten() if c != ""]
                    if non_empty_cells:
                        random_char = random.choice(non_empty_cells)
                        if random_char != "":
                            if preserve_case and is_upper and hasattr(random_char, 'isalpha') and random_char.isalpha():
                                result += random_char
                            elif hasattr(random_char, 'lower') and random_char.isalpha():
                                result += random_char.lower()
                            else:
                                result += random_char
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
    except Exception as e:
        # If there's any error, return the original text
        # This is safer than crashing when processing non-Latin scripts
        print(f"Warning: Error generating noisy utterance for '{text[:20]}...': {e}")
        return text


def generate_minimally_corrected(text):
    """Generate a minimally corrected version of the text"""
    # Add basic capitalization
    if text and text[0].isalpha():
        text = text[0].upper() + text[1:]

    # Add periods at the end if missing
    if text and not text.endswith((".", "!", "?")):
        text += "."

    return text


def generate_fully_corrected(_, intended_text):
    """Generate a fully corrected version based on the intended text

    Args:
        _: Original text (unused, but kept for consistent function signature)
        intended_text: The intended text to use as the basis for correction
    """
    # Start with the intended text
    corrected = intended_text

    # Ensure proper capitalization
    if corrected and corrected[0].isalpha():
        corrected = corrected[0].upper() + corrected[1:]

    # Ensure proper punctuation
    if corrected and not corrected.endswith((".", "!", "?")):
        corrected += "."

    return corrected


def process_conversation(conversation_data, qwerty_grid, abc_grid, frequency_grid):
    """Process a conversation to augment AAC utterances"""
    conversation = conversation_data.get("conversation", [])

    # Define error rate ranges
    ERROR_RATES = {
        "minimal": 0.05,    # 5% errors - very mild typing issues
        "light": 0.15,      # 15% errors - noticeable but clearly readable
        "moderate": 0.25,   # 25% errors - challenging but comprehensible
        "severe": 0.35      # 35% errors - significant difficulty
    }

    for turn in conversation:
        # Use the explicit is_aac_user field if present
        if turn.get("is_aac_user", False):
            utterance = turn["utterance"]
            intended = turn["utterance_intended"]

            # Generate variations with different error rates
            for severity, error_rate in ERROR_RATES.items():
                # QWERTY variations
                turn[f"noisy_qwerty_{severity}"] = generate_noisy_utterance(
                    utterance,
                    qwerty_grid,
                    error_rate=error_rate,
                    error_types=["adjacent", "deletion", "insertion"]
                )

                # ABC variations
                turn[f"noisy_abc_{severity}"] = generate_noisy_utterance(
                    utterance,
                    abc_grid,
                    error_rate=error_rate,
                    error_types=["adjacent", "deletion", "insertion"]
                )

                # Frequency variations
                turn[f"noisy_frequency_{severity}"] = generate_noisy_utterance(
                    utterance,
                    frequency_grid,
                    error_rate=error_rate,
                    error_types=["adjacent", "deletion", "insertion"]
                )

            # Generate corrected versions
            turn["minimally_corrected"] = generate_minimally_corrected(utterance)
            turn["fully_corrected"] = generate_fully_corrected(utterance, intended)

    return conversation_data


def process_file(input_path, output_path=None, lang_code=None):
    """Process a single input file and generate augmented data."""
    # Extract language code from input filename if not provided
    if lang_code is None:
        input_filename = input_path.name
        # Handle both formats: aac_conversations_en.jsonl and aac_conversations_en-GB.jsonl
        if "_" in input_filename and "." in input_filename:
            # Try to extract language code from filename
            parts = input_filename.split("_")
            if len(parts) > 1:
                lang_part = parts[-1].split(".")[0]  # Get 'en' or 'en-GB'
                lang_code = lang_part
            else:
                # Default to English if can't extract language code
                lang_code = "en"
        else:
            # Default to English if can't extract language code
            lang_code = "en"

    print(f"Using language code: {lang_code}")
    if LANGUAGE_SUPPORT and lang_code in LANGUAGE_NAMES:
        print(f"Language: {LANGUAGE_NAMES[lang_code]}")

    # Create language-specific keyboard grids
    qwerty_grid, abc_grid, frequency_grid = create_language_grids(lang_code)

    # Print grid information
    print(f"Created keyboard grids for language: {lang_code}")
    print(f"QWERTY grid shape: {qwerty_grid.shape}")
    print(f"ABC grid shape: {abc_grid.shape}")
    print(f"Frequency grid shape: {frequency_grid.shape}")

    # If output path is not provided, generate it from input path
    if output_path is None:
        # Extract the prefix and original language code from the input filename
        input_filename = input_path.name
        if "_" in input_filename and "." in input_filename:
            # Preserve the original language code in the output filename
            parts = input_filename.split("_")
            if len(parts) > 1:
                prefix = parts[0]  # Get 'aac'
                # Use the original language code from the filename
                original_lang = parts[-1].split(".")[0]  # Get 'en' or 'en-GB'
                output_filename = f"augmented_{prefix}_conversations_{original_lang}.jsonl"
            else:
                # Fallback if the filename doesn't match the expected pattern
                output_filename = f"augmented_{input_filename}"
        else:
            # Fallback if the filename doesn't match the expected pattern
            output_filename = f"augmented_{input_filename}"

        output_path = input_path.parent / output_filename
    else:
        output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process the input file
    print(f"Reading conversations from {input_path}")
    augmented_count = 0
    aac_utterances_count = 0

    with open(output_path, "w", encoding="utf-8") as out_file:
        with open(input_path, "r", encoding="utf-8") as in_file:
            for line_num, line in enumerate(in_file, 1):
                try:
                    # Parse the JSON data
                    data = json.loads(line)

                    # We'll count AAC utterances after processing

                    # Process the conversation with language-specific grids
                    augmented_data = process_conversation(data, qwerty_grid, abc_grid, frequency_grid)

                    # Count AAC utterances after processing
                    post_count = sum(1 for turn in augmented_data.get("conversation", [])
                                   if "noisy_qwerty_minimal" in turn)  # If this field exists, it was augmented

                    aac_utterances_count += post_count

                    # Write the augmented data
                    out_file.write(json.dumps(augmented_data, ensure_ascii=False) + "\n")
                    augmented_count += 1

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")

    print(f"Augmented {augmented_count} conversations containing {aac_utterances_count} AAC utterances")
    print(f"Saved to {output_path}")
    return augmented_count, aac_utterances_count

def find_conversation_files(directory="output"):
    """Find all conversation files in the specified directory."""
    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        print(f"Warning: Directory {directory} does not exist or is not a directory.")
        return []

    # Look for files matching the pattern aac_conversations_*.jsonl
    conversation_files = list(directory_path.glob("aac_conversations_*.jsonl"))
    return conversation_files

def main():
    parser = argparse.ArgumentParser(
        description="Augment AAC conversation data with noisy utterances"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input JSONL file with AAC conversations. If not provided, will process all files in the output directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file for augmented data. If not provided, will be automatically generated from input filename.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Language code to use for keyboard layouts (e.g., 'en-GB', 'fr-FR'). Use 'all' to process all available languages. If not provided, will be extracted from the input filename.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="output",
        help="Directory to search for conversation files when processing all languages.",
    )
    args = parser.parse_args()

    # Process all files if no input is provided or if lang=all
    if args.input is None or args.lang == "all":
        # Find all conversation files
        conversation_files = find_conversation_files(args.dir)
        if not conversation_files:
            print(f"No conversation files found in {args.dir}. Please check the directory or provide a specific input file.")
            return

        print(f"Found {len(conversation_files)} conversation files to process:")
        for file in conversation_files:
            print(f"  - {file.name}")

        total_conversations = 0
        total_utterances = 0
        for input_file in conversation_files:
            print(f"\n{'='*50}\nProcessing file: {input_file}\n{'='*50}")
            conversations, utterances = process_file(input_file)
            total_conversations += conversations
            total_utterances += utterances

        print(f"\nCompleted processing all files.")
        print(f"Total augmented: {total_conversations} conversations containing {total_utterances} AAC utterances.")
    else:
        # Process a single file
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file {args.input} does not exist.")
            return

        conversations, utterances = process_file(input_path, args.output, args.lang)
        print(f"\nCompleted processing file: {input_path}")
        print(f"Augmented {conversations} conversations containing {utterances} AAC utterances.")


if __name__ == "__main__":
    main()
