#!/usr/bin/env python3
"""
Process the results from OpenAI's batch processing system for atomic data translation.
This script takes the batch results and creates the translated atomic data files.
"""
import json
import argparse
import os
from pathlib import Path


def load_batch_results(input_file):
    """Load batch results from input file."""
    with open(input_file, "r", encoding="utf-8") as f:
        # Check if the file is JSONL or JSON
        first_line = f.readline().strip()
        f.seek(0)

        if first_line.startswith("{") and first_line.endswith("}"):
            # JSONL format
            results = [json.loads(line) for line in f if line.strip()]
        else:
            # JSON format
            results = json.load(f)

            # If it's an array, use it directly
            if isinstance(results, list):
                pass
            # If it's an object with a 'data' field, extract the data
            elif isinstance(results, dict) and "data" in results:
                results = results["data"]
            else:
                raise ValueError("Unsupported JSON format")

    return results


def extract_translations(batch_results):
    """Extract translations from batch results."""
    topic_translations = {}
    which_translations = {}

    for result in batch_results:
        # Skip if there's no response
        if "response" not in result or not result["response"]:
            print(f"Warning: No response for request {result.get('id', 'unknown')}")
            continue

        # Parse the response
        try:
            response_data = json.loads(result["response"])
            content = response_data["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing response: {e}")
            continue

        # Extract the custom_id to determine if it's a topic or which field
        custom_id = result.get("custom_id", "")
        if not custom_id.startswith("translate_"):
            print(f"Warning: Unexpected custom_id format: {custom_id}")
            continue

        # Extract the batch information
        parts = custom_id.split("_")
        if len(parts) < 4:
            print(f"Warning: Invalid custom_id format: {custom_id}")
            continue

        lang = parts[1]
        field_type = parts[2]
        batch_index = int(parts[3])

        # Parse the translations
        translations = []
        for line in content.strip().split("\n"):
            if line and line[0].isdigit() and ". " in line:
                # Extract the translation part after the index
                translations.append(line.split(". ", 1)[1])

        # Store the translations
        if field_type == "topic":
            topic_translations[batch_index] = translations
        elif field_type == "which":
            which_translations[batch_index] = translations

    return lang, topic_translations, which_translations


def create_translated_file(
    input_file, output_file, topic_translations, which_translations
):
    """Create the translated atomic data file."""
    # Load the original atomic data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Flatten the translations
    all_topic_translations = []
    all_which_translations = []

    # Sort the batches by index
    topic_batch_indices = sorted(topic_translations.keys())
    which_batch_indices = sorted(which_translations.keys())

    for batch_index in topic_batch_indices:
        all_topic_translations.extend(topic_translations[batch_index])

    for batch_index in which_batch_indices:
        all_which_translations.extend(which_translations[batch_index])

    # Create the translated data
    translated_data = []
    for i, entry in enumerate(data):
        translated_entry = entry.copy()

        # Add translated topic if available
        if i < len(all_topic_translations):
            translated_entry["topic"] = all_topic_translations[i]

        # Add translated which field if available
        if i < len(all_which_translations):
            translated_entry["which"] = all_which_translations[i]

        translated_data.append(translated_entry)

    # Save the translated data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(translated_data)} translated entries to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process batch translation results")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input batch results file from OpenAI's batch processing system",
    )
    parser.add_argument(
        "--atomic",
        type=str,
        default="templates/atomic10x/atomic10x_als_subset.json",
        help="Original atomic subset file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file (default: templates/atomic10x/atomic10x_als_subset_{lang}.json)",
    )
    args = parser.parse_args()

    # Load batch results
    print(f"Loading batch results from {args.input}")
    batch_results = load_batch_results(args.input)
    print(f"Loaded {len(batch_results)} batch results")

    # Extract translations
    lang, topic_translations, which_translations = extract_translations(batch_results)

    # Count the total number of translations
    topic_count = sum(len(batch) for batch in topic_translations.values())
    which_count = sum(len(batch) for batch in which_translations.values())

    print(
        f"Extracted {topic_count} topic translations and {which_count} which field translations for language {lang}"
    )

    # Determine output file
    if not args.output:
        args.output = f"templates/atomic10x/atomic10x_als_subset_{lang}.json"

    # Create the translated file
    create_translated_file(
        args.atomic, args.output, topic_translations, which_translations
    )


if __name__ == "__main__":
    main()
