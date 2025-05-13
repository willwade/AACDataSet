#!/usr/bin/env python3
"""
Process the results from OpenAI's batch processing system for unique atomic data translation.
This script takes the batch results and mapping file to create the translated atomic data files.
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


def load_mapping_file(mapping_file):
    """Load mapping file."""
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    return mapping


def load_atomic_data(input_file):
    """Load original atomic data."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def extract_translations(batch_results):
    """Extract translations from batch results."""
    topic_translations = {}
    which_translations = {}
    target_lang = None

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

        # Extract language and field type
        if target_lang is None:
            target_lang = parts[1]

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

    return target_lang, topic_translations, which_translations


def create_translation_lookup(mapping, topic_translations, which_translations):
    """Create lookup dictionaries for translations."""
    topic_lookup = {}
    which_lookup = {}

    # Process topic translations
    for batch_index, translations in topic_translations.items():
        start_index = batch_index * 100  # Assuming batch size of 100
        for i, translation in enumerate(translations):
            orig_index = str(start_index + i)
            if orig_index in mapping["topics"]:
                original = mapping["topics"][orig_index]
                topic_lookup[original] = translation

    # Process which translations
    for batch_index, translations in which_translations.items():
        start_index = batch_index * 100  # Assuming batch size of 100
        for i, translation in enumerate(translations):
            orig_index = str(start_index + i)
            if orig_index in mapping["which"]:
                original = mapping["which"][orig_index]
                which_lookup[original] = translation

    return topic_lookup, which_lookup


def create_translated_file(atomic_data, output_file, topic_lookup, which_lookup):
    """Create the translated atomic data file."""
    translated_data = []

    # Track statistics
    total_entries = len(atomic_data)
    translated_topics = 0
    translated_which = 0

    # Create translated entries
    for entry in atomic_data:
        translated_entry = entry.copy()

        # Translate topic
        if entry["topic"] in topic_lookup:
            translated_entry["topic"] = topic_lookup[entry["topic"]]
            translated_topics += 1

        # Translate which
        if entry["which"] in which_lookup:
            translated_entry["which"] = which_lookup[entry["which"]]
            translated_which += 1

        translated_data.append(translated_entry)

    # Save the translated data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(translated_data)} translated entries to {output_file}")
    print(
        f"Successfully translated {translated_topics}/{total_entries} topics ({translated_topics/total_entries*100:.1f}%)"
    )
    print(
        f"Successfully translated {translated_which}/{total_entries} which fields ({translated_which/total_entries*100:.1f}%)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Process batch unique translation results"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input batch results file from OpenAI's batch processing system",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        required=True,
        help="Mapping file created by batch_translate_unique_atomic.py",
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

    # Load mapping file
    print(f"Loading mapping file from {args.mapping}")
    mapping = load_mapping_file(args.mapping)
    print(
        f"Loaded mapping with {len(mapping['topics'])} topics and {len(mapping['which'])} which fields"
    )

    # Load original atomic data
    print(f"Loading atomic data from {args.atomic}")
    atomic_data = load_atomic_data(args.atomic)
    print(f"Loaded {len(atomic_data)} atomic entries")

    # Extract translations
    target_lang, topic_translations, which_translations = extract_translations(
        batch_results
    )

    # Count the total number of translations
    topic_count = sum(len(batch) for batch in topic_translations.values())
    which_count = sum(len(batch) for batch in which_translations.values())

    print(
        f"Extracted {topic_count} topic translations and {which_count} which field translations for language {target_lang}"
    )

    # Create translation lookup dictionaries
    topic_lookup, which_lookup = create_translation_lookup(
        mapping, topic_translations, which_translations
    )
    print(
        f"Created lookup dictionaries with {len(topic_lookup)} topics and {len(which_lookup)} which fields"
    )

    # Determine output file
    if not args.output:
        args.output = f"templates/atomic10x/atomic10x_als_subset_{target_lang}.json"

    # Create the translated file
    create_translated_file(atomic_data, args.output, topic_lookup, which_lookup)


if __name__ == "__main__":
    main()
