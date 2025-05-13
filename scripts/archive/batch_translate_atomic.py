#!/usr/bin/env python3
"""
Create a batch file for translating atomic data using OpenAI's batch processing system.
This is a more efficient alternative to the translate_atomic_data.py script.
"""
import json
import argparse
import os
from pathlib import Path
from datetime import datetime

# Constants
BATCH_SIZE = 100  # Number of items per batch request
OUTPUT_DIR = Path("batch_output")
DEFAULT_MODEL = "gpt-3.5-turbo"


def load_atomic_data(input_file, limit=None):
    """Load atomic data from input file."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit and limit < len(data):
        print(f"Limiting to {limit} entries as requested")
        data = data[:limit]

    return data


def create_translation_request(texts, source_lang, target_lang, batch_id, model):
    """Create a batch request for translating a batch of texts."""
    # Prepare the prompt
    prompt = (
        f"Translate the following texts from {source_lang} to {target_lang}. "
        f"Keep the meaning intact but make it sound natural in the target language. "
        f"Return only the translations, one per line, with the same index numbers:\n\n"
    )

    for i, text in enumerate(texts):
        prompt += f"{i+1}. {text}\n"

    return {
        "custom_id": f"translate_{target_lang}_{batch_id}_{datetime.now().timestamp()}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 2000,
        },
    }


def prepare_translation_batch(atomic_data, target_lang, model, field_type="topic"):
    """Prepare batch requests for translating a specific field in the atomic data."""
    batch_requests = []

    # Extract the field values
    field_values = [entry[field_type] for entry in atomic_data]

    # Split into batches
    for i in range(0, len(field_values), BATCH_SIZE):
        batch = field_values[i : i + BATCH_SIZE]
        batch_id = f"{field_type}_{i//BATCH_SIZE}"

        # Create batch request
        request = create_translation_request(batch, "en", target_lang, batch_id, model)
        batch_requests.append(request)

    return batch_requests


def save_batch(batch_requests, target_lang):
    """Save batch requests to a JSONL file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"batch_translate_{target_lang}_{timestamp}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")

    print(f"Saved {len(batch_requests)} batch requests to {output_file}")
    print(f"Total items to translate: {len(batch_requests) * BATCH_SIZE}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Create a batch file for translating atomic data"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="templates/atomic10x/atomic10x_als_subset.json",
        help="Input atomic subset file",
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Target language code (e.g., fr-FR, es-ES)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of entries to translate (for testing)",
    )
    args = parser.parse_args()

    # Load atomic data
    atomic_data = load_atomic_data(args.input, args.limit)
    print(f"Loaded {len(atomic_data)} entries from {args.input}")

    # Prepare batch requests for topic field
    topic_requests = prepare_translation_batch(
        atomic_data, args.lang, args.model, "topic"
    )
    print(f"Created {len(topic_requests)} batch requests for translating topics")

    # Prepare batch requests for which field
    which_requests = prepare_translation_batch(
        atomic_data, args.lang, args.model, "which"
    )
    print(f"Created {len(which_requests)} batch requests for translating which fields")

    # Combine all requests
    all_requests = topic_requests + which_requests

    # Save batch file
    output_file = save_batch(all_requests, args.lang)

    print("\nTo translate the atomic data using OpenAI's batch processing system:")
    print(f"1. Upload the file {output_file} to OpenAI's batch processing system")
    print("2. Download the results when processing is complete")
    print(
        f"3. Run the process_batch_translations.py script (to be created) with the downloaded results"
    )


if __name__ == "__main__":
    main()
