#!/usr/bin/env python3
"""
Translate an entire JSON file while preserving its structure using OpenAI's API.
This script uses OpenAI's API to translate only the text values in a JSON file.
"""
import json
import argparse
import os
from pathlib import Path
from datetime import datetime
import time
import openai
import sys
from tqdm import tqdm

# Constants
OUTPUT_DIR = Path("templates")
DEFAULT_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 4096  # Maximum tokens for a single API call


def load_json_file(input_file):
    """Load JSON data from input file."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def chunk_data(data, chunk_size=100):
    """Split data into chunks for processing."""
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def translate_json_chunk(
    chunk, target_lang, model, fields_to_translate=None, fields_to_skip=None
):
    """
    Translate a chunk of JSON data using OpenAI's API.

    Args:
        chunk: A chunk of JSON data to translate
        target_lang: Target language code
        model: OpenAI model to use
        fields_to_translate: List of field names to translate
        fields_to_skip: List of field names to skip

    Returns:
        Translated JSON chunk
    """
    # Default fields to translate if not specified
    if fields_to_translate is None:
        fields_to_translate = ["topic", "which"]

    # Default fields to skip if not specified
    if fields_to_skip is None:
        fields_to_skip = ["aac_user", "partner", "aac_user_role", "relation"]

    # Prepare the JSON data for translation
    # Extract only the fields we want to translate
    translation_data = []
    for item in chunk:
        item_data = {}
        for field in fields_to_translate:
            if field in item and field not in fields_to_skip:
                item_data[field] = item[field]
        translation_data.append(item_data)

    # Create the prompt
    prompt = f"""Translate the following JSON data from English to {target_lang}. 
Only translate the text values, not the keys. 
Return the translated JSON data in the same format.
Keep special placeholders like 'PersonX', 'PersonY', and '___' unchanged.

JSON data:
{json.dumps(translation_data, ensure_ascii=False)}

Translated JSON data:"""

    # Call OpenAI API
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=MAX_TOKENS,
        )

        # Extract the translated JSON
        translated_text = response.choices[0].message.content.strip()

        # Parse the translated JSON
        # Find the JSON part in the response (it might have additional text)
        json_start = translated_text.find("[")
        json_end = translated_text.rfind("]") + 1

        if json_start >= 0 and json_end > json_start:
            json_text = translated_text[json_start:json_end]
            translated_chunk = json.loads(json_text)
        else:
            # Try to parse the entire response as JSON
            translated_chunk = json.loads(translated_text)

        # Merge the translated fields back into the original data
        result = []
        for i, item in enumerate(chunk):
            new_item = item.copy()
            if i < len(translated_chunk):
                for field in fields_to_translate:
                    if field in translated_chunk[i] and field not in fields_to_skip:
                        new_item[field] = translated_chunk[i][field]
            result.append(new_item)

        return result

    except Exception as e:
        print(f"Error translating chunk: {e}")
        print(
            f"Response: {translated_text if 'translated_text' in locals() else 'No response'}"
        )
        return chunk  # Return the original chunk on error


def translate_json_file(
    data,
    target_lang,
    model,
    fields_to_translate=None,
    fields_to_skip=None,
    chunk_size=100,
):
    """
    Translate a JSON file using OpenAI's API.

    Args:
        data: The JSON data to translate
        target_lang: Target language code
        model: OpenAI model to use
        fields_to_translate: List of field names to translate
        fields_to_skip: List of field names to skip
        chunk_size: Number of items to translate in each API call

    Returns:
        Translated JSON data
    """
    # Split data into chunks
    chunks = list(chunk_data(data, chunk_size))
    print(f"Split data into {len(chunks)} chunks of {chunk_size} items each")

    # Translate each chunk
    translated_data = []
    for i, chunk in enumerate(tqdm(chunks, desc=f"Translating to {target_lang}")):
        translated_chunk = translate_json_chunk(
            chunk, target_lang, model, fields_to_translate, fields_to_skip
        )
        translated_data.extend(translated_chunk)

        # Add a small delay to avoid rate limiting
        if i < len(chunks) - 1:
            time.sleep(1)

    return translated_data


def save_translated_json(data, target_lang, input_file):
    """Save translated JSON data to output file."""
    # Create output filename based on input filename
    input_path = Path(input_file)
    output_filename = f"{input_path.stem}_{target_lang}{input_path.suffix}"
    output_path = OUTPUT_DIR / output_filename

    # Save the translated data
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved translated data to {output_path}")
    return output_path


def get_supported_languages():
    """Get list of supported languages from substitution files."""
    subs_dir = Path("templates/substitutions")
    languages = []

    for file_path in subs_dir.glob("*.json"):
        lang_code = file_path.stem
        if (
            lang_code != "en" and "-" in lang_code
        ):  # Skip English and non-locale specific codes
            languages.append(lang_code)

    return languages


def main():
    parser = argparse.ArgumentParser(
        description="Translate a JSON file while preserving its structure using OpenAI's API"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="templates/atomic10x/atomic10x_als_subset.json",
        help="Input JSON file to translate",
    )
    parser.add_argument(
        "--lang",
        type=str,
        help="Target language code (e.g., fr-FR, es-ES, de-DE)",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        help="Language codes to process",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all supported languages",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--fields",
        type=str,
        nargs="+",
        default=["topic", "which"],
        help="Fields to translate (default: topic, which)",
    )
    parser.add_argument(
        "--skip-fields",
        type=str,
        nargs="+",
        default=["aac_user", "partner", "aac_user_role", "relation"],
        help="Fields to skip (default: aac_user, partner, aac_user_role, relation)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Number of items to translate in each API call (default: 100)",
    )
    args = parser.parse_args()

    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it to your OpenAI API key.")
        print("Example: export OPENAI_API_KEY=your-api-key")
        return

    # Load JSON data
    data = load_json_file(args.input)
    print(f"Loaded {len(data)} entries from {args.input}")

    # Determine languages to process
    if args.all:
        languages = get_supported_languages()
    elif args.languages:
        languages = args.languages
    elif args.lang:
        languages = [args.lang]
    else:
        print("Error: Please specify a language using --lang, --languages, or --all")
        return

    print(f"Processing languages: {', '.join(languages)}")

    # Process each language
    results = {}
    for lang in languages:
        print(f"\n{'='*80}\nProcessing language: {lang}\n{'='*80}\n")

        # Translate JSON data
        translated_data = translate_json_file(
            data,
            lang,
            args.model,
            fields_to_translate=args.fields,
            fields_to_skip=args.skip_fields,
            chunk_size=args.chunk_size,
        )

        # Save translated data
        output_path = save_translated_json(translated_data, lang, args.input)
        results[lang] = str(output_path)

        # Add a small delay to avoid rate limiting
        if lang != languages[-1]:
            time.sleep(2)

    # Print summary
    print(f"\n{'='*80}\nSummary\n{'='*80}")
    print(f"Processed {len(languages)} languages")
    for lang, output_path in results.items():
        print(f"  - {lang}: {output_path}")


if __name__ == "__main__":
    main()
