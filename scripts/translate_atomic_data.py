#!/usr/bin/env python3
"""
Translate the atomic10x_als_subset.json file to multiple languages.
This script translates only the 'topic' and 'which' fields in batches for efficiency.
"""
import json
import argparse
import os
import time
import sys
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Constants
BATCH_SIZE = 25  # Reduced batch size for better progress tracking
RATE_LIMIT_DELAY = 1  # Seconds to wait between API calls


def print_progress_bar(
    iteration, total, prefix="", suffix="", length=50, fill="█", print_end="\r"
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end   - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


def translate_batch(
    texts, source_lang="en", target_lang="fr-FR", batch_num=0, total_batches=1
):
    """
    Translate a batch of texts using OpenAI.

    Args:
        texts: List of texts to translate
        source_lang: Source language code
        target_lang: Target language code
        batch_num: Current batch number for progress tracking
        total_batches: Total number of batches for progress tracking

    Returns:
        List of translated texts
    """
    if not texts:
        return []

    # Get current timestamp for logging
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Prepare the prompt
    prompt = (
        f"Translate the following texts from {source_lang} to {target_lang}. "
        f"Keep the meaning intact but make it sound natural in the target language. "
        f"Return only the translations, one per line, with the same index numbers:\n\n"
    )

    for i, text in enumerate(texts):
        prompt += f"{i+1}. {text}\n"

    try:
        print(
            f"[{timestamp}] Sending batch {batch_num}/{total_batches} ({len(texts)} texts) to OpenAI..."
        )

        # Show a spinner while waiting for the API response
        spinner = ["|", "/", "-", "\\"]
        start_time = time.time()

        # Make the API call in a separate thread to allow for the spinner
        response_future = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        # Parse the response to extract translations
        translation_text = response_future.choices[0].message.content

        # Calculate elapsed time
        elapsed = time.time() - start_time

        print(f"[{timestamp}] Received response in {elapsed:.2f}s")
        # Split by newline and show first line
        first_line = (
            translation_text.split("\n")[0][:50]
            if "\n" in translation_text
            else translation_text[:50]
        )
        print(f"Sample translation: {first_line}...")

        translations = []

        # Parse line by line
        for line in translation_text.strip().split("\n"):
            if line and line[0].isdigit() and ". " in line:
                # Extract the translation part after the index
                translations.append(line.split(". ", 1)[1])

        # Ensure we have the same number of translations as input texts
        if len(translations) != len(texts):
            print(
                f"Warning: Got {len(translations)} translations for {len(texts)} input texts"
            )
            # Fill in missing translations if needed
            while len(translations) < len(texts):
                translations.append(texts[len(translations)])

        return translations

    except Exception as e:
        print(f"Error during translation: {e}")
        # Return original texts as fallback
        return texts


def translate_atomic_subset(input_file, output_file, target_lang, limit=None):
    """
    Translate the topic and which fields in the atomic subset.

    Args:
        input_file: Path to the input JSON file
        output_file: Path to the output JSON file
        target_lang: Target language code
        limit: Optional limit on number of entries to translate
    """
    start_time = time.time()

    # Check if output file already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping translation.")
        return

    # Check if temp file exists - we might be resuming a previous translation
    temp_file = f"{output_file}.temp"
    temp_data = None
    if os.path.exists(temp_file):
        try:
            with open(temp_file, "r", encoding="utf-8") as f:
                temp_data = json.load(f)
            print(
                f"Found temporary file with {len(temp_data)} entries. Resuming translation."
            )
        except Exception as e:
            print(f"Error loading temporary file: {e}")
            temp_data = None

    # Load the atomic subset
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries from {input_file}")

    # Limit the number of entries if specified
    if limit and limit < len(data):
        print(f"Limiting to {limit} entries as requested")
        data = data[:limit]

    # Extract all topics and which fields
    topics = [entry["topic"] for entry in data]
    which_fields = [entry["which"] for entry in data]

    # Initialize translated data from temp file if available
    translated_topics = []
    translated_which = []

    if temp_data:
        # Extract already translated data
        for entry in temp_data:
            if "topic" in entry:
                translated_topics.append(entry["topic"])
            if "which" in entry:
                translated_which.append(entry["which"])

        print(
            f"Loaded {len(translated_topics)} translated topics and {len(translated_which)} translated which fields from temp file"
        )

    # Calculate remaining items to translate
    remaining_topics = max(0, len(topics) - len(translated_topics))
    remaining_which = max(0, len(which_fields) - len(translated_which))

    total_items = remaining_topics + remaining_which
    print(
        f"Translating {remaining_topics} topics and {remaining_which} which fields to {target_lang}..."
    )
    print(f"Total items to translate: {total_items}")

    if total_items == 0:
        print("All items already translated. Creating final output file.")
    else:
        print(
            f"Estimated time: {total_items / BATCH_SIZE * 5:.1f} minutes (assuming 5 seconds per batch)"
        )

    # Create progress bar for overall progress
    overall_progress = tqdm(
        total=total_items,
        desc="Overall Progress",
        position=0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    # Translate remaining topics in batches
    if remaining_topics > 0:
        topics_to_translate = topics[len(translated_topics) :]
        total_topic_batches = (len(topics_to_translate) + BATCH_SIZE - 1) // BATCH_SIZE

        print(
            f"\nTranslating topics ({len(topics_to_translate)} items) in {total_topic_batches} batches:"
        )

        # Create a progress bar for topics
        topic_progress = tqdm(
            total=len(topics_to_translate),
            desc="Topics",
            position=1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        for i in range(0, len(topics_to_translate), BATCH_SIZE):
            batch = topics_to_translate[i : i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1

            # Translate batch
            translations = translate_batch(
                batch, "en", target_lang, batch_num, total_topic_batches
            )
            translated_topics.extend(translations)

            # Update progress bars
            topic_progress.update(len(batch))
            overall_progress.update(len(batch))

            # Save intermediate results to prevent data loss
            if batch_num % 2 == 0 or batch_num == total_topic_batches:
                temp_data = []
                for j, entry in enumerate(data[: len(translated_topics)]):
                    temp_entry = entry.copy()
                    temp_entry["topic"] = translated_topics[j]
                    if j < len(translated_which):
                        temp_entry["which"] = translated_which[j]
                    temp_data.append(temp_entry)

                # Save temporary data
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(temp_data, f, indent=2, ensure_ascii=False)
                tqdm.write(
                    f"Saved intermediate results ({len(temp_data)} entries) to {temp_file}"
                )

            time.sleep(RATE_LIMIT_DELAY)  # Avoid rate limits

        topic_progress.close()
        print("\nCompleted translating topics.")
    else:
        print("\nAll topics already translated. Skipping topic translation.")

    # Translate remaining which fields in batches
    if remaining_which > 0:
        which_to_translate = which_fields[len(translated_which) :]
        total_which_batches = (len(which_to_translate) + BATCH_SIZE - 1) // BATCH_SIZE

        print(
            f"\nTranslating 'which' fields ({len(which_to_translate)} items) in {total_which_batches} batches:"
        )

        # Create a progress bar for which fields
        which_progress = tqdm(
            total=len(which_to_translate),
            desc="Which Fields",
            position=1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        for i in range(0, len(which_to_translate), BATCH_SIZE):
            batch = which_to_translate[i : i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1

            # Translate batch
            translations = translate_batch(
                batch, "en", target_lang, batch_num, total_which_batches
            )
            translated_which.extend(translations)

            # Update progress bars
            which_progress.update(len(batch))
            overall_progress.update(len(batch))

            # Save intermediate results to prevent data loss
            if batch_num % 2 == 0 or batch_num == total_which_batches:
                temp_data = []
                for j, entry in enumerate(
                    data[: min(len(translated_topics), len(data))]
                ):
                    temp_entry = entry.copy()
                    if j < len(translated_topics):
                        temp_entry["topic"] = translated_topics[j]
                    if j < len(translated_which):
                        temp_entry["which"] = translated_which[j]
                    temp_data.append(temp_entry)

                # Save temporary data
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(temp_data, f, indent=2, ensure_ascii=False)
                tqdm.write(
                    f"Saved intermediate results ({len(temp_data)} entries) to {temp_file}"
                )

            time.sleep(RATE_LIMIT_DELAY)  # Avoid rate limits

        which_progress.close()
        print("\nCompleted translating 'which' fields.")
    else:
        print(
            "\nAll 'which' fields already translated. Skipping 'which' field translation."
        )

    overall_progress.close()

    # Create translated data
    translated_data = []
    for i, entry in enumerate(data):
        translated_entry = entry.copy()
        if i < len(translated_topics):
            translated_entry["topic"] = translated_topics[i]
        if i < len(translated_which):
            translated_entry["which"] = translated_which[i]
        translated_data.append(translated_entry)

    # Save translated data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, indent=2, ensure_ascii=False)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    print(f"\nSaved {len(translated_data)} translated entries to {output_file}")
    print(f"Translation completed in {int(minutes)} minutes and {int(seconds)} seconds")

    # Remove temporary file if it exists
    if os.path.exists(temp_file):
        os.remove(temp_file)


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
        description="Translate atomic subset to multiple languages"
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
        help="Target language code (e.g., fr-FR, es-ES). If not specified, will list available languages.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file (default: templates/atomic10x/atomic10x_als_subset_{lang}.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of entries to translate (for testing)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Translate to all supported languages"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip languages that already have translation files",
    )

    args = parser.parse_args()

    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        return

    # Print script information
    print(f"=== Atomic Data Translation Tool ===")
    print(f"Batch size: {BATCH_SIZE} items")
    print(f"Rate limit delay: {RATE_LIMIT_DELAY} seconds between batches")

    if args.all:
        # Translate to all supported languages
        languages = get_supported_languages()
        print(
            f"Found {len(languages)} languages to translate to: {', '.join(languages)}"
        )

        # Count how many languages already have translation files
        if args.skip_existing:
            existing_count = 0
            for lang in languages:
                output_file = f"templates/atomic10x/atomic10x_als_subset_{lang}.json"
                if os.path.exists(output_file):
                    existing_count += 1
            print(
                f"{existing_count} languages already have translation files and will be skipped."
            )

        # Process each language
        for i, lang in enumerate(languages):
            output_file = f"templates/atomic10x/atomic10x_als_subset_{lang}.json"

            # Skip if file exists and skip_existing is True
            if args.skip_existing and os.path.exists(output_file):
                print(
                    f"\n{'='*50}\nSkipping {lang} ({i+1}/{len(languages)}) - file already exists\n{'='*50}"
                )
                continue

            print(
                f"\n{'='*50}\nTranslating to {lang} ({i+1}/{len(languages)})\n{'='*50}"
            )
            translate_atomic_subset(args.input, output_file, lang, args.limit)

    elif args.lang:
        # Translate to a single language
        if not args.output:
            args.output = f"templates/atomic10x/atomic10x_als_subset_{args.lang}.json"

        translate_atomic_subset(args.input, args.output, args.lang, args.limit)
    else:
        # List available languages
        languages = get_supported_languages()
        print(f"Available languages: {', '.join(languages)}")
        print(
            "Use --lang CODE to translate to a specific language or --all to translate to all languages."
        )


if __name__ == "__main__":
    main()
