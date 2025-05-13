#!/usr/bin/env python3
"""
Translate an entire JSON file while preserving its structure.
This script uses deep_translator (Google Translate) to translate only the text values in a JSON file.
No API key required - uses the free tier of Google Translate.

Features:
- Checkpointing: Saves progress regularly and can resume from where it left off
- Batch processing: Translates strings in batches to avoid rate limiting
- Progress tracking: Shows detailed progress information
"""
import json
import argparse
from pathlib import Path
import time
import random
import os
import datetime
from tqdm import tqdm

from deep_translator import GoogleTranslator

# Constants
OUTPUT_DIR = Path("templates")
CHECKPOINT_DIR = Path("checkpoints")
# Rate limiting constants
MIN_DELAY = 0.5  # Minimum delay between batches in seconds
MAX_DELAY = 1.5  # Maximum delay between batches in seconds
BATCH_SIZE = 50  # Number of texts to translate in one batch
# Checkpointing constants
SAVE_FREQUENCY = 10  # Save progress every N batches

# Language code mapping
# Map our locale-specific codes to the language codes that deep_translator supports
LANGUAGE_CODE_MAP = {
    # European languages
    "af-ZA": "af",  # Afrikaans
    "ca-ES": "ca",  # Catalan
    "cs-CZ": "cs",  # Czech
    "cy-GB": "cy",  # Welsh
    "da-DK": "da",  # Danish
    "de-AT": "de",  # German (Austria)
    "de-DE": "de",  # German
    "el-GR": "el",  # Greek
    "es-ES": "es",  # Spanish
    "es-US": "es",  # Spanish (US)
    "eu-ES": "eu",  # Basque
    "fi-FI": "fi",  # Finnish
    "fr-CA": "fr",  # French (Canada)
    "fr-FR": "fr",  # French
    "hr-HR": "hr",  # Croatian
    "it-IT": "it",  # Italian
    "nb-NO": "no",  # Norwegian
    "nl-BE": "nl",  # Dutch (Belgium)
    "nl-NL": "nl",  # Dutch
    "pl-PL": "pl",  # Polish
    "pt-BR": "pt",  # Portuguese (Brazil)
    "pt-PT": "pt",  # Portuguese
    "ru-RU": "ru",  # Russian
    "sk-SK": "sk",  # Slovak
    "sl-SI": "sl",  # Slovenian
    "sv-SE": "sv",  # Swedish
    "uk-UA": "uk",  # Ukrainian
    # Asian languages
    "ar-SA": "ar",  # Arabic
    "he-IL": "iw",  # Hebrew (note: Google uses 'iw' for Hebrew)
    "ja-JP": "ja",  # Japanese
    "ko-KR": "ko",  # Korean
    "zh-CN": "zh-CN",  # Chinese (Simplified)
    # English variants (for completeness)
    "en-AU": "en",  # English (Australia)
    "en-CA": "en",  # English (Canada)
    "en-GB": "en",  # English (UK)
    "en-NZ": "en",  # English (New Zealand)
    "en-US": "en",  # English (US)
    "en-ZA": "en",  # English (South Africa)
}

# Languages not supported by deep_translator
UNSUPPORTED_LANGUAGES = [
    "fo-FO",  # Faroese
]


def load_json_file(input_file):
    """Load JSON data from input file."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def collect_strings_to_translate(data, fields_to_translate=None, fields_to_skip=None):
    """
    Collect all strings that need to be translated from the JSON data.

    Args:
        data: The JSON data to process
        fields_to_translate: List of field names to translate
        fields_to_skip: List of field names to skip

    Returns:
        A list of (path, string) tuples, where path is a tuple of keys/indices
    """
    # Default fields to translate if not specified
    if fields_to_translate is None:
        fields_to_translate = ["topic", "which"]

    # Default fields to skip if not specified
    if fields_to_skip is None:
        fields_to_skip = ["aac_user", "partner", "aac_user_role", "relation"]

    strings_to_translate = []

    def collect_strings(value, path=(), field_name=None):
        """Recursively collect strings to translate."""
        if isinstance(value, dict):
            for k, v in value.items():
                collect_strings(v, path + (k,), k)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                collect_strings(item, path + (i,), field_name)
        elif (
            isinstance(value, str)
            and field_name in fields_to_translate
            and field_name not in fields_to_skip
            and value.strip()  # Skip empty strings
        ):
            strings_to_translate.append((path, value))

    collect_strings(data)
    return strings_to_translate


def translate_strings_batch(strings, target_lang):
    """
    Translate a batch of strings using deep_translator.

    Args:
        strings: List of strings to translate
        target_lang: Target language code

    Returns:
        List of translated strings
    """
    # Check if the language is supported
    if target_lang in UNSUPPORTED_LANGUAGES:
        print(f"Warning: Language {target_lang} is not supported by deep_translator.")
        print("Returning original strings without translation.")
        return strings

    # Map the language code to what deep_translator expects
    if target_lang in LANGUAGE_CODE_MAP:
        deep_translator_lang = LANGUAGE_CODE_MAP[target_lang]
    else:
        # Try to use the first part of the code (e.g., "fr-FR" -> "fr")
        extracted_code = (
            target_lang.split("-")[0] if "-" in target_lang else target_lang
        )

        # Check if this is a direct language code (not from a locale)
        if target_lang == extracted_code:
            deep_translator_lang = extracted_code
        else:
            deep_translator_lang = extracted_code

    # Initialize the translator
    try:
        translator = GoogleTranslator(source="en", target=deep_translator_lang)
    except Exception as e:
        print(f"Error initializing translator for {deep_translator_lang}: {e}")
        print("Returning original strings without translation.")
        return strings

    # Translate the batch
    try:
        translated_strings = translator.translate_batch(strings)
        return translated_strings
    except Exception as e:
        print(f"Error translating batch: {e}")
        # Fall back to translating one by one
        translated_strings = []
        for s in strings:
            try:
                translated = translator.translate(s)
                translated_strings.append(translated)
            except Exception as e:
                print(f"Error translating '{s}': {e}")
                translated_strings.append(s)  # Use original string on error
            # Add a small delay to avoid rate limiting
            time.sleep(random.uniform(0.5, 1.0))
        return translated_strings


def translate_json(
    data,
    target_lang,
    input_file,
    fields_to_translate=None,
    fields_to_skip=None,
    batch_size=BATCH_SIZE,
    min_delay=MIN_DELAY,
    max_delay=MAX_DELAY,
    save_frequency=SAVE_FREQUENCY,
):
    """
    Translate text values in a JSON structure using deep_translator.

    Args:
        data: The JSON data to translate
        target_lang: Target language code
        input_file: Path to the input file (used for checkpointing)
        fields_to_translate: List of field names to translate
        fields_to_skip: List of field names to skip
        batch_size: Number of strings to translate in each batch
        min_delay: Minimum delay between batches in seconds
        max_delay: Maximum delay between batches in seconds
        save_frequency: Save checkpoint every N batches

    Returns:
        Translated JSON data
    """
    # Default fields to translate if not specified
    if fields_to_translate is None:
        fields_to_translate = ["topic", "which"]

    # Default fields to skip if not specified
    if fields_to_skip is None:
        fields_to_skip = ["aac_user", "partner", "aac_user_role", "relation"]

    # Check if we have a checkpoint to resume from
    checkpoint = load_checkpoint(target_lang, input_file)

    if checkpoint:
        # Resume from checkpoint
        translated_data = checkpoint["translated_data"]
        strings_to_translate = checkpoint["strings_to_translate"]
        start_batch = checkpoint["batch_index"] + 1  # Start from the next batch

        # Split strings into batches (same as before to ensure consistency)
        batches = []
        for i in range(0, len(strings_to_translate), batch_size):
            end_idx = i + batch_size
            batches.append(strings_to_translate[i:end_idx])

        print(
            f"Resuming translation for {target_lang} from batch {start_batch}/{len(batches)}"
        )
    else:
        # Start fresh
        # Collect all strings that need to be translated
        print("Collecting strings to translate...")
        strings_to_translate = collect_strings_to_translate(
            data, fields_to_translate, fields_to_skip
        )
        print(f"Found {len(strings_to_translate)} strings to translate")

        # Create a copy of the data to modify
        translated_data = json.loads(json.dumps(data))

        # Split strings into batches
        batches = []
        for i in range(0, len(strings_to_translate), batch_size):
            end_idx = i + batch_size
            batches.append(strings_to_translate[i:end_idx])
        print(f"Split into {len(batches)} batches of up to {batch_size} strings each")

        start_batch = 0

    # Calculate estimated time
    avg_time_per_batch = 50  # seconds, based on observed performance
    remaining_batches = len(batches) - start_batch
    estimated_time_seconds = remaining_batches * avg_time_per_batch
    estimated_time_hours = estimated_time_seconds / 3600

    print(f"Estimated time to complete {target_lang}: {estimated_time_hours:.2f} hours")

    # Translate each batch
    print(f"Translating to {target_lang}...")
    progress_bar = tqdm(
        total=len(batches), initial=start_batch, desc=f"Translating to {target_lang}"
    )

    for i in range(start_batch, len(batches)):
        batch = batches[i]

        # Extract strings from the batch
        strings = [string for _, string in batch]

        # Translate the strings
        translated_strings = translate_strings_batch(strings, target_lang)

        # Update the data with translated strings
        for j, (path, _) in enumerate(batch):
            if j < len(translated_strings):
                # Navigate to the correct position in the data
                target = translated_data
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = translated_strings[j]

        # Save checkpoint periodically
        if (i + 1) % save_frequency == 0 or i == len(batches) - 1:
            save_checkpoint(
                target_lang,
                input_file,
                translated_data,
                i,
                len(batches),
                strings_to_translate,
            )

            # Also save the current progress as a partial result file
            partial_output_path = save_translated_json(
                translated_data, f"{target_lang}_partial", input_file
            )
            print(f"Saved partial results to {partial_output_path}")

        # Add a delay between batches to avoid rate limiting
        if i < len(batches) - 1:
            time.sleep(random.uniform(min_delay, max_delay))

        # Update progress bar
        progress_bar.update(1)

    progress_bar.close()
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
        # Skip English variants and non-locale specific codes
        if "-" in lang_code and not lang_code.startswith("en-"):
            languages.append(lang_code)

    return languages


def ensure_checkpoint_dir():
    """Ensure the checkpoint directory exists."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_checkpoint_path(lang, input_file):
    """Get the path for a language checkpoint file."""
    input_path = Path(input_file)
    checkpoint_filename = f"{input_path.stem}_{lang}_checkpoint.json"
    return CHECKPOINT_DIR / checkpoint_filename


def save_checkpoint(
    lang, input_file, translated_data, batch_index, total_batches, strings_to_translate
):
    """Save a checkpoint for the current language and translation progress."""
    ensure_checkpoint_dir()
    checkpoint_path = get_checkpoint_path(lang, input_file)

    checkpoint = {
        "lang": lang,
        "input_file": str(input_file),
        "batch_index": batch_index,
        "total_batches": total_batches,
        "timestamp": datetime.datetime.now().isoformat(),
        "translated_data": translated_data,
        "strings_to_translate": strings_to_translate,
    }

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)

    print(f"Checkpoint saved: {checkpoint_path} (Batch {batch_index}/{total_batches})")


def load_checkpoint(lang, input_file):
    """Load a checkpoint for the given language and input file if it exists."""
    checkpoint_path = get_checkpoint_path(lang, input_file)

    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)

        print(f"Loaded checkpoint: {checkpoint_path}")
        print(
            f"Resuming from batch {checkpoint['batch_index']}/{checkpoint['total_batches']}"
        )

        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


def check_output_exists(lang, input_file):
    """Check if the output file for this language already exists."""
    input_path = Path(input_file)
    output_filename = f"{input_path.stem}_{lang}{input_path.suffix}"
    output_path = OUTPUT_DIR / output_filename

    return output_path.exists()


def main():
    parser = argparse.ArgumentParser(
        description="Translate a JSON file while preserving its structure"
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
        help="Target language code (e.g., fr, es, de)",
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
        "--skip-english",
        action="store_true",
        help="Skip all English variants (en-*)",
    )
    parser.add_argument(
        "--skip-unsupported",
        action="store_true",
        help="Skip languages not supported by deep_translator",
    )
    parser.add_argument(
        "--process-unsupported",
        action="store_true",
        help="Process unsupported languages by creating untranslated copies",
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
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of strings to translate in each batch (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--min-delay",
        type=float,
        default=MIN_DELAY,
        help=f"Minimum delay between batches in seconds (default: {MIN_DELAY})",
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=MAX_DELAY,
        help=f"Maximum delay between batches in seconds (default: {MAX_DELAY})",
    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=SAVE_FREQUENCY,
        help=f"Save checkpoint every N batches (default: {SAVE_FREQUENCY})",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip languages that already have completed output files",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints if available",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only estimate time required without performing translations",
    )
    args = parser.parse_args()

    # Ensure checkpoint directory exists
    ensure_checkpoint_dir()

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

    # Filter out English variants if specified
    if args.skip_english:
        languages = [lang for lang in languages if not lang.startswith("en-")]
        print(f"Skipping English variants. Processing {len(languages)} languages.")

    # Filter out unsupported languages if specified
    if args.skip_unsupported:
        languages = [lang for lang in languages if lang not in UNSUPPORTED_LANGUAGES]
        print(f"Skipping unsupported languages. Processing {len(languages)} languages.")

    # Skip completed languages if specified
    if args.skip_completed:
        original_count = len(languages)
        languages = [
            lang for lang in languages if not check_output_exists(lang, args.input)
        ]
        skipped_count = original_count - len(languages)
        if skipped_count > 0:
            print(f"Skipping {skipped_count} already completed languages.")
            print(f"Processing {len(languages)} remaining languages.")

    # Calculate total estimated time
    if len(languages) > 0:
        # Estimate based on observed performance
        avg_time_per_batch = 50  # seconds
        avg_batches_per_lang = 1949  # based on observed data
        total_estimated_hours = (
            len(languages) * avg_batches_per_lang * avg_time_per_batch
        ) / 3600
        print(
            f"\nEstimated total time for all languages: {total_estimated_hours:.2f} hours"
        )
        print(
            f"Estimated completion time: {datetime.datetime.now() + datetime.timedelta(hours=total_estimated_hours)}"
        )

        if args.estimate_only:
            print("Estimate-only mode. Exiting without performing translations.")
            return

    # Process each language
    results = {}
    for lang_index, lang in enumerate(languages):
        print(
            f"\n{'='*80}\nProcessing language: {lang} ({lang_index+1}/{len(languages)})\n{'='*80}\n"
        )

        # Skip English variants
        if lang.startswith("en-"):
            print(f"Skipping English variant: {lang}")
            continue

        # Skip unsupported languages
        if lang in UNSUPPORTED_LANGUAGES and not args.process_unsupported:
            print(f"Skipping unsupported language: {lang}")
            continue

        # Skip if already completed and --skip-completed is specified
        if args.skip_completed and check_output_exists(lang, args.input):
            print(f"Output file already exists for {lang}. Skipping.")
            continue

        # Get the appropriate language code for deep_translator
        if lang in LANGUAGE_CODE_MAP:
            google_lang_code = LANGUAGE_CODE_MAP[lang]
        else:
            # Extract the language code without the region (e.g., "fr-FR" -> "fr")
            extracted_code = lang.split("-")[0] if "-" in lang else lang

            # Check if this is a direct language code (not from a locale)
            if lang == extracted_code:
                google_lang_code = extracted_code
            else:
                print(f"Note: Using {extracted_code} for {lang}")
                google_lang_code = extracted_code

        # For unsupported languages, we'll still create a copy of the original data
        if lang in UNSUPPORTED_LANGUAGES and args.process_unsupported:
            print(f"Creating untranslated copy for unsupported language: {lang}")
            # Just copy the original data
            output_path = save_translated_json(data, lang, args.input)
            results[lang] = str(output_path)
            continue

        # Translate JSON data
        translated_data = translate_json(
            data,
            google_lang_code,
            args.input,  # Pass input file path for checkpointing
            fields_to_translate=args.fields,
            fields_to_skip=args.skip_fields,
            batch_size=args.batch_size,
            min_delay=args.min_delay,
            max_delay=args.max_delay,
            save_frequency=args.save_frequency,
        )

        # Save translated data
        output_path = save_translated_json(translated_data, lang, args.input)
        results[lang] = str(output_path)

        # Remove checkpoint file after successful completion
        checkpoint_path = get_checkpoint_path(lang, args.input)
        if checkpoint_path.exists():
            os.remove(checkpoint_path)
            print(f"Removed checkpoint file: {checkpoint_path}")

        # Add a small delay to avoid rate limiting
        if lang != languages[-1]:
            time.sleep(random.uniform(args.min_delay, args.max_delay))

    # Print summary
    print(f"\n{'='*80}\nSummary\n{'='*80}")
    print(f"Processed {len(results)} languages")
    for lang, output_path in results.items():
        print(f"  - {lang}: {output_path}")


if __name__ == "__main__":
    main()
