#!/usr/bin/env python3
"""
Translate JSON templates using Google's Gemini API.
This approach is much faster than using Google Translate for each string
and is free of charge (as of current Gemini pricing).
"""
import json
import argparse
import os
import time
import random
import multiprocessing
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables (for API keys)
load_dotenv()

# Set up the Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
    print(
        "Warning: GEMINI_API_KEY environment variable not set or using default value."
    )
    print(
        "This is fine for --estimate-only mode, but you'll need a valid API key to run translations."
    )
    # Use a dummy key for estimate-only mode
    GEMINI_API_KEY = "dummy_key_for_estimate_only"
genai.configure(api_key=GEMINI_API_KEY)

# Constants
OUTPUT_DIR = Path("templates")
INITIAL_BATCH_SIZE = 20  # Initial number of items to translate in one API call
MAX_BATCH_SIZE = 50  # Maximum batch size to try
MIN_BATCH_SIZE = 5  # Minimum batch size if we encounter errors
MODEL = "models/gemma-3-27b-it"  # Using Gemma 3 27B model
MAX_REQUESTS_PER_MINUTE = 60  # Rate limit for Gemini API (adjust as needed)
CHECKPOINT_DIR = Path("checkpoints")
MAX_TOKENS = 100000  # Approximate max tokens for Gemma 3 27B
CHARS_PER_TOKEN = 4  # Approximate characters per token
API_TIMEOUT = 30  # Timeout for API calls in seconds
PARALLEL_LANGUAGES = 3  # Process multiple languages in parallel

# Language names for better prompting
LANGUAGE_NAMES = {
    "af-ZA": "Afrikaans",
    "ar-SA": "Arabic",
    "ca-ES": "Catalan",
    "cs-CZ": "Czech",
    "cy-GB": "Welsh",
    "da-DK": "Danish",
    "de-AT": "German (Austria)",
    "de-DE": "German",
    "el-GR": "Greek",
    "es-ES": "Spanish",
    "es-US": "Spanish (US)",
    "eu-ES": "Basque",
    "fi-FI": "Finnish",
    "fr-CA": "French (Canada)",
    "fr-FR": "French",
    "he-IL": "Hebrew",
    "hr-HR": "Croatian",
    "it-IT": "Italian",
    "ja-JP": "Japanese",
    "ko-KR": "Korean",
    "nb-NO": "Norwegian",
    "nl-BE": "Dutch (Belgium)",
    "nl-NL": "Dutch",
    "pl-PL": "Polish",
    "pt-BR": "Portuguese (Brazil)",
    "pt-PT": "Portuguese",
    "ru-RU": "Russian",
    "sk-SK": "Slovak",
    "sl-SI": "Slovenian",
    "sv-SE": "Swedish",
    "uk-UA": "Ukrainian",
    "zh-CN": "Chinese (Simplified)",
    # English variants
    "en-AU": "English (Australia)",
    "en-CA": "English (Canada)",
    "en-GB": "English (UK)",
    "en-NZ": "English (New Zealand)",
    "en-US": "English (US)",
    "en-ZA": "English (South Africa)",
}

# Unsupported languages
UNSUPPORTED_LANGUAGES = [
    "fo-FO",  # Faroese
]


def load_json_file(input_file):
    """Load JSON data from input file."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_translated_json(data, target_lang, input_file):
    """Save translated JSON data to output file."""
    # Create output filename based on input filename
    input_path = Path(input_file)
    output_filename = f"{input_path.stem}_{target_lang}{input_path.suffix}"
    output_path = OUTPUT_DIR / output_filename

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save the translated data with explicit flush
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk

        # Double-check the file was written
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Successfully saved data to {output_path}")
        else:
            print(f"Warning: File may not have been written properly: {output_path}")
    except Exception as e:
        print(f"Error saving file {output_path}: {e}")
        return None

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


def check_output_exists(lang, input_file):
    """Check if the output file for this language already exists."""
    input_path = Path(input_file)
    output_filename = f"{input_path.stem}_{lang}{input_path.suffix}"
    output_path = OUTPUT_DIR / output_filename

    return output_path.exists()


def ensure_checkpoint_dir():
    """Ensure the checkpoint directory exists."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_checkpoint_path(lang, input_file):
    """Get the path for a language checkpoint file."""
    input_path = Path(input_file)
    checkpoint_filename = f"{input_path.stem}_{lang}_checkpoint.json"
    return CHECKPOINT_DIR / checkpoint_filename


def save_checkpoint(lang, input_file, translated_data, batch_index, total_batches):
    """Save a checkpoint for the current language and translation progress."""
    ensure_checkpoint_dir()
    checkpoint_path = get_checkpoint_path(lang, input_file)

    checkpoint = {
        "lang": lang,
        "input_file": str(input_file),
        "batch_index": batch_index,
        "total_batches": total_batches,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "translated_data": translated_data,
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


def rate_limit():
    """Implement rate limiting for the Gemini API."""
    # Calculate delay to maintain MAX_REQUESTS_PER_MINUTE
    delay = 60.0 / MAX_REQUESTS_PER_MINUTE
    # Add some jitter to avoid synchronized requests
    jitter = random.uniform(-0.1, 0.1) * delay
    time.sleep(delay + jitter)


def estimate_tokens(text):
    """
    Estimate the number of tokens in a text string.
    This is a rough approximation based on character count.
    """
    return len(text) // CHARS_PER_TOKEN


def estimate_batch_tokens(batch):
    """
    Estimate the number of tokens in a batch of JSON objects.
    """
    # Convert batch to JSON string
    batch_json = json.dumps(batch, ensure_ascii=False)
    # Add prompt template length (approximate)
    prompt_template_length = 500  # Characters in the prompt template
    total_chars = len(batch_json) + prompt_template_length
    return total_chars // CHARS_PER_TOKEN


def adjust_batch_size(current_size, success, tokens_estimate=None):
    """
    Adjust batch size based on success/failure and token estimates.

    Args:
        current_size: Current batch size
        success: Whether the last API call was successful
        tokens_estimate: Estimated tokens for the current batch size

    Returns:
        New batch size
    """
    if not success:
        # If failed, reduce batch size
        new_size = max(MIN_BATCH_SIZE, current_size // 2)
        print(f"Reducing batch size from {current_size} to {new_size} due to API error")
        return new_size

    if tokens_estimate and tokens_estimate > MAX_TOKENS * 0.8:
        # If we're using more than 80% of the token limit, reduce batch size
        new_size = max(MIN_BATCH_SIZE, int(current_size * 0.8))
        print(
            f"Reducing batch size from {current_size} to {new_size} due to token limit"
        )
        return new_size

    if tokens_estimate and tokens_estimate < MAX_TOKENS * 0.5:
        # If we're using less than 50% of the token limit, increase batch size
        new_size = min(MAX_BATCH_SIZE, int(current_size * 1.2))
        if new_size > current_size:
            print(f"Increasing batch size from {current_size} to {new_size}")
        return new_size

    # Otherwise keep the same size
    return current_size


def translate_batch_with_gemini(batch, target_lang):
    """
    Translate a batch of JSON objects using Gemini API.

    Args:
        batch: List of JSON objects to translate
        target_lang: Target language code

    Returns:
        Tuple of (translated_batch, success, token_estimate)
    """
    # Get the language name for better prompting
    language_name = LANGUAGE_NAMES.get(target_lang, target_lang)

    # Create the prompt
    prompt = f"""You are a professional translator specializing in {language_name}.
Your task is to translate JSON objects from English to {language_name}.
Translate ONLY the text values, not the keys or structure.
Maintain the same tone, style, and meaning in your translations.
Return ONLY the translated JSON objects in valid JSON format.

Here are the JSON objects to translate:
{json.dumps(batch, indent=2, ensure_ascii=False)}

Important instructions:
1. Preserve any placeholders like ____ or {{variable}}
2. Return ONLY the translated JSON objects in the same format
3. Make sure your response is valid JSON that can be parsed
4. Do not include any explanations or notes, just the JSON
"""

    # Estimate tokens
    token_estimate = estimate_tokens(prompt)
    if token_estimate > MAX_TOKENS * 0.9:
        print(
            f"Warning: Prompt is too large ({token_estimate} tokens). Returning original batch."
        )
        return batch, False, token_estimate

    # Apply rate limiting
    rate_limit()

    # Call the Gemini API
    print(f"Sending batch of {len(batch)} items to API...")
    start_time = time.time()

    try:
        model = genai.GenerativeModel(MODEL)

        # Set a timeout for the API call
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,  # Low temperature for consistent translations
                max_output_tokens=MAX_TOKENS // 2,
                top_p=0.95,
                top_k=40,
            ),
        )

        api_time = time.time() - start_time
        print(f"API response received in {api_time:.2f} seconds")

        # Extract the translated JSON from the response
        translated_text = response.text

        # Parse the JSON response
        try:
            # Try to parse the entire response as JSON
            translated_batch = json.loads(translated_text)
            print(
                f"Successfully parsed JSON response with {len(translated_batch)} items"
            )
            return translated_batch, True, token_estimate
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the text
            try:
                # Look for JSON-like content (between [ and ])
                start_idx = translated_text.find("[")
                end_idx = translated_text.rfind("]") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_content = translated_text[start_idx:end_idx]
                    translated_batch = json.loads(json_content)
                    print(
                        f"Extracted and parsed JSON content with {len(translated_batch)} items"
                    )
                    return translated_batch, True, token_estimate
                else:
                    print("Error: Could not find JSON content in response")
                    print(f"Response starts with: {translated_text[:200]}...")
                    print(f"Response ends with: ...{translated_text[-200:]}")

                    # Try to fix common JSON formatting issues
                    fixed_text = (
                        translated_text.replace("```json", "")
                        .replace("```", "")
                        .strip()
                    )
                    if fixed_text.startswith("[") and fixed_text.endswith("]"):
                        try:
                            translated_batch = json.loads(fixed_text)
                            print(
                                f"Fixed JSON formatting and parsed {len(translated_batch)} items"
                            )
                            return translated_batch, True, token_estimate
                        except json.JSONDecodeError:
                            pass

                    return (
                        batch,
                        False,
                        token_estimate,
                    )  # Return original batch on error
            except Exception as e:
                print(f"Error parsing JSON response: {e}")
                print(f"Response starts with: {translated_text[:200]}...")
                return batch, False, token_estimate  # Return original batch on error

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")

        # Check for timeout or connection issues
        if "timeout" in str(e).lower() or "connection" in str(e).lower():
            print("Network issue detected. Reducing batch size and adding delay.")
            time.sleep(5)  # Add extra delay

        return batch, False, token_estimate  # Return original batch on error


def translate_json_with_gemini(
    data, target_lang, input_file, batch_size=INITIAL_BATCH_SIZE, resume=False
):
    """
    Translate a JSON file using Gemini API with adaptive batch sizing.

    Args:
        data: The JSON data to translate
        target_lang: Target language code
        input_file: Path to the input file
        batch_size: Initial number of items to translate in one API call
        resume: Whether to resume from a checkpoint if available

    Returns:
        Translated JSON data
    """
    # Skip if the language is unsupported
    if target_lang in UNSUPPORTED_LANGUAGES:
        print(f"Warning: Language {target_lang} is not supported.")
        return data

    # Skip English variants
    if target_lang.startswith("en-"):
        print(f"Skipping English variant: {target_lang}")
        return data

    # Check if we have a checkpoint to resume from
    checkpoint = None
    if resume:
        checkpoint = load_checkpoint(target_lang, input_file)

    if checkpoint:
        # Resume from checkpoint
        translated_data = checkpoint["translated_data"]
        start_index = len(translated_data)  # Start from where we left off
        remaining_data = data[start_index:]
        print(f"Resuming from item {start_index}/{len(data)}")
    else:
        # Start fresh
        translated_data = []
        remaining_data = data
        start_index = 0

    # Calculate estimated time (initial estimate)
    current_batch_size = batch_size
    total_batches = (len(remaining_data) + current_batch_size - 1) // current_batch_size
    seconds_per_batch = 60.0 / MAX_REQUESTS_PER_MINUTE
    estimated_time_seconds = total_batches * seconds_per_batch
    estimated_time_hours = estimated_time_seconds / 3600

    print(f"Initial batch size: {current_batch_size}")
    print(f"Estimated time to complete {target_lang}: {estimated_time_hours:.2f} hours")

    # Translate remaining data with adaptive batch sizing
    print(f"Translating to {target_lang}...")

    # Initialize progress bar with total items and correct initial position
    progress_bar = tqdm(
        total=len(data),  # Total items in the full dataset
        initial=len(translated_data),  # Start from current progress
        desc=f"Translating to {target_lang}",
    )

    # Store start time for accurate time estimation
    progress_bar.start_t = time.time()

    i = 0
    while i < len(remaining_data):
        # Get the current batch
        end_idx = min(i + current_batch_size, len(remaining_data))
        current_batch = remaining_data[i:end_idx]

        # Translate the batch
        translated_batch, success, token_estimate = translate_batch_with_gemini(
            current_batch, target_lang
        )

        if success:
            # Add the translated batch to the result
            translated_data.extend(translated_batch)

            # Update progress
            batch_items = len(current_batch)
            progress_bar.update(batch_items)
            i += batch_items

            # Adjust batch size based on success and token usage
            current_batch_size = adjust_batch_size(
                current_batch_size, True, token_estimate
            )
        else:
            # If failed and batch size > MIN_BATCH_SIZE, reduce batch size and retry
            if current_batch_size > MIN_BATCH_SIZE:
                current_batch_size = adjust_batch_size(current_batch_size, False)
                # Don't increment i, retry with smaller batch
            else:
                # If we're already at minimum batch size, skip this batch
                print(
                    f"Warning: Failed to translate batch at minimum size. Skipping {len(current_batch)} items."
                )
                translated_data.extend(current_batch)  # Add original items
                progress_bar.update(len(current_batch))
                i += len(current_batch)

        # Save checkpoint and intermediate results periodically
        current_position = start_index + i
        # Save more frequently (every 50 items) and always after each batch
        if (
            current_position % 50 == 0
            or i >= len(remaining_data)
            or len(current_batch) > 0
        ):
            # Save checkpoint
            save_checkpoint(
                target_lang, input_file, translated_data, current_position, len(data)
            )

            # Save partial results - force flush to ensure it's written
            partial_output_path = save_translated_json(
                translated_data, f"{target_lang}_partial", input_file
            )
            print(
                f"Saved partial results to {partial_output_path} ({len(translated_data)} items)"
            )

            # Verify the file was written
            if os.path.exists(partial_output_path):
                file_size = os.path.getsize(partial_output_path)
                print(
                    f"Verified file exists: {partial_output_path} (Size: {file_size} bytes)"
                )
            else:
                print(
                    f"WARNING: Failed to save partial results to {partial_output_path}"
                )

            # Update estimated time - calculate based on total remaining items
            total_items = len(data)
            processed_items = len(translated_data)
            remaining_items = total_items - processed_items

            # Calculate time per item based on recent performance
            elapsed_time = time.time() - progress_bar.start_t
            if processed_items > 0 and elapsed_time > 0:
                time_per_item = elapsed_time / processed_items
                time_left_seconds = remaining_items * time_per_item
                time_left_hours = time_left_seconds / 3600

                # Calculate completion percentage and ETA
                percent_complete = (processed_items / total_items) * 100
                eta = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(time.time() + time_left_seconds)
                )

                print(
                    f"Progress: {processed_items}/{total_items} items ({percent_complete:.1f}%)"
                )
                print(f"Current batch size: {current_batch_size}")
                print(f"Estimated time remaining: {time_left_hours:.2f} hours")
                print(f"Estimated completion time: {eta}")

    progress_bar.close()
    return translated_data


def process_language(
    data, lang, input_file, batch_size, resume, skip_unsupported, skip_completed
):
    """Process a single language."""
    # Skip English variants
    if lang.startswith("en-"):
        print(f"Skipping English variant: {lang}")
        return None

    # Skip unsupported languages
    if lang in UNSUPPORTED_LANGUAGES and skip_unsupported:
        print(f"Skipping unsupported language: {lang}")
        return None

    # Skip if already completed and skip_completed is specified
    if skip_completed and check_output_exists(lang, input_file):
        print(f"Output file already exists for {lang}. Skipping.")
        return None

    # Translate JSON data
    translated_data = translate_json_with_gemini(
        data, lang, input_file, batch_size=batch_size, resume=resume
    )

    # Save translated data
    output_path = save_translated_json(translated_data, lang, input_file)

    # Remove checkpoint file after successful completion
    checkpoint_path = get_checkpoint_path(lang, input_file)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Removed checkpoint file: {checkpoint_path}")

    return str(output_path)


def main():
    global MAX_REQUESTS_PER_MINUTE

    parser = argparse.ArgumentParser(
        description="Translate a JSON file using Google's Gemini API"
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
        help="Target language code (e.g., fr-FR, es-ES)",
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
        help="Skip languages not supported",
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
        "--batch-size",
        type=int,
        default=INITIAL_BATCH_SIZE,
        help=f"Initial number of items to translate in one API call (default: {INITIAL_BATCH_SIZE})",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=MAX_REQUESTS_PER_MINUTE,
        help=f"Maximum requests per minute (default: {MAX_REQUESTS_PER_MINUTE})",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=PARALLEL_LANGUAGES,
        help=f"Number of languages to process in parallel (default: {PARALLEL_LANGUAGES})",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only estimate time without performing translations",
    )
    args = parser.parse_args()

    # Update rate limit if specified
    MAX_REQUESTS_PER_MINUTE = args.rate_limit

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

    # Calculate estimated time
    if len(languages) > 0:
        # Estimate based on rate limits and adaptive batch sizing
        # Start with initial batch size, but account for adaptive sizing
        avg_batch_size = args.batch_size * 0.8  # Conservative estimate
        batches_per_language = len(data) / avg_batch_size

        # Account for parallel processing
        parallel_factor = min(args.parallel, len(languages))

        total_batches = batches_per_language * len(languages)
        seconds_per_batch = 60.0 / MAX_REQUESTS_PER_MINUTE
        total_time_seconds = total_batches * seconds_per_batch / parallel_factor
        total_time_hours = total_time_seconds / 3600

        print("\nEstimated time for all languages:")
        print(f"  - Items per language: {len(data)}")
        print(f"  - Languages to process: {len(languages)}")
        print(f"  - Languages in parallel: {parallel_factor}")
        print(f"  - Initial batch size: {args.batch_size}")
        print(f"  - Estimated average batch size: {avg_batch_size:.1f}")
        print(f"  - Estimated total batches: {total_batches:,.0f}")
        print(f"  - Rate limit: {MAX_REQUESTS_PER_MINUTE} requests per minute")
        print(
            f"  - Estimated time: {total_time_hours:.2f} hours ({total_time_hours/24:.1f} days)"
        )

        completion_time = time.time() + total_time_seconds
        estimated_completion = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(completion_time)
        )
        print(f"  - Estimated completion: {estimated_completion}")

        if args.estimate_only:
            print("Estimate-only mode. Exiting without performing translations.")
            return

    # Process languages in parallel using multiprocessing
    if args.parallel > 1 and len(languages) > 1:
        print(
            f"\nProcessing {len(languages)} languages with {args.parallel} parallel workers"
        )

        # Create a multiprocessing pool
        from multiprocessing import Pool

        # Define a worker function for the pool
        def worker(lang):
            separator = "=" * 80
            print(f"\n{separator}\nProcessing language: {lang}\n{separator}\n")
            return (
                lang,
                process_language(
                    data,
                    lang,
                    args.input,
                    args.batch_size,
                    args.resume,
                    args.skip_unsupported,
                    args.skip_completed,
                ),
            )

        # Process languages in parallel
        with Pool(processes=args.parallel) as pool:
            results_list = pool.map(worker, languages)

        # Filter out None results and convert to dictionary
        results = {lang: path for lang, path in results_list if path is not None}
    else:
        # Process languages sequentially (original behavior)
        results = {}
        for lang_index, lang in enumerate(languages):
            separator = "=" * 80
            progress = f"{lang_index+1}/{len(languages)}"
            print(
                f"\n{separator}\nProcessing language: {lang} ({progress})\n{separator}\n"
            )

            output_path = process_language(
                data,
                lang,
                args.input,
                args.batch_size,
                args.resume,
                args.skip_unsupported,
                args.skip_completed,
            )

            if output_path:
                results[lang] = output_path

    # Print summary
    print(f"\n{'='*80}\nSummary\n{'='*80}")
    print(f"Processed {len(results)} languages")
    for lang, output_path in results.items():
        print(f"  - {lang}: {output_path}")


if __name__ == "__main__":
    main()
