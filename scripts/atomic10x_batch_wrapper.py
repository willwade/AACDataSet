#!/usr/bin/env python3
"""
Wrapper script for the Atomic10x workflow.
This script runs the entire workflow for multiple languages and with configurable batch sizes.
"""
import argparse
import subprocess
import os
import time
import json
import sys
import shutil
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Default settings
DEFAULT_LANGUAGES = [
    # Currently supported languages
    "en-GB",
    "fr-FR",
    "es-ES",
    "de-DE",
    "it-IT",
    "nl-NL",
    "pt-BR",
    "ru-RU",
    "zh-CN",
    "ja-JP",
    "ar-SA",
    "he-IL",
    "el-GR",
    "cy-GB",
    # Additional languages to support
    "af-ZA",
    "eu-ES",
    "ca-ES",
    "hr-HR",
    "cs-CZ",
    "da-DK",
    "nl-BE",
    "en-AU",
    "en-CA",
    "en-NZ",
    "en-ZA",
    "en-US",
    "fo-FO",
    "fi-FI",
    "fr-CA",
    "de-AT",
    "nb-NO",
    "pl-PL",
    "pt-PT",
    "sk-SK",
    "sl-SI",
    "es-US",
    "sv-SE",
    "uk-UA",
    "ko-KR",
]
DEFAULT_REQUESTS_PER_BATCH = 100
DEFAULT_NUM_BATCHES = 1
DEFAULT_MODEL = "gpt-4-turbo"
DEFAULT_ATOMIC_LIMIT = None  # No limit by default

# Global progress tracking
global_progress = {
    "total_languages": 0,
    "completed_languages": 0,
    "total_batches": 0,
    "completed_batches": 0,
    "current_language": "",
    "current_step": "",
    "start_time": None,
}


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


def update_progress(step_name, lang=None, batch_num=None, total_batches=None):
    """Update and display progress information."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if global_progress["start_time"] is None:
        global_progress["start_time"] = time.time()

    elapsed = time.time() - global_progress["start_time"]
    elapsed_str = f"{int(elapsed // 3600):02d}:{int((elapsed % 3600) // 60):02d}:{int(elapsed % 60):02d}"

    if lang:
        global_progress["current_language"] = lang

    global_progress["current_step"] = step_name

    # Calculate estimated time remaining
    if (
        global_progress["completed_languages"] > 0
        and global_progress["total_languages"] > 0
    ):
        avg_time_per_lang = elapsed / global_progress["completed_languages"]
        remaining_langs = (
            global_progress["total_languages"] - global_progress["completed_languages"]
        )

        if batch_num and total_batches:
            # Adjust for partial completion of current language
            remaining_langs -= batch_num / total_batches

        est_remaining = avg_time_per_lang * remaining_langs
        est_remaining_str = f"{int(est_remaining // 3600):02d}:{int((est_remaining % 3600) // 60):02d}:{int(est_remaining % 60):02d}"
    else:
        est_remaining_str = "calculating..."

    if batch_num and total_batches:
        progress_msg = f"[{now}] [{elapsed_str}] {step_name} - Language: {lang} - Batch: {batch_num}/{total_batches}"
        progress_msg += f" - Est. remaining: {est_remaining_str}"

        # Update batch progress
        if batch_num > 0:  # Only count completed batches
            global_progress["completed_batches"] = batch_num

        # Calculate overall progress percentage
        lang_progress = (batch_num / total_batches) * 100
        overall_progress = (
            (global_progress["completed_languages"] + (batch_num / total_batches))
            / global_progress["total_languages"]
        ) * 100

        # Print progress bars
        print(f"\n{progress_msg}")
        print_progress_bar(
            batch_num,
            total_batches,
            prefix=f"Language Progress ({lang}):",
            suffix=f"{batch_num}/{total_batches} batches ({lang_progress:.1f}%)",
            length=40,
        )
        print_progress_bar(
            global_progress["completed_languages"] + (batch_num / total_batches),
            global_progress["total_languages"],
            prefix="Overall Progress:",
            suffix=f'{global_progress["completed_languages"]}/{global_progress["total_languages"]} languages ({overall_progress:.1f}%)',
            length=40,
        )
    else:
        progress_msg = f"[{now}] [{elapsed_str}] {step_name}"
        if lang:
            progress_msg += f" - Language: {lang}"
        if global_progress["completed_languages"] > 0:
            progress_msg += f" - Est. remaining: {est_remaining_str}"

        # Print overall progress if we have multiple languages
        if global_progress["total_languages"] > 1:
            print(f"\n{progress_msg}")
            print_progress_bar(
                global_progress["completed_languages"],
                global_progress["total_languages"],
                prefix="Overall Progress:",
                suffix=f'{global_progress["completed_languages"]}/{global_progress["total_languages"]} languages',
                length=40,
            )
        else:
            print(f"\n{progress_msg}")

    # Flush stdout to ensure progress is displayed immediately
    sys.stdout.flush()


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


def run_command(
    command,
    description=None,
    verbose=True,
    lang=None,
    batch_num=None,
    total_batches=None,
):
    """
    Run a command and print its output with progress tracking.

    Args:
        command: The command to run
        description: Optional description of the command
        verbose: Whether to print the command and its output
        lang: Language code for progress tracking
        batch_num: Current batch number for progress tracking
        total_batches: Total number of batches for progress tracking

    Returns:
        True if the command succeeded, False otherwise
    """
    if description:
        if lang and batch_num and total_batches:
            update_progress(description, lang, batch_num, total_batches)
        elif lang:
            update_progress(description, lang)
        else:
            update_progress(description)

        print(f"\n=== {description} ===\n")

    print(f"Running: {command}")

    # For translation commands, use a different approach to handle tqdm output
    is_translation = "translate_atomic_data.py" in command

    # Show a spinner while the command is running
    start_time = time.time()

    if is_translation:
        # For translation commands, use a direct approach that preserves tqdm output
        print("\nStarting translation process. Progress will be shown below:")
        print("-" * 80)

        # Run the command directly to preserve tqdm output
        result = subprocess.run(command, shell=True, text=True)
        success = result.returncode == 0

        print("-" * 80)
        if not success:
            print(f"Translation process failed with return code {result.returncode}")
        else:
            print("Translation process completed successfully")

        return success
    else:
        # Standard approach for non-translation commands
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Process output in real-time with a simple spinner
        spinner = ["|", "/", "-", "\\"]
        spin_idx = 0
        stdout_lines = []
        stderr_lines = []

        while process.poll() is None:
            # Check for output
            if process.stdout:
                stdout_line = process.stdout.readline()
                if stdout_line:
                    stdout_lines.append(stdout_line)
                    if verbose:
                        print(stdout_line, end="")

            if process.stderr:
                stderr_line = process.stderr.readline()
                if stderr_line:
                    stderr_lines.append(stderr_line)
                    print(f"Error: {stderr_line}", end="")

            # Update spinner every 0.1 seconds
            if (not process.stdout or not stdout_line) and (
                not process.stderr or not stderr_line
            ):
                elapsed = time.time() - start_time
                elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
                print(f"\r{spinner[spin_idx]} Running... {elapsed_str}", end="")
                spin_idx = (spin_idx + 1) % len(spinner)
                time.sleep(0.1)

        # Get any remaining output
        stdout, stderr = process.communicate()
        if stdout:
            stdout_lines.append(stdout)
            if verbose:
                print(stdout, end="")

        if stderr:
            stderr_lines.append(stderr)
            print(f"Error: {stderr}", end="")

    # Clear the spinner line if we were using one (only for non-translation commands)
    if not is_translation:
        print("\r" + " " * 50 + "\r", end="")

        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}")
            return False

    # Update progress after command completes
    if description and lang and batch_num and total_batches:
        update_progress(f"{description} - Completed", lang, batch_num, total_batches)

    return True


def process_language(
    lang,
    requests_per_batch,
    num_batches,
    model,
    atomic_limit,
    skip_steps,
    verbose,
    prepare_only=False,
    skip_prepare_atomic=False,
):
    """
    Process a single language.

    Args:
        lang: Language code
        requests_per_batch: Number of requests per batch
        num_batches: Number of batches to create
        model: OpenAI model to use
        atomic_limit: Limit on the number of atomic entries to include
        skip_steps: List of steps to skip
        verbose: Whether to print verbose output
        prepare_only: If True, only prepare batch files without processing them

    Returns:
        True if all steps succeeded, False otherwise
    """
    print(f"\n{'='*80}\nProcessing language: {lang}\n{'='*80}\n")

    # Update global progress
    global_progress["current_language"] = lang

    # Track start time for this language
    lang_start_time = time.time()

    if prepare_only:
        update_progress(f"Preparing batch files for language: {lang}", lang)
    else:
        update_progress(f"Starting processing for language: {lang}", lang)

    # Step 1: Prepare atomic subset (only needs to be done once)
    atomic_file = Path("templates/atomic10x/atomic10x_als_subset.json")
    if not atomic_file.exists():
        if "prepare" not in skip_steps:
            print(
                f"\n--- Step 1: Preparing atomic subset (this only needs to be done once) ---"
            )
            prepare_cmd = "python scripts/prepare_atomic_subset.py"
            if atomic_limit:
                prepare_cmd += f" --limit {atomic_limit}"

            if not run_command(
                prepare_cmd, "Step 1: Prepare Atomic Subset", verbose, lang
            ):
                return False

            # Check if the file was created
            if not atomic_file.exists():
                print(f"Error: Failed to create atomic subset file: {atomic_file}")
                return False

            print(f"Successfully created atomic subset file: {atomic_file}")
        else:
            print(
                f"Error: Atomic subset file {atomic_file} does not exist and preparation is skipped."
            )
            return False
    else:
        print(f"Atomic subset file already exists: {atomic_file}")
        # Show file size and entry count
        file_size = os.path.getsize(atomic_file) / 1024  # KB
        with open(atomic_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                entry_count = len(data)
                print(f"File contains {entry_count} entries ({file_size:.1f} KB)")
            except json.JSONDecodeError:
                print(f"Warning: Could not parse atomic subset file")

    # Step 2: Translate atomic data
    translated_file = Path(f"templates/atomic10x/atomic10x_als_subset_{lang}.json")

    # Skip translation for English variants
    if lang.startswith("en"):
        print(f"Skipping translation for English variant: {lang}")
        # For English variants, use the original atomic subset file
        if (
            not translated_file.exists()
            and Path("templates/atomic10x/atomic10x_als_subset.json").exists()
        ):
            # Create a symlink or copy the file
            shutil.copy(
                "templates/atomic10x/atomic10x_als_subset.json", translated_file
            )
            print(f"Created copy of English atomic subset for {lang}")
    # Check if translation file already exists
    elif translated_file.exists():
        print(f"Translated file already exists: {translated_file}")
        # Show file size and entry count
        file_size = os.path.getsize(translated_file) / 1024  # KB
        with open(translated_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                entry_count = len(data)
                print(f"File contains {entry_count} entries ({file_size:.1f} KB)")
            except json.JSONDecodeError:
                print("Warning: Could not parse translated file")
    # Need to translate
    elif "translate" not in skip_steps:
        # Check if we should skip translation based on the --skip-prepare-atomic flag
        if skip_prepare_atomic and translated_file.exists():
            print(
                f"Skipping translation for {lang} as file exists and --skip-prepare-atomic was specified"
            )
        else:
            print(f"\n--- Step 2: Translating atomic data for {lang} ---")
            print(
                "This may take several minutes. Translation is done in batches of 25 items."
            )

            # First check if we can find a pre-translated file in a different location
            alt_locations = [
                Path(f"templates/atomic10x/atomic10x_als_subset_{lang}.json.bak"),
                Path(f"data/atomic10x_als_subset_{lang}.json"),
                Path(f"data/templates/atomic10x/atomic10x_als_subset_{lang}.json"),
            ]

            found_alt = False
            for alt_file in alt_locations:
                if alt_file.exists():
                    print(f"Found alternative translation file: {alt_file}")
                    shutil.copy(alt_file, translated_file)
                    print(f"Copied to {translated_file}")
                    found_alt = True
                    break

            if not found_alt:
                # Run the translation with the --skip-existing flag to avoid redoing work
                translate_cmd = f"python scripts/translate_atomic_data.py --lang {lang} --skip-existing"
                if atomic_limit:
                    translate_cmd += f" --limit {atomic_limit}"

                if not run_command(
                    translate_cmd, "Step 2: Translate Atomic Data", verbose, lang
                ):
                    return False

                # Check if the file was created
                if not translated_file.exists():
                    print(f"Error: Failed to create translated file: {translated_file}")
                    return False

                print(f"Successfully created translated file: {translated_file}")
    else:
        print(
            f"Error: Translated file {translated_file} does not exist and translation is skipped."
        )
        return False

    # Create output directories
    Path("batch_output").mkdir(exist_ok=True)
    Path("data/output").mkdir(parents=True, exist_ok=True)

    # Process each batch
    for batch_num in range(1, num_batches + 1):
        batch_start_time = time.time()
        print(f"\n--- Processing batch {batch_num}/{num_batches} for {lang} ---\n")

        # Step 3: Generate batch requests
        if "batch" not in skip_steps:
            print(
                f"\n--- Step 3: Generating batch requests for {lang} (batch {batch_num}/{num_batches}) ---"
            )
            batch_cmd = (
                f"python scripts/batch_openai_prepare.py --lang {lang} "
                f"--num_requests {requests_per_batch} --model {model}"
            )

            if not run_command(
                batch_cmd,
                f"Step 3: Generate Batch Requests (Batch {batch_num}/{num_batches})",
                verbose,
                lang,
                batch_num,
                num_batches,
            ):
                return False

            # Find the most recent batch file
            batch_files = list(
                Path("batch_output").glob(f"batch_requests_{lang}_*.jsonl")
            )
            if not batch_files:
                print(f"Error: No batch files found for language {lang}.")
                return False

            batch_file = max(batch_files, key=os.path.getctime)
            print(f"Using batch file: {batch_file}")

            # Show file size and line count
            file_size = os.path.getsize(batch_file) / 1024  # KB
            with open(batch_file, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            print(f"Batch file contains {line_count} requests ({file_size:.1f} KB)")
        else:
            # Find the most recent batch file
            batch_files = list(
                Path("batch_output").glob(f"batch_requests_{lang}_*.jsonl")
            )
            if not batch_files:
                print(f"Error: No batch files found for language {lang}.")
                return False

            batch_file = max(batch_files, key=os.path.getctime)
            print(f"Using existing batch file: {batch_file}")

            # Show file size and line count
            file_size = os.path.getsize(batch_file) / 1024  # KB
            with open(batch_file, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            print(f"Batch file contains {line_count} requests ({file_size:.1f} KB)")

        # If prepare_only is True, skip the processing and augmentation steps
        if prepare_only:
            update_progress(
                f"Batch file prepared: {batch_file}", lang, batch_num, num_batches
            )
            print("\nTo process this batch file with OpenAI's batch processing system:")
            print(
                f"1. Upload the file {batch_file} to OpenAI's batch processing system"
            )
            print("2. Download the results when processing is complete")
            print(
                "3. Run this script with --process_from_batch <downloaded_results_file>"
            )

            # Calculate elapsed time for this batch
            batch_elapsed = time.time() - batch_start_time
            print(f"Batch preparation completed in {batch_elapsed:.1f} seconds")
            continue

        # Step 4: Process batch responses
        if "process" not in skip_steps:
            print(
                f"\n--- Step 4: Processing batch responses for {lang} (batch {batch_num}/{num_batches}) ---"
            )
            process_cmd = f"python scripts/process_batch.py {batch_file}"

            if not run_command(
                process_cmd,
                f"Step 4: Process Batch Responses (Batch {batch_num}/{num_batches})",
                verbose,
                lang,
                batch_num,
                num_batches,
            ):
                return False

            # Find the response file
            response_file = Path(
                f"{batch_file.parent}/{batch_file.stem}_responses.json"
            )
            if not response_file.exists():
                print(f"Error: Response file {response_file} not found.")
                return False

            print(f"Using response file: {response_file}")

            # Show file size and entry count
            file_size = os.path.getsize(response_file) / 1024  # KB
            with open(response_file, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    entry_count = len(data)
                    print(
                        f"Response file contains {entry_count} responses ({file_size:.1f} KB)"
                    )
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse response file")
        else:
            # Find the response file
            response_file = Path(
                f"{batch_file.parent}/{batch_file.stem}_responses.json"
            )
            if not response_file.exists():
                print(f"Error: Response file {response_file} not found.")
                return False

            print(f"Using existing response file: {response_file}")

            # Show file size and entry count
            file_size = os.path.getsize(response_file) / 1024  # KB
            with open(response_file, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    entry_count = len(data)
                    print(
                        f"Response file contains {entry_count} responses ({file_size:.1f} KB)"
                    )
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse response file")

        # Step 5: Augment AAC data
        if "augment" not in skip_steps:
            print(
                f"\n--- Step 5: Augmenting AAC data for {lang} (batch {batch_num}/{num_batches}) ---"
            )
            # Check if we need to append to an existing file
            output_file = Path(f"data/output/augmented_aac_conversations_{lang}.jsonl")
            augment_cmd = f"python scripts/augment_aac_data.py --input {response_file}"

            if output_file.exists() and batch_num > 1:
                augment_cmd += " --append"
                print(f"Appending to existing output file: {output_file}")
            else:
                print(f"Creating new output file: {output_file}")

            if not run_command(
                augment_cmd,
                f"Step 5: Augment AAC Data (Batch {batch_num}/{num_batches})",
                verbose,
                lang,
                batch_num,
                num_batches,
            ):
                return False

            # Show file size and line count if the file exists
            if output_file.exists():
                file_size = os.path.getsize(output_file) / 1024  # KB
                with open(output_file, "r", encoding="utf-8") as f:
                    line_count = sum(1 for _ in f)
                print(
                    f"Output file contains {line_count} conversations ({file_size:.1f} KB)"
                )

        # Calculate elapsed time for this batch
        batch_elapsed = time.time() - batch_start_time
        print(
            f"Batch {batch_num}/{num_batches} completed in {batch_elapsed:.1f} seconds"
        )

    # Calculate total elapsed time for this language
    lang_elapsed = time.time() - lang_start_time
    minutes, seconds = divmod(lang_elapsed, 60)
    print(f"\n{'='*80}")
    print(f"Completed processing for language: {lang}")
    print(f"Total time: {int(minutes)} minutes and {int(seconds)} seconds")
    print(f"{'='*80}\n")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Wrapper script for the Atomic10x workflow"
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        help=f"Language codes to process (default: {DEFAULT_LANGUAGES})",
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all supported languages"
    )
    parser.add_argument(
        "--requests_per_batch",
        type=int,
        default=DEFAULT_REQUESTS_PER_BATCH,
        help=f"Number of requests per batch (default: {DEFAULT_REQUESTS_PER_BATCH})",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=DEFAULT_NUM_BATCHES,
        help=f"Number of batches to create per language (default: {DEFAULT_NUM_BATCHES})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--atomic_limit",
        type=int,
        default=DEFAULT_ATOMIC_LIMIT,
        help="Limit the number of atomic entries to include",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Alias for --atomic_limit. Limit the number of entries to process (for testing)",
    )
    parser.add_argument(
        "--skip",
        type=str,
        nargs="+",
        choices=["prepare", "translate", "batch", "process", "augment"],
        help="Steps to skip",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only prepare batch files for OpenAI batch processing (don't process them)",
    )
    parser.add_argument(
        "--process_from_batch",
        type=str,
        help="Process results from a batch file downloaded from OpenAI batch processing",
    )
    parser.add_argument(
        "--skip-prepare-atomic",
        action="store_true",
        help="Skip translation step if the translated file already exists",
    )
    args = parser.parse_args()

    # Check if OpenAI API key is set (only needed if not in prepare_only mode)
    if not args.prepare_only and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        return

    # Use limit as atomic_limit if provided
    if args.limit and not args.atomic_limit:
        args.atomic_limit = args.limit
        print(f"Using --limit value ({args.limit}) as atomic_limit")

    # Determine languages to process
    if args.all:
        languages = get_supported_languages()
    elif args.languages:
        languages = args.languages
    else:
        languages = DEFAULT_LANGUAGES

    print(f"Processing languages: {', '.join(languages)}")
    print(f"Requests per batch: {args.requests_per_batch}")
    print(f"Number of batches per language: {args.num_batches}")
    print(f"Model: {args.model}")
    if args.atomic_limit:
        print(f"Atomic limit: {args.atomic_limit}")
    if args.skip:
        print(f"Skipping steps: {', '.join(args.skip)}")
    if args.skip_prepare_atomic:
        print("Skipping atomic data preparation if translated file exists")

    # Create a summary file
    summary = {
        "timestamp": datetime.now().isoformat(),
        "languages": languages,
        "requests_per_batch": args.requests_per_batch,
        "num_batches": args.num_batches,
        "model": args.model,
        "atomic_limit": args.atomic_limit,
        "skipped_steps": args.skip or [],
        "skip_prepare_atomic": args.skip_prepare_atomic,
        "results": {},
    }

    # Initialize global progress tracking
    global_progress["total_languages"] = len(languages)
    global_progress["completed_languages"] = 0
    global_progress["total_batches"] = args.num_batches * len(languages)
    global_progress["completed_batches"] = 0
    global_progress["start_time"] = time.time()

    # Handle process_from_batch option
    if args.process_from_batch:
        batch_file = Path(args.process_from_batch)
        if not batch_file.exists():
            print(f"Error: Batch file {batch_file} not found.")
            return

        update_progress(f"Processing batch results from: {batch_file}")

        # Extract language code from filename (assuming format like batch_responses_en-GB_*.json)
        filename = batch_file.name
        lang_match = filename.split("_")[1] if len(filename.split("_")) > 1 else None

        if not lang_match:
            print(
                "Warning: Could not determine language from filename. Using 'unknown'."
            )
            lang = "unknown"
        else:
            lang = lang_match

        # Process the batch file
        start_time = time.time()

        # Step 1: Process the batch results using process_batch_results.py
        update_progress(f"Step 1: Processing batch results for language: {lang}", lang)
        process_cmd = (
            f"python scripts/process_batch_results.py --batch_file {batch_file}"
        )

        success_process = run_command(
            process_cmd,
            f"Processing batch results for language: {lang}",
            args.verbose,
            lang,
        )

        if not success_process:
            print(f"Error: Failed to process batch results for language: {lang}")
            end_time = time.time()

            # Update summary
            summary["results"][lang] = {
                "success": False,
                "processing_time_seconds": round(end_time - start_time, 2),
                "total_requests": 0,
                "batch_file": str(batch_file),
                "error": "Failed to process batch results",
            }

            print(f"\n{'='*80}\nSummary\n{'='*80}\n")
            print(f"Processed batch file: {batch_file}")
            print("Success: False")
            print(f"Processing time: {round(end_time - start_time, 2)} seconds")

            return

        # Step 2: Augment the processed data using augment_aac_data.py
        update_progress(f"Step 2: Augmenting AAC data for language: {lang}", lang)

        # Determine the input file for augmentation (output from process_batch_results.py)
        # The output file from process_batch_results.py should be in data/output/aac_conversations_{lang}.jsonl
        processed_file = Path(f"data/output/aac_conversations_{lang}.jsonl")

        if not processed_file.exists():
            print(
                f"Warning: Processed file {processed_file} not found. Trying to find alternative..."
            )
            # Try to find any file matching the pattern
            possible_files = list(
                Path("data/output").glob(f"aac_conversations_{lang}*.jsonl")
            )
            if possible_files:
                processed_file = possible_files[0]
                print(f"Found alternative file: {processed_file}")
            else:
                print(f"Error: No processed file found for language: {lang}")
                end_time = time.time()

                # Update summary
                summary["results"][lang] = {
                    "success": False,
                    "processing_time_seconds": round(end_time - start_time, 2),
                    "total_requests": 0,
                    "batch_file": str(batch_file),
                    "error": "No processed file found for augmentation",
                }

                print(f"\n{'='*80}\nSummary\n{'='*80}\n")
                print(f"Processed batch file: {batch_file}")
                print("Success: False")
                print(f"Processing time: {round(end_time - start_time, 2)} seconds")

                return

        # Run the augmentation
        augment_cmd = (
            f"python scripts/augment_aac_data.py --input {processed_file} --lang {lang}"
        )

        success_augment = run_command(
            augment_cmd,
            f"Augmenting AAC data for language: {lang}",
            args.verbose,
            lang,
        )

        end_time = time.time()

        # Update summary
        summary["results"][lang] = {
            "success": success_process and success_augment,
            "processing_time_seconds": round(end_time - start_time, 2),
            "total_requests": 0,  # We don't know how many requests were in the batch
            "batch_file": str(batch_file),
            "processed_file": str(processed_file),
            "augmented": success_augment,
        }

        # Calculate final statistics
        successful_langs = sum(
            1 for lang, result in summary["results"].items() if result["success"]
        )

        # Final progress update
        update_progress(
            f"Batch processing complete! {successful_langs}/{len(summary['results'])} languages successful"
        )

        print(f"\n{'='*80}\nSummary\n{'='*80}\n")
        print(f"Processed batch file: {batch_file}")
        print(f"Processed data saved to: {processed_file}")
        if success_augment:
            print(
                f"Augmented data saved to: data/output/augmented_aac_conversations_{lang}.jsonl"
            )
        print(f"Success: {success_process and success_augment}")
        print(f"Processing time: {round(end_time - start_time, 2)} seconds")

        return

    # Process each language
    for i, lang in enumerate(languages):
        update_progress(f"Starting language {i+1}/{len(languages)}: {lang}")
        start_time = time.time()
        success = process_language(
            lang=lang,
            requests_per_batch=args.requests_per_batch,
            num_batches=args.num_batches,
            model=args.model,
            atomic_limit=args.atomic_limit,
            skip_steps=args.skip or [],
            verbose=args.verbose,
            prepare_only=args.prepare_only,
            skip_prepare_atomic=args.skip_prepare_atomic,
        )
        end_time = time.time()

        # Update completed languages count
        global_progress["completed_languages"] += 1

        # Update summary
        summary["results"][lang] = {
            "success": success,
            "processing_time_seconds": round(end_time - start_time, 2),
            "total_requests": (
                args.requests_per_batch * args.num_batches if success else 0
            ),
            "prepare_only": args.prepare_only,
        }

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = Path(f"batch_output/atomic10x_summary_{timestamp}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Calculate final statistics
    successful_langs = sum(
        1 for lang, result in summary["results"].items() if result["success"]
    )
    failed_langs = sum(
        1 for lang, result in summary["results"].items() if not result["success"]
    )
    total_requests = sum(
        result["total_requests"] for lang, result in summary["results"].items()
    )

    # Final progress update
    if args.prepare_only:
        update_progress(
            f"Batch file preparation complete! {successful_langs}/{len(languages)} languages successful"
        )
    else:
        update_progress(
            f"Processing complete! {successful_langs}/{len(languages)} languages successful"
        )

    print(f"\n{'='*80}\nSummary\n{'='*80}\n")
    print(f"Total languages processed: {len(languages)}")
    print(f"Successful languages: {successful_langs}")
    print(f"Failed languages: {failed_langs}")
    print(f"Total requests: {total_requests}")

    if args.prepare_only:
        print("\nBatch files have been prepared but not processed.")
        print("To process these files with OpenAI's batch processing system:")
        print("1. Upload the batch files to OpenAI's batch processing system")
        print("2. Download the results when processing is complete")
        print("3. Run this script with --process_from_batch <downloaded_results_file>")

    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
