#!/usr/bin/env python3
"""
OpenAI Batch Helper Script

This script automates the process of uploading batch files to OpenAI's API,
checking the status of batch jobs, downloading results, and processing them.

Usage:
    # Upload batch files for all languages
    python scripts/openai_batch_helper.py --upload --all

    # Upload batch files for specific languages
    python scripts/openai_batch_helper.py --upload --lang en-GB fr-FR

    # Check status of all batch jobs
    python scripts/openai_batch_helper.py --check --all

    # Check status of specific batch jobs
    python scripts/openai_batch_helper.py --check --batch-id batch_abc123 batch_def456

    # Download results for completed batch jobs
    python scripts/openai_batch_helper.py --download --all

    # Process downloaded results
    python scripts/openai_batch_helper.py --process --all

    # Complete workflow: upload, check, download, and process
    python scripts/openai_batch_helper.py --workflow --all
"""

import os
import json
import argparse
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from openai import OpenAI

# Constants
BATCH_DIR = Path("batch_files")
BATCH_INFO_FILE = BATCH_DIR / "batch_jobs_info.json"
DEFAULT_COMPLETION_WINDOW = "24h"
DEFAULT_POLL_INTERVAL = 300  # 5 minutes


def setup_client() -> OpenAI:
    """Set up and return an OpenAI client."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    return OpenAI(api_key=api_key)


def load_batch_info() -> Dict[str, Any]:
    """Load batch job information from the info file."""
    if not BATCH_INFO_FILE.exists():
        return {"batch_jobs": {}}

    try:
        with open(BATCH_INFO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error parsing {BATCH_INFO_FILE}. Creating new batch info.")
        return {"batch_jobs": {}}


def save_batch_info(batch_info: Dict[str, Any]) -> None:
    """Save batch job information to the info file."""
    BATCH_INFO_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(BATCH_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(batch_info, f, indent=2)


def find_batch_files(lang_codes: Optional[List[str]] = None) -> Dict[str, List[Path]]:
    """
    Find batch files for the specified languages or all languages.

    Args:
        lang_codes: List of language codes to find batch files for, or None for all

    Returns:
        Dictionary mapping language codes to lists of batch file paths
    """
    if not BATCH_DIR.exists():
        print(f"Batch directory {BATCH_DIR} does not exist")
        return {}

    batch_files = {}

    # If no language codes specified, find all language directories
    if not lang_codes:
        lang_dirs = [d for d in BATCH_DIR.iterdir() if d.is_dir()]
        lang_codes = [d.name for d in lang_dirs]

    for lang in lang_codes:
        lang_dir = BATCH_DIR / lang
        if not lang_dir.exists():
            print(f"Language directory {lang_dir} does not exist")
            continue

        # Find OpenAI batch files
        openai_files = list(lang_dir.glob("openai_batch_*.jsonl"))
        if not openai_files:
            print(f"No OpenAI batch files found in {lang_dir}")
            continue

        batch_files[lang] = openai_files

    return batch_files


def upload_batch_file(client: OpenAI, batch_file: Path) -> Dict[str, Any]:
    """
    Upload a batch file to OpenAI's API and create a batch job.

    Args:
        client: OpenAI client
        batch_file: Path to the batch file

    Returns:
        Dictionary with batch job information
    """
    print(f"Uploading {batch_file}...")

    # Upload the file
    with open(batch_file, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    print(f"File uploaded with ID: {file_obj.id}")

    # Create the batch job
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window=DEFAULT_COMPLETION_WINDOW,
    )

    print(f"Batch job created with ID: {batch.id}")

    # Return batch job information
    return {
        "batch_id": batch.id,
        "file_id": file_obj.id,
        "lang_code": batch_file.parent.name,
        "batch_file": str(batch_file),
        "status": batch.status,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "result_file": None,
    }


def upload_batch_files(lang_codes: Optional[List[str]] = None) -> None:
    """
    Upload batch files for the specified languages or all languages.

    Args:
        lang_codes: List of language codes to upload batch files for, or None for all
    """
    client = setup_client()
    batch_files = find_batch_files(lang_codes)

    if not batch_files:
        print("No batch files found")
        return

    # Load existing batch info
    batch_info = load_batch_info()

    # Upload each batch file
    for lang, files in batch_files.items():
        print(f"Processing {len(files)} batch files for language {lang}")

        for batch_file in files:
            # Check if this file has already been uploaded
            file_path_str = str(batch_file)
            already_uploaded = any(
                job.get("batch_file") == file_path_str
                for job in batch_info["batch_jobs"].values()
            )

            if already_uploaded:
                print(f"Skipping {batch_file} as it has already been uploaded")
                continue

            # Upload the batch file
            job_info = upload_batch_file(client, batch_file)

            # Add to batch info
            batch_info["batch_jobs"][job_info["batch_id"]] = job_info

            # Save batch info after each upload
            save_batch_info(batch_info)

            # Wait a bit to avoid rate limits
            time.sleep(1)


def check_batch_status(batch_ids: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Check the status of batch jobs.

    Args:
        batch_ids: List of batch job IDs to check, or None for all

    Returns:
        Dictionary mapping batch job IDs to their status
    """
    client = setup_client()
    batch_info = load_batch_info()

    if not batch_ids:
        batch_ids = list(batch_info["batch_jobs"].keys())

    if not batch_ids:
        print("No batch jobs found")
        return {}

    statuses = {}
    updated = False

    for batch_id in batch_ids:
        if batch_id not in batch_info["batch_jobs"]:
            print(f"Batch job {batch_id} not found in batch info")
            continue

        job_info = batch_info["batch_jobs"][batch_id]

        try:
            batch = client.batches.retrieve(batch_id)
            status = batch.status

            # Update status in batch info if changed
            if status != job_info["status"]:
                job_info["status"] = status
                updated = True

                # If completed, update completed_at
                if status == "completed":
                    job_info["completed_at"] = datetime.now().isoformat()

            statuses[batch_id] = status
            print(f"Batch job {batch_id} ({job_info['lang_code']}): {status}")

        except Exception as e:
            print(f"Error checking status of batch job {batch_id}: {e}")
            statuses[batch_id] = "error"

    # Save batch info if updated
    if updated:
        save_batch_info(batch_info)

    return statuses


def download_batch_results(batch_ids: Optional[List[str]] = None) -> None:
    """
    Download results for completed batch jobs.

    Args:
        batch_ids: List of batch job IDs to download results for, or None for all
    """
    client = setup_client()
    batch_info = load_batch_info()

    if not batch_ids:
        # Only consider completed jobs that haven't been downloaded yet
        batch_ids = [
            batch_id
            for batch_id, job in batch_info["batch_jobs"].items()
            if job["status"] == "completed" and not job.get("result_file")
        ]

    if not batch_ids:
        print("No completed batch jobs found to download")
        return

    for batch_id in batch_ids:
        if batch_id not in batch_info["batch_jobs"]:
            print(f"Batch job {batch_id} not found in batch info")
            continue

        job_info = batch_info["batch_jobs"][batch_id]

        # Skip if not completed
        if job_info["status"] != "completed":
            print(
                f"Skipping batch job {batch_id} as it is not completed (status: {job_info['status']})"
            )
            continue

        # Skip if already downloaded
        if job_info.get("result_file"):
            print(
                f"Skipping batch job {batch_id} as results have already been downloaded"
            )
            continue

        try:
            print(f"Downloading results for batch job {batch_id}...")

            # Get the batch job
            batch = client.batches.retrieve(batch_id)

            # Get the output file ID
            if not batch.output_file_id:
                print(f"No output file ID found for batch job {batch_id}")
                continue

            # Download the output file
            output_file = client.files.content(batch.output_file_id)

            # Create output directory
            lang_dir = BATCH_DIR / job_info["lang_code"]
            lang_dir.mkdir(parents=True, exist_ok=True)

            # Save the output file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = (
                lang_dir / f"batch_output_{job_info['lang_code']}_{timestamp}.jsonl"
            )

            with open(output_path, "wb") as f:
                f.write(output_file.content)

            print(f"Results saved to {output_path}")

            # Update batch info
            job_info["result_file"] = str(output_path)
            save_batch_info(batch_info)

        except Exception as e:
            print(f"Error downloading results for batch job {batch_id}: {e}")


def process_batch_results(lang_codes: Optional[List[str]] = None) -> None:
    """
    Process downloaded batch results using transform_batch_output.py.

    Args:
        lang_codes: List of language codes to process results for, or None for all
    """
    batch_info = load_batch_info()

    # Filter jobs by language code if specified
    jobs_to_process = []
    for job_id, job in batch_info["batch_jobs"].items():
        if job.get("result_file") and (
            not lang_codes or job["lang_code"] in lang_codes
        ):
            jobs_to_process.append(job)

    if not jobs_to_process:
        print("No downloaded batch results found to process")
        return

    for job in jobs_to_process:
        result_file = job["result_file"]
        lang_code = job["lang_code"]

        # Skip if already processed
        if job.get("processed"):
            print(f"Skipping {result_file} as it has already been processed")
            continue

        print(f"Processing batch results for {lang_code} from {result_file}...")

        try:
            # Run transform_batch_output.py
            transform_script = "scripts/transform_batch_output.py"
            cmd = ["python", transform_script, result_file]

            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

            # Mark as processed
            job["processed"] = True
            save_batch_info(batch_info)

            # Get the transformed file path
            result_path = Path(result_file)
            transformed_file = str(
                result_path.parent / f"{result_path.stem}_transformed.jsonl"
            )

            print(f"Results transformed and saved to {transformed_file}")

            # Run augment_aac_data.py if the transformed file exists
            if Path(transformed_file).exists():
                augment_script = "scripts/augment_aac_data.py"
                cmd = [
                    "python",
                    augment_script,
                    "--input",
                    transformed_file,
                    "--lang",
                    lang_code,
                ]

                print(f"Running: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)

                print(f"Data augmented for {lang_code}")

        except Exception as e:
            print(f"Error processing batch results for {lang_code}: {e}")


def run_workflow(
    lang_codes: Optional[List[str]] = None, poll_interval: int = DEFAULT_POLL_INTERVAL
) -> None:
    """
    Run the complete workflow: upload, check, download, and process.

    Args:
        lang_codes: List of language codes to process, or None for all
        poll_interval: Interval in seconds to poll for batch job status
    """
    # Upload batch files
    print("=== Step 1: Uploading batch files ===")
    upload_batch_files(lang_codes)

    # Check status and download results when complete
    print("=== Step 2: Checking batch job status and downloading results ===")
    all_completed = False

    while not all_completed:
        # Check status of all batch jobs
        statuses = check_batch_status()

        # Download results for completed jobs
        download_batch_results()

        # Check if all jobs are completed
        incomplete_jobs = [
            batch_id
            for batch_id, status in statuses.items()
            if status not in ["completed", "error", "cancelled"]
        ]

        if not incomplete_jobs:
            all_completed = True
            print("All batch jobs completed")
        else:
            print(
                f"{len(incomplete_jobs)} batch jobs still in progress. Checking again in {poll_interval} seconds..."
            )
            time.sleep(poll_interval)

    # Process downloaded results
    print("=== Step 3: Processing batch results ===")
    process_batch_results(lang_codes)

    print("=== Workflow completed ===")


def main():
    parser = argparse.ArgumentParser(description="OpenAI Batch Helper Script")

    # Action arguments
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--upload", action="store_true", help="Upload batch files"
    )
    action_group.add_argument(
        "--check", action="store_true", help="Check batch job status"
    )
    action_group.add_argument(
        "--download", action="store_true", help="Download batch results"
    )
    action_group.add_argument(
        "--process", action="store_true", help="Process batch results"
    )
    action_group.add_argument(
        "--workflow", action="store_true", help="Run complete workflow"
    )

    # Filter arguments
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--all", action="store_true", help="Process all languages"
    )
    filter_group.add_argument("--lang", nargs="+", help="Language code(s) to process")
    filter_group.add_argument(
        "--batch-id", nargs="+", help="Batch job ID(s) to process"
    )

    # Other arguments
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Interval in seconds to poll for batch job status (default: {DEFAULT_POLL_INTERVAL})",
    )

    args = parser.parse_args()

    # Determine language codes
    lang_codes = None
    if args.lang:
        lang_codes = args.lang

    # Run the requested action
    if args.upload:
        upload_batch_files(lang_codes)
    elif args.check:
        check_batch_status(args.batch_id)
    elif args.download:
        download_batch_results(args.batch_id)
    elif args.process:
        process_batch_results(lang_codes)
    elif args.workflow:
        run_workflow(lang_codes, args.poll_interval)


if __name__ == "__main__":
    main()
