#!/usr/bin/env python3
"""
Run the entire Atomic10x workflow.
This script orchestrates the entire process of generating AAC conversations using the Atomic10x approach.
"""
import argparse
import subprocess
import os
import time
from pathlib import Path
from datetime import datetime

def run_command(command, description=None):
    """
    Run a command and print its output.
    
    Args:
        command: The command to run
        description: Optional description of the command
    """
    if description:
        print(f"\n=== {description} ===\n")
    
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.stderr:
        print(f"Error: {result.stderr}")
    
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        return False
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run the entire Atomic10x workflow")
    parser.add_argument("--lang", type=str, required=True, help="Language code (e.g., fr-FR, es-ES)")
    parser.add_argument("--num_requests", type=int, default=100, help="Number of requests to generate")
    parser.add_argument("--model", type=str, default="gpt-4-turbo", help="OpenAI model to use")
    parser.add_argument("--limit", type=int, help="Limit the number of atomic entries to include")
    parser.add_argument("--skip_prepare", action="store_true", help="Skip preparing the atomic subset")
    parser.add_argument("--skip_translate", action="store_true", help="Skip translating the atomic data")
    parser.add_argument("--skip_batch", action="store_true", help="Skip generating batch requests")
    parser.add_argument("--skip_process", action="store_true", help="Skip processing batch responses")
    parser.add_argument("--skip_augment", action="store_true", help="Skip augmenting AAC data")
    args = parser.parse_args()
    
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        return
    
    # Step 1: Prepare atomic subset
    if not args.skip_prepare:
        prepare_cmd = "python scripts/prepare_atomic_subset.py"
        if args.limit:
            prepare_cmd += f" --limit {args.limit}"
        
        if not run_command(prepare_cmd, "Step 1: Prepare Atomic Subset"):
            return
    
    # Step 2: Translate atomic data
    if not args.skip_translate and args.lang != "en":
        translate_cmd = f"python scripts/translate_atomic_data.py --lang {args.lang}"
        if args.limit:
            translate_cmd += f" --limit {args.limit}"
        
        if not run_command(translate_cmd, "Step 2: Translate Atomic Data"):
            return
    
    # Step 3: Generate batch requests
    if not args.skip_batch:
        batch_cmd = f"python scripts/batch_openai_prepare.py --lang {args.lang} --num_requests {args.num_requests} --model {args.model}"
        
        if not run_command(batch_cmd, "Step 3: Generate Batch Requests"):
            return
        
        # Find the most recent batch file
        batch_files = list(Path("batch_output").glob(f"batch_requests_{args.lang}_*.jsonl"))
        if not batch_files:
            print("Error: No batch files found.")
            return
        
        batch_file = max(batch_files, key=os.path.getctime)
        print(f"Using batch file: {batch_file}")
    else:
        # Find the most recent batch file
        batch_files = list(Path("batch_output").glob(f"batch_requests_{args.lang}_*.jsonl"))
        if not batch_files:
            print("Error: No batch files found.")
            return
        
        batch_file = max(batch_files, key=os.path.getctime)
        print(f"Using existing batch file: {batch_file}")
    
    # Step 4: Process batch responses
    if not args.skip_process:
        process_cmd = f"python scripts/process_batch.py {batch_file}"
        
        if not run_command(process_cmd, "Step 4: Process Batch Responses"):
            return
        
        # Find the most recent response file
        response_file = Path(f"{batch_file.parent}/{batch_file.stem}_responses.json")
        if not response_file.exists():
            print(f"Error: Response file {response_file} not found.")
            return
        
        print(f"Using response file: {response_file}")
    else:
        # Find the most recent response file
        response_file = Path(f"{batch_file.parent}/{batch_file.stem}_responses.json")
        if not response_file.exists():
            print(f"Error: Response file {response_file} not found.")
            return
        
        print(f"Using existing response file: {response_file}")
    
    # Step 5: Augment AAC data
    if not args.skip_augment:
        augment_cmd = f"python scripts/augment_aac_data.py --input {response_file}"
        
        if not run_command(augment_cmd, "Step 5: Augment AAC Data"):
            return
    
    print("\n=== Workflow Complete ===\n")
    print(f"Generated {args.num_requests} conversations for language {args.lang}")
    print(f"Response file: {response_file}")
    print(f"Augmented data: data/output/augmented_aac_conversations_{args.lang}.jsonl")
    
    # Display a sample of the conversations
    display_cmd = f"python scripts/display_conversations.py {response_file} --num 3 --random"
    run_command(display_cmd, "Sample Conversations")

if __name__ == "__main__":
    main()
