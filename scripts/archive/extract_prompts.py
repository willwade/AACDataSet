#!/usr/bin/env python3
"""
Extract and display prompts from a batch file.
"""
import json
import sys

def extract_prompts(batch_file):
    with open(batch_file, 'r') as f:
        requests = [json.loads(line) for line in f if line.strip()]
    
    for i, req in enumerate(requests):
        prompt = req["body"]["messages"][1]["content"]
        print(f"\n=== PROMPT {i+1} ===\n")
        print(prompt)
        print("\n" + "="*80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_prompts.py <batch_file>")
        sys.exit(1)
    
    extract_prompts(sys.argv[1])
