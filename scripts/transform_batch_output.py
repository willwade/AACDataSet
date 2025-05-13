#!/usr/bin/env python3
"""
Transform OpenAI batch output file to the format expected by augment_aac_data.py.

This script takes the output from OpenAI batch processing and transforms it into
the format expected by the augmentation script.

Usage:
    python transform_batch_output.py input_file.jsonl output_file.jsonl
"""

import json
import argparse
from pathlib import Path

def transform_batch_output(input_file, output_file):
    """
    Transform OpenAI batch output to the format expected by augment_aac_data.py.
    
    Args:
        input_file: Path to the input batch output file
        output_file: Path to the output file for transformed data
    """
    transformed_data = []
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse the JSON data
                data = json.loads(line)
                
                # Extract the response content
                if 'response' in data and 'body' in data['response']:
                    body = data['response']['body']
                    
                    # Extract the message content from the choices
                    if 'choices' in body and len(body['choices']) > 0:
                        choice = body['choices'][0]
                        if 'message' in choice and 'content' in choice['message']:
                            content = choice['message']['content']
                            
                            # Parse the content as JSON
                            try:
                                content_json = json.loads(content)
                                
                                # Add utterance_intended field if missing
                                if 'conversation' in content_json:
                                    for turn in content_json['conversation']:
                                        if turn.get('is_aac_user', False) and 'utterance_intended' not in turn:
                                            # Use the same text for intended and actual utterance
                                            turn['utterance_intended'] = turn['utterance']
                                
                                # Add the transformed data
                                transformed_data.append(content_json)
                                print(f"Successfully transformed line {line_num}")
                                
                            except json.JSONDecodeError as e:
                                print(f"Error parsing content JSON on line {line_num}: {e}")
                                print(f"Content: {content[:200]}...")
                        else:
                            print(f"No message content found on line {line_num}")
                    else:
                        print(f"No choices found on line {line_num}")
                else:
                    print(f"No response body found on line {line_num}")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
    
    # Write the transformed data
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in transformed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Transformed {len(transformed_data)} conversations and saved to {output_file}")
    
    # Print a sample of the transformed data
    if transformed_data:
        print("\n=== SAMPLE TRANSFORMED DATA ===\n")
        sample = transformed_data[0]
        if "conversation" in sample:
            print("Conversation:")
            for turn in sample["conversation"][:5]:  # Print first 5 turns
                speaker = turn.get("speaker", "Unknown")
                utterance = turn.get("utterance", "")
                utterance_intended = turn.get("utterance_intended", "")
                is_aac_user = turn.get("is_aac_user", False)
                
                print(f"{speaker} ({'AAC User' if is_aac_user else 'Partner'}):")
                if is_aac_user and utterance != utterance_intended:
                    print(f"  Intended: {utterance_intended}")
                    print(f"  Actual: {utterance}")
                else:
                    print(f"  {utterance}")
            
            if len(sample["conversation"]) > 5:
                print("... (more turns)")
        else:
            print(json.dumps(sample, indent=2))

def main():
    parser = argparse.ArgumentParser(description="Transform OpenAI batch output to the format expected by augment_aac_data.py")
    parser.add_argument("input", help="Input batch output file")
    parser.add_argument("--output", help="Output file for transformed data (default: <input>_transformed.jsonl)")
    
    args = parser.parse_args()
    
    input_file = args.input
    if not args.output:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_transformed.jsonl")
    else:
        output_file = args.output
    
    transform_batch_output(input_file, output_file)

if __name__ == "__main__":
    main()
