#!/usr/bin/env python3
"""
Process a batch file by sending requests to OpenAI and saving the responses.
"""
import json
import os
import time
import argparse
from pathlib import Path
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def process_batch(input_file, output_file):
    """
    Process a batch file by sending requests to OpenAI and saving the responses.
    
    Args:
        input_file: Path to the input batch file
        output_file: Path to the output file for responses
    """
    # Load the batch requests
    with open(input_file, 'r') as f:
        requests = [json.loads(line) for line in f if line.strip()]
    
    print(f"Loaded {len(requests)} requests from {input_file}")
    
    # Process each request
    responses = []
    for i, req in enumerate(requests):
        print(f"Processing request {i+1}/{len(requests)}...")
        
        # Extract the necessary information
        model = req["body"]["model"]
        messages = req["body"]["messages"]
        temperature = req["body"]["temperature"]
        max_tokens = req["body"]["max_tokens"]
        custom_id = req["custom_id"]
        
        try:
            # Send the request to OpenAI
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Extract the response content
            content = response.choices[0].message.content
            
            # Parse the JSON content
            try:
                content_json = json.loads(content)
                
                # Add the response to the list
                responses.append({
                    "custom_id": custom_id,
                    "prompt": messages[1]["content"],
                    "response": content_json
                })
                
                print(f"Successfully processed request {i+1}")
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                print(f"Response content: {content[:200]}...")
                
                # Add the raw response
                responses.append({
                    "custom_id": custom_id,
                    "prompt": messages[1]["content"],
                    "response": content
                })
                
        except Exception as e:
            print(f"Error processing request {i+1}: {e}")
            
            # Add an error entry
            responses.append({
                "custom_id": custom_id,
                "prompt": messages[1]["content"],
                "error": str(e)
            })
        
        # Sleep to avoid rate limits
        time.sleep(1)
    
    # Save the responses
    with open(output_file, 'w') as f:
        json.dump(responses, f, indent=2)
    
    print(f"Saved {len(responses)} responses to {output_file}")
    
    # Print a sample of the responses
    if responses:
        print("\n=== SAMPLE RESPONSE ===\n")
        sample = responses[0]
        if "response" in sample and isinstance(sample["response"], dict):
            # Print the conversation
            if "conversation" in sample["response"]:
                print("Conversation:")
                for turn in sample["response"]["conversation"][:5]:  # Print first 5 turns
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
                
                if len(sample["response"]["conversation"]) > 5:
                    print("... (more turns)")
            else:
                print(json.dumps(sample["response"], indent=2))
        else:
            print(sample.get("response", "No response") if "response" in sample else f"Error: {sample.get('error', 'Unknown error')}")

def main():
    parser = argparse.ArgumentParser(description="Process a batch file by sending requests to OpenAI")
    parser.add_argument("input", help="Input batch file")
    parser.add_argument("--output", help="Output file for responses (default: <input>_responses.json)")
    
    args = parser.parse_args()
    
    input_file = args.input
    if not args.output:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_responses.json")
    else:
        output_file = args.output
    
    process_batch(input_file, output_file)

if __name__ == "__main__":
    main()
