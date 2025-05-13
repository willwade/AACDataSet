#!/usr/bin/env python3
"""
Process OpenAI Batch API responses and convert them to our standard JSONL format.

This script takes the OpenAI batch API response file and the corresponding metadata
file, combines them, and outputs a standard JSONL file that can be used with our
augmentation pipeline.

Expected OpenAI Batch API Response Format:
{
    "id": "batch_req_123",
    "custom_id": "20250512_132435_b20eab4e_0",
    "response": {
        "status_code": 200,
        "request_id": "req123",
        "body": {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677693600,
            "model": "gpt-4-turbo",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "..." 
                    },
                    "logprobs": null,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            },
            "service_tier": "default",
            "system_fingerprint": "fp_123"
        }
    },
    "error": null
}

Usage:
    python process_openai_batch_responses.py --responses RESPONSES_FILE --metadata METADATA_FILE --output OUTPUT_FILE

Example:
    python process_openai_batch_responses.py \
        --responses batch_files/en-GB/responses_en-GB_20250512.jsonl \
        --metadata batch_files/en-GB/batch_en-GB_openai_20250512_132116_e05f2ed8_metadata.jsonl \
        --output output/en-GB/en-GB_all_conversations.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to a JSONL file."""
    # Ensure the directory exists
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the data
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def process_openai_batch_responses(
    responses_file: str, 
    metadata_file: str, 
    output_file: str,
    response_format: str = "auto",
    verbose: bool = True
) -> None:
    """
    Process OpenAI batch API responses and convert them to our standard format.
    
    The OpenAI batch API returns responses in a specific format that includes:
    - custom_id: The ID provided in the original request
    - response: Contains the API response with nested body
    - body: Contains the actual response with choices array
    - choices: Array containing the response message
    - message: Contains role and content
    - content: The actual generated conversation in JSON format
    
    Args:
        responses_file: Path to the file containing OpenAI batch API responses
        metadata_file: Path to the file containing request metadata
        output_file: Path to save the processed data
    """
    # Load the responses and metadata
    responses = load_jsonl(responses_file)
    metadata = load_jsonl(metadata_file)
    
    print(f"Loaded {len(responses)} responses and {len(metadata)} metadata entries")
    
    # Create a lookup table for metadata by custom_id
    metadata_by_id = {item['custom_id']: item for item in metadata}
    print(f"Created lookup table with {len(metadata_by_id)} metadata entries")
    
    # Process each response
    processed_data = []
    for i, response in enumerate(responses):
        if verbose:
            print(f"\nProcessing response {i+1}/{len(responses)}")
        
        # Auto-detect response format if not specified
        detected_format = response_format
        if response_format == "auto":
            if 'method' in response and 'url' in response and 'body' in response and 'metadata' in response:
                detected_format = "raw_openai"
                if verbose:
                    print("Auto-detected raw_openai format")
            # Check for OpenAI Batch API direct download format
            elif 'response' in response and isinstance(response['response'], dict) and 'status_code' in response['response'] and 'body' in response['response']:
                detected_format = "openai_batch_direct"
                if verbose:
                    print("Auto-detected OpenAI batch direct download format")
            # Check for another OpenAI batch download format (like the one in cy-GB file)
            elif 'id' in response and 'custom_id' in response and 'response' in response and isinstance(response['response'], dict) and 'body' in response['response']:
                detected_format = "openai_batch_direct"
                if verbose:
                    print("Auto-detected OpenAI batch direct download format (id/custom_id type)")
            else:
                detected_format = "standard"
                if verbose:
                    print("Auto-detected standard format")
                
        # Extract the custom_id and check for valid response structure
        custom_id = response.get('custom_id')
        if not custom_id:
            print(f"Warning: Missing custom_id in response #{i+1}")
            continue
            
        if verbose:
            print(f"Found custom_id: {custom_id}")
        
        # For raw OpenAI format, pre-process to extract relevant content
        if detected_format == "raw_openai":
            # This is the format from raw OpenAI batch response files
            # Transform it into our expected format
            if 'metadata' in response and 'response' in response['metadata']:
                response = {
                    'custom_id': custom_id,
                    'response': {
                        'body': response['metadata']['response']
                    }
                }
                if verbose:
                    print(f"Transformed raw_openai format for {custom_id}")
        # For OpenAI batch direct download format
        elif detected_format == "openai_batch_direct":
            # Format from direct downloading batch results from OpenAI
            # No preprocessing needed, as we'll handle it in the response_body extraction
            if verbose:
                print(f"Using OpenAI batch direct download format for {custom_id}")
            
        # Handle the nested response structure from OpenAI batch API
        # The actual response is in a nested 'body' field
        response_body = None
        
        # Handle OpenAI batch direct download format
        if detected_format == "openai_batch_direct":
            if 'response' in response and 'body' in response['response']:
                # This is the format from directly downloading OpenAI batch results
                response_body_data = response['response']['body']
                
                # Check if body is a dict or string (might be serialized)
                if isinstance(response_body_data, str):
                    try:
                        response_body = json.loads(response_body_data)
                        if verbose:
                            print(f"Parsed response.body from string for {custom_id}")
                    except json.JSONDecodeError:
                        if verbose:
                            print(f"Error parsing response.body as JSON: {response_body_data[:100]}...")
                        continue
                else:
                    response_body = response_body_data
                    if verbose:
                        print(f"Using response.body object for {custom_id}")
                
                # Special handling for format with empty choices array but content in message
                if not response_body.get('choices') and response_body.get('id') and response_body.get('object') == 'chat.completion':
                    if verbose:
                        print(f"Restructuring response body for OpenAI format")
                    # Create a choices structure if none exists
                    response_body['choices'] = [
                        {
                            "message": {
                                "role": "assistant", 
                                "content": response_body.get('content', '')
                            }
                        }
                    ]
        # Handle standard format    
        elif 'response' in response and isinstance(response['response'], dict):
            if 'body' in response['response'] and isinstance(response['response']['body'], dict):
                if 'choices' in response['response']['body']:
                    # Use the nested structure
                    response_body = response['response']['body']
                    if verbose:
                        print(f"Using nested response structure for {custom_id}")
                else:
                    if verbose:
                        print(f"Warning: 'choices' not found in response.body for {custom_id}")
            else:
                if verbose:
                    print(f"Warning: 'body' not found or not a dict in response for {custom_id}")
        elif 'choices' in response:
            # Use the direct structure (older format)
            response_body = response
            if verbose:
                print(f"Using direct response structure for {custom_id}")
        else:
            # Try to extract response data from request-style format
            # This is for batch_*.jsonl files downloaded directly from OpenAI
            if 'id' in response and response['id'].startswith('batch_req_') and 'custom_id' in response and 'response' in response:
                # This appears to be the format in your cy-GB file
                if verbose:
                    print(f"Found batch_req_ format for {custom_id}")
                
                batch_response = response['response']
                if 'body' in batch_response:
                    body_data = batch_response['body']
                    if isinstance(body_data, str):
                        try:
                            response_body = json.loads(body_data)
                            if verbose:
                                print(f"Parsed batch_response.body from string for {custom_id}")
                        except json.JSONDecodeError:
                            print(f"Error parsing batch_response.body as JSON: {body_data[:100]}...")
                            continue
                    else:
                        response_body = body_data
                        if verbose:
                            print(f"Using batch_response.body object for {custom_id}")
            elif 'method' in response and 'url' in response and 'body' in response:
                # This appears to be a request format, not a response format
                print(f"Warning: Found request format instead of response format for {custom_id}")
                
                # Try to extract from body if it's a JSON string
                if isinstance(response['body'], str):
                    try:
                        body_json = json.loads(response['body'])
                        if isinstance(body_json, dict) and 'messages' in body_json:
                            # This might be an input request, not a response
                            print(f"Found request body format, not response format for {custom_id}")
                        else:
                            response_body = body_json
                            print(f"Extracted response from parsed body JSON for {custom_id}")
                    except json.JSONDecodeError:
                        print(f"Could not parse body as JSON for {custom_id}")
                        
                # Try to extract from metadata if available
                elif 'metadata' in response and 'response' in response['metadata']:
                    # Try to extract response from metadata
                    metadata_response = response['metadata']['response']
                    if isinstance(metadata_response, dict) and 'choices' in metadata_response:
                        response_body = metadata_response
                        print(f"Extracted response from metadata.response for {custom_id}")
                    else:
                        print(f"Warning: Could not extract valid response from metadata.response for {custom_id}")
                        print(f"Response keys: {list(response.keys())}")
                        continue
                else:
                    print(f"Warning: Missing proper response structure for {custom_id}")
                    print(f"Response keys: {list(response.keys())}")
                    continue
            else:
                print(f"Warning: Missing proper response structure for {custom_id}")
                print(f"Response keys: {list(response.keys())}")
                continue
        
        # Get the corresponding metadata
        if custom_id not in metadata_by_id:
            print(f"Warning: No metadata found for custom_id: {custom_id}")
            # Try to find a partial match
            partial_matches = [k for k in metadata_by_id.keys() if custom_id.startswith(k) or k.startswith(custom_id)]
            if partial_matches:
                print(f"Found potential metadata matches: {partial_matches}")
                meta = metadata_by_id[partial_matches[0]]
                print(f"Using metadata from {partial_matches[0]} for {custom_id}")
            else:
                continue
        else:
            meta = metadata_by_id[custom_id]
            print(f"Found metadata for {custom_id}")
        
        # Extract metadata from the right location based on the format
        if "metadata" in meta:
            # New format with metadata field
            meta_info = meta["metadata"]
            print(f"Using nested metadata for {custom_id}")
        else:
            # Old format or direct metadata
            meta_info = meta
            print(f"Using direct metadata for {custom_id}")
        
        try:
            # Parse the response content - OpenAI batch API returns data in a specific format
            choices = response_body.get('choices', [])
            if verbose:
                print(f"Found {len(choices)} choices in response")
            
            if choices and len(choices) > 0:
                choice = choices[0]
                
                # Check for message field (standard GPT format)
                if 'message' in choice and 'content' in choice['message']:
                    content = choice['message']['content']
                    if verbose:
                        print(f"Found content in choice.message.content")
                    conversation_data = json.loads(content)
                # Check for text field (alternative format)
                elif 'text' in choice:
                    content = choice['text']
                    if verbose:
                        print(f"Found content in choice.text")
                    conversation_data = json.loads(content)
                else:
                    # Try to find content in other possible locations
                    content = None
                    for key in choice:
                        if isinstance(choice[key], dict) and 'content' in choice[key]:
                            content = choice[key]['content']
                            if verbose:
                                print(f"Found content in choice.{key}.content")
                            break
                    
                    if content:
                        try:
                            conversation_data = json.loads(content)
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error: {e}")
                            print(f"Content preview: {content[:100]}...")
                            conversation_data = {"error": f"Could not parse response: {content[:100]}..."}
                    else:
                        if verbose:
                            print(f"Could not find content in choice: {choice.keys()}")
                        content = str(choice)
                        conversation_data = {"error": f"Could not parse response: {content[:100]}..."}
            else:
                # Fallback for empty choices
                if verbose:
                    print(f"No choices found in response_body")
                content = str(response_body)
                conversation_data = {"error": f"Empty choices array: {content[:100]}..."}
            
            # Add metadata
            if verbose:
                print(f"Adding metadata to conversation data")
            conversation_data['metadata'] = {
                'lang_code': meta_info.get('lang_code', meta.get('lang_code')),
                'generated_at': meta_info.get('created_at', meta.get('created_at')),
                'provider': meta_info.get('provider', meta.get('provider', 'openai')),
                'model': meta_info.get('model', meta.get('model', meta.get('body', {}).get('model', 'gpt-4-turbo')))
            }
            
            # Add to processed data
            processed_data.append(conversation_data)
            if verbose:
                print(f"Added conversation data for {custom_id}")
            
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error processing response for {custom_id}: {e}")
            print(f"Response keys: {list(response.keys())}")
            if 'response' in response:
                print(f"Response.response keys: {list(response['response'].keys())}")
                if 'body' in response['response']:
                    print(f"Response.response.body keys: {list(response['response']['body'].keys())}")
    
    # Save the processed data
    if processed_data:
        save_jsonl(processed_data, output_file)
        print(f"Processed {len(processed_data)} conversations and saved to {output_file}")
    else:
        print(f"Warning: No conversations were successfully processed. Check the logs above for errors.")
        # Create an empty file to avoid later errors
        with open(output_file, 'w') as f:
            f.write('')


def process_directory(
    responses_dir: str,
    metadata_dir: str,
    output_dir: str,
    lang_code: str = None,
    response_format: str = "auto",
    verbose: bool = True
) -> None:
    """
    Process all response files in a directory.
    
    Args:
        responses_dir: Directory containing response files
        metadata_dir: Directory containing metadata files
        output_dir: Directory to save processed files
        lang_code: Optional language code to filter files
    """
    responses_dir = Path(responses_dir)
    metadata_dir = Path(metadata_dir)
    output_dir = Path(output_dir)
    
    # Find all response files
    response_pattern = f"*{lang_code}*.jsonl" if lang_code else "*.jsonl"
    response_files = list(responses_dir.glob(response_pattern))
    
    if not response_files:
        print(f"No response files found in {responses_dir} matching pattern {response_pattern}")
        return
    
    print(f"Found {len(response_files)} response files")
    
    # List all available metadata files
    all_metadata_files = list(metadata_dir.glob("*_metadata.jsonl"))
    print(f"Found {len(all_metadata_files)} total metadata files")
    
    for response_file in response_files:
        print(f"\n{'='*40}")
        print(f"Processing response file: {response_file.name}")
        
        # Try different matching strategies for metadata
        metadata_file = None
        
        # Strategy 1: Extract info from filename and search for matching pattern
        filename = response_file.name
        parts = filename.split('_')
        
        if len(parts) >= 3:
            # Extract language code and timestamp
            try:
                # Try standard format: responses_{lang_code}_{timestamp}.jsonl
                file_lang = parts[1]
                timestamp = parts[2].split('.')[0]
                
                # Find matching metadata file
                metadata_pattern = f"batch_{file_lang}_openai_{timestamp}*_metadata.jsonl"
                print(f"Looking for metadata matching pattern: {metadata_pattern}")
                metadata_files = list(metadata_dir.glob(metadata_pattern))
                
                if metadata_files:
                    metadata_file = metadata_files[0]
                    print(f"Found matching metadata file: {metadata_file.name}")
            except Exception as e:
                print(f"Error parsing filename {filename}: {e}")
        
        # Strategy 2: Look for batch_* files with IDs from the response file
        if not metadata_file:
            print("Trying alternative metadata matching strategy...")
            try:
                # Load the response file to extract custom_ids
                with open(response_file, 'r') as f:
                    first_line = f.readline().strip()
                    response_data = json.loads(first_line)
                    custom_id = response_data.get('custom_id', '')
                    
                    if custom_id:
                        # Extract the batch ID part
                        batch_id_parts = custom_id.split('_')
                        if len(batch_id_parts) >= 4:
                            batch_date = batch_id_parts[0]
                            batch_time = batch_id_parts[1]
                            batch_id = batch_id_parts[2]
                            
                            # Search for metadata files with this batch ID
                            batch_pattern = f"*{batch_date}_{batch_time}_{batch_id}*_metadata.jsonl"
                            print(f"Looking for metadata matching batch ID pattern: {batch_pattern}")
                            matching_files = list(metadata_dir.glob(batch_pattern))
                            
                            if matching_files:
                                metadata_file = matching_files[0]
                                print(f"Found matching metadata file by batch ID: {metadata_file.name}")
            except Exception as e:
                print(f"Error in alternative matching strategy: {e}")
        
        # Strategy 3: Prompt user if no match found
        if not metadata_file:
            print(f"Warning: No matching metadata file found automatically for {response_file.name}")
            print("Available metadata files:")
            for i, mf in enumerate(all_metadata_files):
                print(f"  {i+1}. {mf.name}")
            
            print(f"Please specify the correct metadata file manually and run:")
            print(f"python scripts/process_openai_batch_responses.py --responses {response_file} --metadata <metadata_file> --output <output_file>")
            continue
        
        # Generate output file path
        # Try to extract language code
        lang_code_match = None
        for part in filename.split('_'):
            if '-' in part:  # Most language codes have a hyphen (e.g., en-GB)
                lang_code_match = part
                break
        
        if not lang_code_match:
            # Try to extract from metadata filename
            for part in metadata_file.name.split('_'):
                if '-' in part:
                    lang_code_match = part
                    break
        
        if not lang_code_match:
            # Default to a generic name
            lang_code_match = "output"
        
        # Handle if output_dir is already a file path
        if output_dir.suffix:
            output_file = output_dir
        else:
            output_lang_dir = output_dir / lang_code_match
            output_lang_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_lang_dir / f"{lang_code_match}_all_conversations.jsonl"
        
        # Ensure output directory exists
        output_parent = Path(output_file).parent
        output_parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing: {response_file.name}")
        print(f"Metadata: {metadata_file.name}")
        print(f"Output: {output_file}")
        
        # Process the file
        process_openai_batch_responses(
            str(response_file),
            str(metadata_file),
            str(output_file),
            response_format,
            verbose
        )


def find_latest_output_file(directory, pattern):
    """
    Find the most recent file matching a pattern in a directory.
    
    Args:
        directory: Directory to search in
        pattern: Glob pattern to match
        
    Returns:
        Path to the most recent file or None if not found
    """
    dir_path = Path(directory)
    matching_files = list(dir_path.glob(pattern))
    
    if not matching_files:
        # Try alternative pattern with batch_*output.jsonl (no underscore)
        matching_files = list(dir_path.glob("batch_*output.jsonl"))
        
    if not matching_files:
        return None
        
    # Sort by modification time, newest first
    matching_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(matching_files[0])

def main():
    parser = argparse.ArgumentParser(description="Process OpenAI Batch API responses")
    parser.add_argument("--responses", type=str, help="Path to responses file or directory (from OpenAI batch API)")
    parser.add_argument("--metadata", type=str, help="Path to metadata file or directory (from batch preparation)")
    parser.add_argument("--output", type=str, help="Path to output file or directory (for processed conversations)")
    parser.add_argument("--lang", type=str, help="Language code (e.g. 'en-GB') to process using default paths")
    parser.add_argument("--all-langs", action="store_true", help="Process all available language directories") 
    parser.add_argument("--response-format", type=str, choices=["standard", "raw_openai", "openai_batch_direct", "auto"], 
                        default="auto", help="Format of response files: 'standard' (processed), 'raw_openai' (direct from OpenAI), 'openai_batch_direct' (batch download from OpenAI), or 'auto' (auto-detect)")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity of output")
    
    args = parser.parse_args()
    
    # Handle the --all-langs flag
    if args.all_langs:
        batch_files_dir = Path("batch_files")
        output_dir = Path("output")
        
        if not batch_files_dir.exists():
            parser.error(f"Cannot find batch_files directory at {batch_files_dir}")
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True)
        
        # Process each language directory
        success_count = 0
        lang_dirs = [d for d in batch_files_dir.iterdir() if d.is_dir()]
        for lang_dir in lang_dirs:
            lang_code = lang_dir.name
            try:
                # Find the appropriate response files
                response_file = find_latest_output_file(str(batch_files_dir / lang_code), "batch_*_output.jsonl")
                if not response_file:
                    print(f"Warning: Could not find response file for {lang_code}")
                    print(f"Looking for any file matching batch_*_output.jsonl in {batch_files_dir / lang_code}")
                    continue
                    
                # Find the matching metadata file
                metadata_files = list((batch_files_dir / lang_code).glob("*_metadata.jsonl"))
                if not metadata_files:
                    print(f"Warning: Could not find metadata file for {lang_code}")
                    continue
                metadata_file = str(metadata_files[0])  # Use the first one found
                
                # Create output directory for this language if it doesn't exist
                # Check if output_dir already has a suffix (is a file path)
                if output_dir.suffix:
                    output_file = output_dir
                else:
                    # Create parent directories if needed
                    lang_output_dir = output_dir / lang_code
                    lang_output_dir.mkdir(parents=True, exist_ok=True)
                    output_file = lang_output_dir / f"{lang_code}_all_conversations.jsonl"
                
                # Ensure parent directory exists
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                
                # Process this language directory
                print(f"Processing language: {lang_code}")
                print(f"Using response file: {response_file}")
                print(f"Using metadata file: {metadata_file}")
                print(f"Output will be written to: {output_file}")
                
                # Process the files directly
                process_openai_batch_responses(
                    response_file,
                    metadata_file,
                    str(output_file),
                    "auto" if not hasattr(args, 'response_format') else args.response_format,
                    not args.quiet if hasattr(args, 'quiet') else True
                )
                success_count += 1
            except Exception as e:
                print(f"Error processing language {lang_code}: {e}")
        
        print(f"Successfully processed {success_count} out of {len(lang_dirs)} language directories")
        return
    
    # Handle the --lang option
    if args.lang:
        lang_code = args.lang
        batch_files_dir = Path("batch_files") / lang_code
        output_dir = Path("output") / lang_code
        
        if not batch_files_dir.exists():
            parser.error(f"Cannot find batch files directory for language {lang_code} at {batch_files_dir}")
        
        # Find the appropriate response files
        response_file = find_latest_output_file(str(batch_files_dir), "batch_*_output.jsonl")
        if not response_file:
            print(f"Warning: Could not find response file for {lang_code}")
            print(f"Looking for any file matching batch_*_output.jsonl in {batch_files_dir}")
            return
            
        # Find the matching metadata file
        metadata_files = list(batch_files_dir.glob("*_metadata.jsonl"))
        if not metadata_files:
            print(f"Warning: Could not find metadata file for {lang_code}")
            return
        metadata_file = str(metadata_files[0])  # Use the first one found
            
        # Check if output_dir is already a file path
        if Path(output_dir).suffix:
            output_file = Path(output_dir)
        else:
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{lang_code}_all_conversations.jsonl"
        
        # Process this language
        print(f"Processing language: {lang_code}")
        print(f"Using response file: {response_file}")
        print(f"Using metadata file: {metadata_file}")
        print(f"Output will be written to: {output_file}")
        
        # Process the file directly rather than the directory
        try:
            process_openai_batch_responses(
                response_file,
                metadata_file,
                str(output_file),
                "auto" if not hasattr(args, 'response_format') else args.response_format,
                not args.quiet if hasattr(args, 'quiet') else True
            )
        except Exception as e:
            print(f"Error processing {lang_code}: {e}")
        return
    
    # Traditional mode with explicit paths
    if not args.responses or not args.metadata or not args.output:
        parser.error("Please provide --responses, --metadata, and --output arguments, or use --lang or --all-langs")
    
    response_path = Path(args.responses)
    metadata_path = Path(args.metadata)
    
    # Check if we're processing a single file or a directory
    if response_path.is_file() and metadata_path.is_file():
        # Process a single file
        process_openai_batch_responses(
            args.responses, 
            args.metadata, 
            args.output,
            args.response_format if hasattr(args, 'response_format') else "auto",
            not args.quiet if hasattr(args, 'quiet') else True
        )
    elif response_path.is_dir() and metadata_path.is_dir():
        # Process a directory
        process_directory(
            args.responses, 
            args.metadata, 
            args.output, 
            args.lang,
            args.response_format if hasattr(args, 'response_format') else "auto",
            not args.quiet if hasattr(args, 'quiet') else True
        )
    else:
        parser.error("Both --responses and --metadata must be either files or directories")


if __name__ == "__main__":
    main()