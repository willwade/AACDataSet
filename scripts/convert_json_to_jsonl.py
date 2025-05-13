import json
import argparse
from pathlib import Path

def convert_json_to_jsonl(input_path, output_path=None):
    """
    Convert a JSON file containing an array of objects to JSONL format,
    where each line is a valid JSON object.
    
    Args:
        input_path (str): Path to the input JSON file
        output_path (str, optional): Path to the output JSONL file. 
            If not provided, will use the same name with .jsonl extension.
    
    Returns:
        Path: Path to the output JSONL file
    """
    input_path = Path(input_path)
    
    # If output path is not provided, generate it from input path
    if output_path is None:
        output_path = input_path.with_suffix('.jsonl')
    else:
        output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read the JSON file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ensure data is an array/list
    if not isinstance(data, list):
        raise ValueError(f"Input file {input_path} does not contain a JSON array")
    
    # Write each object to a separate line in the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return output_path

def process_directory(directory, pattern="*_all_conversations.json", output_dir=None):
    """
    Process all files matching the pattern in the directory and its subdirectories.
    
    Args:
        directory (str): Directory to search for files
        pattern (str): Glob pattern to match files
        output_dir (str, optional): Output directory for JSONL files
    
    Returns:
        int: Number of files processed
    """
    directory = Path(directory)
    count = 0
    
    for input_file in directory.glob(f"**/{pattern}"):
        # Generate output path if output_dir is provided
        if output_dir:
            rel_path = input_file.relative_to(directory)
            out_path = Path(output_dir) / rel_path.with_suffix('.jsonl')
        else:
            out_path = None
        
        try:
            output_file = convert_json_to_jsonl(input_file, out_path)
            print(f"Converted {input_file} -> {output_file}")
            count += 1
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
    
    return count

def main():
    parser = argparse.ArgumentParser(description="Convert JSON files to JSONL format")
    parser.add_argument("--input", type=str, help="Input JSON file or directory")
    parser.add_argument("--output", type=str, help="Output JSONL file (only used if input is a file)")
    parser.add_argument("--pattern", type=str, default="*_all_conversations.json", 
                        help="File pattern to match when input is a directory")
    parser.add_argument("--output-dir", type=str, help="Output directory for JSONL files when processing a directory")
    
    args = parser.parse_args()
    
    if not args.input:
        parser.error("Please provide an input file or directory")
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        output_file = convert_json_to_jsonl(input_path, args.output)
        print(f"Converted {input_path} -> {output_file}")
    elif input_path.is_dir():
        count = process_directory(input_path, args.pattern, args.output_dir)
        print(f"Processed {count} files")
    else:
        print(f"Error: {input_path} does not exist or is not a file/directory")

if __name__ == "__main__":
    main()