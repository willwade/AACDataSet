import json
from pathlib import Path
import argparse
from datetime import datetime
import re

OUTPUT_DIR = Path("data/output")
BATCH_OUTPUT_DIR = Path("data/batch_output")
ERRORS_DIR = Path("data/batch_errors")


def attempt_json_fix(json_str):
    """Attempt to fix common JSON parsing errors."""
    # Try some basic fixes
    try:
        # 1. Fix missing quotes around property names
        fixed = re.sub(r"(\s*)([a-zA-Z0-9_]+)(\s*):(\s*)", r'\1"\2"\3:\4', json_str)

        # 2. Fix common escape character issues
        fixed = fixed.replace('\\"', '"').replace('\\"', '"')
        fixed = re.sub(r'([^\\])\\([^"\\/bfnrtu])', r"\1\\\\\2", fixed)

        # 3. Fix missing commas in arrays
        fixed = re.sub(r'(["}\]])(\s*)(["{\[])', r"\1,\2\3", fixed)

        # 4. Fix trailing commas
        fixed = re.sub(r",(\s*)([\]}])", r"\1\2", fixed)

        # 5. Try to fix unterminated strings (more complex)
        # This is a simplistic approach and won't catch all cases
        opens = fixed.count('"')
        if opens % 2 == 1:  # Odd number of quotes
            fixed += '"'

        # Test if the fixed JSON is valid
        json.loads(fixed)
        return fixed, True

    except Exception:
        return json_str, False


def process_batch_results(batch_file):
    """Process batch results and integrate them with existing workflow."""
    if not batch_file.exists():
        print(f"Error: Batch file {batch_file} not found")
        return

    # Create directories
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    ERRORS_DIR.mkdir(exist_ok=True, parents=True)

    # Read the batch results
    with open(batch_file, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]

    processed_count = 0
    error_count = 0
    recovered_count = 0

    # Prepare error log file
    error_log = ERRORS_DIR / f"errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    # Process each result
    for result in results:
        if "response" not in result or "body" not in result["response"]:
            print(f"Missing response data: {result.get('id', 'unknown')}")
            error_count += 1
            continue

        # Extract the conversation data
        try:
            # The body is already a JSON object in the response, not a string
            conversation_data = result["response"]["body"]

            if "choices" not in conversation_data or not conversation_data["choices"]:
                print(f"Missing choices: {result.get('id', 'unknown')}")
                error_count += 1
                continue

            # Extract the actual conversation content (which is a JSON string)
            content = conversation_data["choices"][0]["message"]["content"]

            try:
                # First try to parse as-is
                conversation = json.loads(content)
            except json.JSONDecodeError as e:
                # Try to fix common JSON errors
                fixed_content, success = attempt_json_fix(content)
                if success:
                    conversation = json.loads(fixed_content)
                    recovered_count += 1
                    print(f"Recovered JSON for: {result.get('id', 'unknown')}")
                else:
                    # Log the error and continue to next result
                    with open(error_log, "a", encoding="utf-8") as f:
                        error_info = {
                            "id": result.get("id", "unknown"),
                            "custom_id": result.get("custom_id", "unknown"),
                            "error": str(e),
                            "content": content,
                        }
                        f.write(json.dumps(error_info) + "\n")
                    print(f"JSON error: {result.get('id', 'unknown')}: {e}")
                    error_count += 1
                    continue

            # Add metadata
            conversation["metadata"] = {
                "batch_id": result["id"],
                "custom_id": result["custom_id"],
                "model": conversation_data.get("model", "unknown"),
                "processed_at": datetime.now().isoformat(),
            }

            # Determine the language from the custom_id
            lang_code = result["custom_id"].split("_")[0]
            output_file = OUTPUT_DIR / f"aac_conversations_{lang_code}.jsonl"

            # Append to the existing output file
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(conversation) + "\n")

            processed_count += 1

        except Exception as e:
            # Save the problematic content for review
            with open(error_log, "a", encoding="utf-8") as f:
                error_info = {
                    "id": result.get("id", "unknown"),
                    "custom_id": result.get("custom_id", "unknown"),
                    "error": str(e),
                    "raw_result": result,
                }
                f.write(json.dumps(error_info) + "\n")
            print(f"Error: {result.get('id', 'unknown')}: {e}")
            error_count += 1

    print(
        f"Processing complete: {processed_count} processed, "
        f"{recovered_count} recovered, {error_count} errors"
    )
    if error_count > 0:
        print(f"Error details saved to: {error_log}")


def main():
    parser = argparse.ArgumentParser(description="Process OpenAI Batch API results")
    parser.add_argument(
        "--batch_file", type=str, required=True, help="Path to the batch results file"
    )
    args = parser.parse_args()

    batch_file = Path(args.batch_file)
    process_batch_results(batch_file)


if __name__ == "__main__":
    main()
