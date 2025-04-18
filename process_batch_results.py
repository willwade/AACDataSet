import json
from pathlib import Path
import argparse
from datetime import datetime

OUTPUT_DIR = Path("output")
BATCH_OUTPUT_DIR = Path("batch_output")


def process_batch_results(batch_file):
    """Process batch results and integrate them with existing workflow."""
    if not batch_file.exists():
        print(f"Error: Batch file {batch_file} not found")
        return

    # Read the batch results
    with open(batch_file, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]

    # Process each result
    for result in results:
        if "response" not in result or "body" not in result["response"]:
            continue

        # Extract the conversation data
        try:
            conversation_data = json.loads(result["response"]["body"])
            if "choices" not in conversation_data or not conversation_data["choices"]:
                continue

            # Extract the actual conversation
            content = conversation_data["choices"][0]["message"]["content"]
            conversation = json.loads(content)

            # Add metadata
            conversation["metadata"] = {
                "batch_id": result["id"],
                "custom_id": result["custom_id"],
                "processed_at": datetime.now().isoformat(),
            }

            # Determine the language from the custom_id
            lang_code = result["custom_id"].split("_")[0]
            output_file = OUTPUT_DIR / f"aac_conversations_{lang_code}.jsonl"

            # Append to the existing output file
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(conversation) + "\n")

        except Exception as e:
            print(f"Error processing result {result.get('id', 'unknown')}: {e}")


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
