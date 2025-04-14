import json
import csv

def jsonl_to_csv(jsonl_filepath, csv_filepath):
    """
    Converts a JSON Lines file to a CSV file.

    Args:
        jsonl_filepath (str): The path to the input JSONL file.
        csv_filepath (str): The path to the output CSV file.
    """
    try:
        with open(jsonl_filepath, 'r', encoding='utf-8') as jsonl_file, \
             open(csv_filepath, 'w', newline='', encoding='utf-8') as csv_file:

            first_line = jsonl_file.readline()
            if not first_line:
                print("JSONL file is empty.")
                return

            # Use the first line to get headers
            try:
                first_obj = json.loads(first_line)
                headers = list(first_obj.keys())
            except json.JSONDecodeError:
                print(f"Error decoding JSON from the first line: {first_line.strip()}")
                return
            except AttributeError:
                 print(f"Error: First line ({first_line.strip()}) is not a JSON object (dictionary).")
                 return


            writer = csv.DictWriter(csv_file, fieldnames=headers)
            writer.writeheader()

            # Write the first object
            writer.writerow(first_obj)

            # Process remaining lines
            for line in jsonl_file:
                line = line.strip()
                if line: # Ensure line is not empty
                    try:
                        obj = json.loads(line)
                        # Ensure the object is a dictionary before writing
                        if isinstance(obj, dict):
                             # Only write values for keys present in the header
                            filtered_obj = {k: obj.get(k) for k in headers}
                            writer.writerow(filtered_obj)
                        else:
                             print(f"Skipping line: Not a JSON object (dictionary) - {line}")
                    except json.JSONDecodeError:
                        print(f"Skipping line: Invalid JSON - {line}")

        print(f"Successfully converted '{jsonl_filepath}' to '{csv_filepath}'")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{jsonl_filepath}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Example Usage ---
# Replace 'input.jsonl' with the path to your JSONL file
# Replace 'output.csv' with the desired path for your CSV file
jsonl_to_csv('aac_gemini_conversations_output.jsonl', 'aac_gemini_conversations_output.csv')