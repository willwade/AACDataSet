import json
import os
import argparse


def get_conversation_schema():
    """Return the JSON schema for conversation structure."""
    return {
        "type": "object",
        "properties": {
            "template_id": {"type": "integer"},
            "scene": {"type": "string"},
            "conversation": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string"},
                        "utterance": {"type": "string"},
                        "utterance_intended": {"type": "string"},
                    },
                    "required": ["speaker", "utterance", "utterance_intended"],
                },
            },
        },
        "required": ["template_id", "scene", "conversation"],
    }


def generate_templates(language_code):
    """Generate prompt templates for a specific language."""

    # Load the specific instructions for this language
    instructions_file = f"templates/instructions/{language_code}.json"
    if not os.path.exists(instructions_file):
        print(f"Instructions file not found: {instructions_file}")
        return

    with open(instructions_file, "r", encoding="utf-8") as f:
        instructions = json.load(f)

    # Generate the templates
    templates = []

    # Check if instructions is a dictionary with a 'templates' key
    if isinstance(instructions, dict) and "templates" in instructions:
        instruction_list = instructions["templates"]
    elif isinstance(instructions, dict) and "json" in instructions:
        # Handle the case where the key is 'json' instead of 'templates'
        instruction_list = instructions["json"]
    else:
        # If it's already a list or some other format, use it directly
        instruction_list = instructions

    # Debug output
    print(f"Instruction type: {type(instructions)}")
    if isinstance(instructions, dict):
        print(f"Keys: {list(instructions.keys())}")

    # Make sure instruction_list is a list
    if not isinstance(instruction_list, list):
        print(f"Warning: instruction_list is not a list: {type(instruction_list)}")
        if isinstance(instruction_list, str) and instruction_list == "templates":
            # Special case: if the value is just the string "templates",
            # use the templates from the instructions
            instruction_list = instructions["templates"]
            print(f"Using templates from instructions: {len(instruction_list)} items")
        elif isinstance(instruction_list, str):
            # If it's a string but not "templates", it's probably a mistake
            print(f"Warning: instruction_list is a string: {instruction_list}")
            # Try to use the templates key directly
            if "templates" in instructions:
                instruction_list = instructions["templates"]
                print(
                    f"Using templates from instructions: {len(instruction_list)} items"
                )

    # Make sure we have a list now
    if not isinstance(instruction_list, list):
        print(
            f"Error: Could not convert instruction_list to a list: {instruction_list}"
        )
        instruction_list = []

    for i, instruction in enumerate(instruction_list):
        # Just use the instruction as the template
        templates.append(instruction)

    # Save the templates
    output_dir = "templates/prompt_templates"
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/{language_code}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(templates)} templates for {language_code}")


def get_available_languages():
    """Get a list of all available languages with template instructions."""
    available_langs = []
    template_instructions_dir = "templates/instructions"

    if not os.path.exists(template_instructions_dir):
        return available_langs

    for file in os.listdir(template_instructions_dir):
        if file.endswith(".json"):
            lang_code = file.split(".")[0]  # Remove the .json extension
            available_langs.append(lang_code)

    return available_langs


def main():

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Generate prompt templates for AAC conversations."
    )
    parser.add_argument(
        "--lang",
        type=str,
        help="Language code (e.g., en-GB) to process. "
        "If not provided, all languages will be processed.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode, prompting for language code.",
    )
    args = parser.parse_args()

    # Determine which language(s) to process
    if args.interactive:
        # Interactive mode - prompt for language code
        language_code = input(
            "Enter language code (e.g., en-GB) or press Enter "
            "to process all languages: "
        ).strip()
        if language_code:
            # Process a single language
            generate_templates(language_code)
            return
    elif args.lang:
        # Command-line argument provided - process a single language
        generate_templates(args.lang)
        return

    # Process all available languages
    available_languages = get_available_languages()
    if not available_languages:
        print(
            "No language template instructions found. "
            "Please create at least one template instruction file."
        )
        return

    lang_count = len(available_languages)
    lang_list = ", ".join(available_languages)
    print(f"Found {lang_count} languages with template instructions: {lang_list}")
    print("Processing all languages...")

    for lang in available_languages:
        print(f"\n{'='*50}\nProcessing language: {lang}\n{'='*50}")
        generate_templates(lang)


if __name__ == "__main__":
    main()
