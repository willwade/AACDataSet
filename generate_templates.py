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
    instructions_file = f"template_instructions/{language_code}.json"
    if not os.path.exists(instructions_file):
        print(f"Instructions file not found: {instructions_file}")
        return

    with open(instructions_file, "r", encoding="utf-8") as f:
        instructions = json.load(f)

    # Generate the templates
    templates = []
    for i, instruction in enumerate(instructions):
        # Add schema information to the prompt
        template = f"{instruction}\n\nPlease respond with a JSON object following this schema:\n{json.dumps(get_conversation_schema(), indent=2)}"
        templates.append(template)

    # Save the templates
    output_dir = "prompt_templates"
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/{language_code}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(templates)} templates for {language_code}")


def get_available_languages():
    """Get a list of all available languages with template instructions."""
    available_langs = []
    template_instructions_dir = "template_instructions"

    if not os.path.exists(template_instructions_dir):
        return available_langs

    for file in os.listdir(template_instructions_dir):
        if file.endswith(".json"):
            lang_code = file.split(".")[0]  # Remove the .json extension
            available_langs.append(lang_code)

    return available_langs


def main():
    # Create the necessary directories if they don't exist
    os.makedirs("template_instructions", exist_ok=True)

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Generate prompt templates for AAC conversations."
    )
    parser.add_argument(
        "--lang",
        type=str,
        help="Language code (e.g., en-GB) to process. If not provided, all languages will be processed.",
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
            "Enter language code (e.g., en-GB) or press Enter to process all languages: "
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
            "No language template instructions found. Please create at least one template instruction file."
        )
        return

    print(
        f"Found {len(available_languages)} languages with template instructions: {', '.join(available_languages)}"
    )
    print("Processing all languages...")

    for lang in available_languages:
        print(f"\n{'='*50}\nProcessing language: {lang}\n{'='*50}")
        generate_templates(lang)


if __name__ == "__main__":
    main()
