import json
import os

def get_common_suffix(language_code):
    """Load the common suffix for a specific language from file."""
    try:
        with open(f'common_suffixes/{language_code}.txt', 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fall back to English if the language-specific suffix is not found
        try:
            with open('common_suffixes/en-GB.txt', 'r', encoding='utf-8') as f:
                print(f"Warning: No common suffix found for {language_code}, using English (en-GB) instead.")
                return f.read().strip()
        except FileNotFoundError:
            print("Error: Could not find even the fallback common suffix file.")
            return ""


def generate_templates(language_code):
    """Generate prompt templates for a specific language."""

    # Get the appropriate common suffix for this language
    common_suffix = get_common_suffix(language_code)
    if not common_suffix:
        print(f"Error: Could not load common suffix for {language_code}")
        return

    # Load the specific instructions for this language
    instructions_file = f'template_instructions/{language_code}.json'
    if not os.path.exists(instructions_file):
        print(f"Instructions file not found: {instructions_file}")
        return

    with open(instructions_file, 'r', encoding='utf-8') as f:
        instructions = json.load(f)

    # Generate the templates
    templates = []
    for i, instruction in enumerate(instructions):
        template = instruction + ". " + common_suffix.format(template_id=i)
        templates.append(template)

    # Save the templates
    output_dir = 'prompt_templates'
    os.makedirs(output_dir, exist_ok=True)

    output_file = f'{output_dir}/{language_code}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(templates)} templates for {language_code}")

def main():
    # Create the necessary directories if they don't exist
    os.makedirs('template_instructions', exist_ok=True)
    os.makedirs('common_suffixes', exist_ok=True)

    # Example usage
    language_code = input("Enter language code (e.g., en-GB): ")
    generate_templates(language_code)

if __name__ == "__main__":
    main()
