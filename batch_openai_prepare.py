import json
from pathlib import Path
import random
import argparse
from datetime import datetime
import time

# Constants
BATCH_SIZE = 20000  # Number of dialogues per language
MAX_BATCH_FILE_SIZE = 50 * 1024 * 1024  # 50MB in bytes
MAX_REQUEST_SIZE = 4 * 1024 * 1024  # 4MB in bytes
OUTPUT_DIR = Path("output")
BATCH_OUTPUT_DIR = Path("batch_output")
PROMPT_TEMPLATES_DIR = Path("prompt_templates")
SUBSTITUTIONS_DIR = Path("substitutions")


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


def get_language_paths(lang_code):
    """Get file paths for a given language code, handling locale variants."""
    if "-" in lang_code:
        base_lang, locale = lang_code.split("-", 1)
        output_file = f"aac_conversations_{lang_code}.jsonl"
        template_paths = [
            PROMPT_TEMPLATES_DIR / f"{lang_code}.json",
            PROMPT_TEMPLATES_DIR / f"{base_lang}.json",
        ]
        substitution_paths = [
            SUBSTITUTIONS_DIR / f"{lang_code}.json",
            SUBSTITUTIONS_DIR / f"{base_lang}.json",
        ]
    else:
        output_file = f"aac_conversations_{lang_code}.jsonl"
        template_paths = [PROMPT_TEMPLATES_DIR / f"{lang_code}.json"]
        substitution_paths = [SUBSTITUTIONS_DIR / f"{lang_code}.json"]

    return output_file, template_paths, substitution_paths


def load_language_data(lang_code):
    """Load templates and substitutions for a specific language."""
    output_file, template_paths, substitution_paths = get_language_paths(lang_code)

    # Load templates
    templates = []
    for template_path in template_paths:
        if template_path.exists():
            with open(template_path, "r", encoding="utf-8") as f:
                templates = json.load(f)
            break

    if not templates:
        print(f"Warning: No templates found for language '{lang_code}'")
        return None, None

    # Load substitutions
    substitutions = {}
    for substitution_path in substitution_paths:
        if substitution_path.exists():
            with open(substitution_path, "r", encoding="utf-8") as f:
                substitutions = json.load(f)
            break

    if not substitutions:
        print(f"Warning: No substitutions found for language '{lang_code}'")
        return None, None

    return templates, substitutions


def expand_prompt(template, substitutions):
    """Expand a template with random substitutions."""
    sub_values = {}
    for key in substitutions.keys():
        if f"{{{{ {key} }}}}" in template:
            sub_values[key] = random.choice(substitutions[key])

    if "{ writing_style }" in template and "writing_style" in sub_values:
        template = template.replace("{ writing_style }", sub_values["writing_style"])

    return template, sub_values


def prepare_batch_requests(lang_code, num_requests=BATCH_SIZE):
    """Prepare batch requests for a specific language."""
    templates, substitutions = load_language_data(lang_code)
    if not templates or not substitutions:
        return None

    batch_requests = []
    for _ in range(num_requests):
        template = random.choice(templates)
        prompt, chosen_subs = expand_prompt(template, substitutions)

        if prompt:
            request = {
                "custom_id": f"{lang_code}_{int(time.time())}_{len(batch_requests)}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4-turbo-preview",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that generates "
                            "AAC-like conversations. Your response must "
                            "follow this JSON schema: "
                            + json.dumps(get_conversation_schema()),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "response_format": {"type": "json_object"},
                },
            }
            batch_requests.append(request)

    return batch_requests


def save_batch_requests(lang_code, batch_requests):
    """Save batch requests to a file."""
    BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = BATCH_OUTPUT_DIR / f"batch_requests_{lang_code}_{timestamp}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Prepare batch requests for OpenAI Batch API"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="all",
        help="Language code (e.g., 'en', 'es') or 'all' for all languages",
    )
    parser.add_argument(
        "--num_requests",
        type=int,
        default=BATCH_SIZE,
        help="Number of requests to generate per language",
    )
    args = parser.parse_args()

    languages = []
    if args.lang.lower() == "all":
        # Get all available languages
        for template_file in PROMPT_TEMPLATES_DIR.glob("*.json"):
            lang_code = template_file.stem
            if (SUBSTITUTIONS_DIR / f"{lang_code}.json").exists():
                languages.append(lang_code)
    else:
        languages = [args.lang]

    for lang in languages:
        print(f"\nPreparing batch requests for language: {lang}")
        batch_requests = prepare_batch_requests(lang, args.num_requests)
        if batch_requests:
            output_file = save_batch_requests(lang, batch_requests)
            print(f"Saved {len(batch_requests)} batch requests to {output_file}")
        else:
            print(f"Failed to prepare batch requests for {lang}")


if __name__ == "__main__":
    main()
