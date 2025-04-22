import json
from pathlib import Path
import random
import argparse
from datetime import datetime
import time
from jinja2 import Template

# Constants
BATCH_SIZE = 20000  # Number of dialogues per language
MAX_BATCH_FILE_SIZE = 50 * 1024 * 1024  # 50MB in bytes
MAX_REQUEST_SIZE = 4 * 1024 * 1024  # 4MB in bytes
OUTPUT_DIR = Path("output")
BATCH_OUTPUT_DIR = Path("batch_output")
PROMPT_TEMPLATES_DIR = Path("prompt_templates")
SUBSTITUTIONS_DIR = Path("substitutions")
DEFAULT_MODEL = (
    "gpt-4.1-mini"  # Updated to use the more efficient model with higher rate limits
)


def get_conversation_schema():
    """Return the JSON schema for conversation structure, now including is_aac_user."""
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
                        "is_aac_user": {"type": "boolean"},
                    },
                    "required": ["speaker", "utterance", "utterance_intended", "is_aac_user"],
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


def expand_prompt(template, substitutions, atomic_relationship=None):
    """
    Expand a template with random substitutions, including aac_user and partner names.
    Always include AAC system and writing style. Use provided atomic_relationship if available.
    Map ATOMIC10x relation labels (e.g., xReact, oEffect) to human-readable partner roles.
    """
    # Mapping for ATOMIC10x relations to human-readable partner roles
    relation_label_to_role = {
        "xReact": "a friend reacting to the event",
        "oEffect": "someone affected by the event",
        "xWant": "a companion discussing next steps",
        "oWant": "someone who wants something as a result",
        "xIntent": "a friend considering intentions",
        "xNeed": "someone helping with preparations",
        "xEffect": "someone seeing the outcome",
        "xAttr": "someone describing the AAC user",
        "oReact": "someone reacting to the AAC user",
        "oNeed": "someone needing something before the event",
        "isAfter": "someone present after the event",
        "isBefore": "someone present before the event",
        "HinderedBy": "someone or something hindering communication",
    }
    sub_values = {}
    # Always pick AAC user and partner names
    aac_user = random.choice(substitutions.get("aac_user_names", ["Alex"]))
    partner = random.choice(substitutions.get("partner_names", ["Taylor"]))
    sub_values["aac_user"] = aac_user
    sub_values["partner"] = partner
    # Support backward compatibility with PersonX/PersonY
    sub_values["PersonX"] = aac_user
    sub_values["PersonY"] = partner
    # Always include AAC system and writing style
    sub_values["aac_system"] = random.choice(substitutions.get("aac_system", ["keyboard-based AAC device"]))
    sub_values["writing_style"] = random.choice(substitutions.get("writing_style", [
        "The AAC user's messages should be shown as they would appear on their AAC device message bar."
    ]))
    # Relationship: use atomic10x value if provided, else fallback to substitutions
    if atomic_relationship:
        # If it's an ATOMIC10x label, map to human-readable
        sub_values["relationship"] = relation_label_to_role.get(atomic_relationship, atomic_relationship)
    else:
        sub_values["relationship"] = random.choice(substitutions.get("relationship", ["partner"]))
    # Fill in all other substitutions
    for key in substitutions.keys():
        if key not in ("aac_user_names", "partner_names", "aac_system", "writing_style", "relationship"):
            if f"{{{{ {key} }}}}" in template or f"{{{{ {key} }}}}" in template:
                sub_values[key] = random.choice(substitutions[key])
    # Ensure AAC system and writing style are present in the prompt
    if "{aac_system}" not in template:
        template = template.rstrip('.') + f" The AAC system in use is {{aac_system}}."
    if "{writing_style}" not in template:
        template = template.rstrip('.') + f" Writing style: {{writing_style}}."
    try:
        rendered = Template(template).render(**sub_values)
        return rendered.strip(), sub_values
    except Exception as e:
        print(f"Error rendering template with substitutions {sub_values}: {e}")
        return None, None



def prepare_batch_requests(lang_code, num_requests=BATCH_SIZE, model=DEFAULT_MODEL):
    """Prepare batch requests for a specific language."""
    templates, substitutions = load_language_data(lang_code)
    if not templates or not substitutions:
        return None

    # Load ATOMIC10x role map (required for scenario/event/relationship)
    atomic_role_map_path = Path("aac_user_role_map.json")
    atomic_role_map = []
    if atomic_role_map_path.exists():
        with open(atomic_role_map_path, "r") as f:
            try:
                atomic_role_map = json.load(f)
            except Exception:
                atomic_role_map = []
    if not atomic_role_map:
        print("ERROR: ATOMIC10x role map is required for scenario/event/relationship context.")
        return None

    batch_requests = []
    requests_per_template = max(1, num_requests // len(templates))
    remaining = num_requests - (requests_per_template * len(templates))
    max_retries = 1000  # Prevent infinite loops

    for template_id, template in enumerate(templates):
        count = requests_per_template + (1 if remaining > 0 else 0)
        if remaining > 0:
            remaining -= 1

        for _ in range(count):
            # Strictly enforce all scenario/event/relationship context from a single ATOMIC10x entry
            for retry in range(max_retries):
                entry = random.choice(atomic_role_map)
                # Must have topic, relation, aac_user, partner, aac_user_role, which
                required_fields = ["topic", "relation", "aac_user", "partner", "aac_user_role", "which"]
                if all(entry.get(f) for f in required_fields):
                    atomic_relationship = entry["relation"]
                    # Only use scenario/event/relationship from this entry
                    prompt, chosen_subs = expand_prompt(template, substitutions, atomic_relationship=atomic_relationship)
                    # Optionally, you could inject topic/setting/partner/aac_user from entry into the prompt context if templates use those fields
                    # (You could also extend expand_prompt to accept these fields directly)
                    if prompt:
                        request = {
                            "custom_id": f"{lang_code}_{template_id}_{int(time.time())}_{len(batch_requests)}",
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": model,
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": (
                                            f"You are a helpful assistant that generates AAC-like conversations in {lang_code}. "
                                            "Your response must follow this JSON schema: "
                                            + json.dumps(get_conversation_schema())
                                            + f" Important: Use template_id = {template_id} in your response. "
                                            "For each turn in the conversation, set is_aac_user: true if the speaker is the AAC user, and is_aac_user: false otherwise. "
                                            "Be sure to include all required fields for every turn."
                                        ),
                                    },
                                    {"role": "user", "content": prompt},
                                ],
                                "temperature": 0.7,
                                "max_tokens": 1000,
                                "response_format": {"type": "json_object"},
                            },
                        }
                        batch_requests.append(request)
                        break  # Success for this example, move to next
                # If not valid, retry
            else:
                print(f"Warning: Could not find valid ATOMIC10x entry after {max_retries} retries for template {template_id}")

    random.shuffle(batch_requests)
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
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})",
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
        batch_requests = prepare_batch_requests(lang, args.num_requests, args.model)
        if batch_requests:
            output_file = save_batch_requests(lang, batch_requests)
            print(f"Saved {len(batch_requests)} batch requests to {output_file}")
            print(f"Using model: {args.model}")
        else:
            print(f"Failed to prepare batch requests for {lang}")


if __name__ == "__main__":
    main()
