#!/usr/bin/env python3
"""
Generate a batch of 20 prompts using the atomic10x data.
"""
import json
import random
from pathlib import Path
from datetime import datetime
from jinja2 import Template

# Constants
NUM_REQUESTS = 20
LANG_CODE = "en-GB"
MODEL = "gpt-4-turbo"
TEMPLATE_FILE = Path("templates/atomic10x/als_template.json")
SUBSTITUTIONS_FILE = Path("templates/atomic10x/als_substitutions.json")
ROLE_MAP_FILE = Path("templates/atomic10x/atomic10x_als_subset.json")
OUTPUT_FILE = Path(
    f"batch_output/batch_20_{LANG_CODE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
)


# Load resources
def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_resources(lang_code):
    """Load resources for the specified language."""
    # Load templates
    template_file = TEMPLATE_FILE
    templates = load_file(template_file)

    # Load language-specific substitutions
    subs_file = Path(f"templates/substitutions/{lang_code}.json")
    if not subs_file.exists():
        subs_file = SUBSTITUTIONS_FILE
    substitutions = load_file(subs_file)

    # Load language-specific atomic data
    role_map_file = Path(f"templates/atomic10x/atomic10x_als_subset_{lang_code}.json")
    if not role_map_file.exists():
        role_map_file = ROLE_MAP_FILE
    role_map = load_file(role_map_file)

    return templates, substitutions, role_map


def expand_template(template_str, substitutions, atomic_entry):
    # Create a new dictionary for substitutions to avoid modifying the original
    subs = {}

    # Always use language-specific names from substitutions file
    aac_user_name = random.choice(substitutions.get("aac_user", ["Alex"]))
    partner_name = random.choice(substitutions.get("partner", ["Taylor"]))

    # Process the topic from atomic_entry - replace PersonX/PersonY with actual names
    topic = atomic_entry.get("topic", "daily care")

    # Replace PersonX with AAC user name and PersonY with partner name in topic
    topic = topic.replace("PersonX", aac_user_name).replace("PersonY", partner_name)

    # Get the relation and which fields from atomic_entry
    relation = atomic_entry.get("relation", "")
    which = atomic_entry.get("which", "")

    # Replace PersonX/PersonY in the which field too
    which = which.replace("PersonX", aac_user_name).replace("PersonY", partner_name)

    # Set key fields with atomic entry values and processed data
    subs["topic"] = topic
    subs["relationship"] = atomic_entry.get("relationship", "caregiver")
    subs["setting"] = atomic_entry.get("setting", "home")
    subs["aac_user"] = aac_user_name
    subs["partner"] = partner_name
    subs["which"] = which
    subs["relation"] = relation

    # Randomly select other substitution values
    for key in ["time_of_day", "tone", "aac_system", "aac_mlu_length", "writing_style"]:
        if key in substitutions:
            subs[key] = random.choice(substitutions[key])

    # Render the template
    template = Template(template_str)
    return template.render(**subs)


def build_request(lang_code, prompt, template_id, model):
    return {
        "custom_id": f"{lang_code}_{template_id}_{datetime.now().timestamp()}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant generating AAC-like conversations for adults with ALS/MND. "
                        "Follow this JSON schema exactly:\n"
                        "{\n"
                        "  'template_id': int,\n"
                        "  'scene': str,\n"
                        "  'conversation': [\n"
                        "    {\n"
                        "      'speaker': str,\n"
                        "      'utterance': str,\n"
                        "      'utterance_intended': str,\n"
                        "      'is_aac_user': bool\n"
                        "    },\n"
                        "    ...\n"
                        "  ]\n"
                        "}\n\n"
                        "For AAC users, 'utterance_intended' should be what they wanted to say "
                        "(complete, grammatical), while 'utterance' should be what they actually "
                        "typed on their device (may be shorter, telegraphic, or have typos). "
                        "For non-AAC users, 'utterance_intended' and 'utterance' should be identical. "
                        "Keep AAC utterances realistic for users with physical limitations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 1500,
            "response_format": {"type": "json_object"},
        },
    }


def main():
    # Create output directory if it doesn't exist
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load resources
    templates, substitutions, role_map = load_resources(LANG_CODE)

    # Generate requests
    batch_requests = []

    # Ensure we use different atomic entries
    atomic_entries = random.sample(role_map, min(NUM_REQUESTS, len(role_map)))
    if len(atomic_entries) < NUM_REQUESTS:
        # If we need more, add random ones
        additional_entries = random.choices(
            role_map, k=NUM_REQUESTS - len(atomic_entries)
        )
        atomic_entries.extend(additional_entries)

    for i in range(NUM_REQUESTS):
        template_id = random.randint(0, len(templates) - 1)
        template = templates[template_id]
        atomic_entry = atomic_entries[i]

        prompt = expand_template(template, substitutions, atomic_entry)
        request = build_request(LANG_CODE, prompt, template_id, MODEL)
        batch_requests.append(request)

        # Print the prompt for inspection
        print(f"\n=== PROMPT {i+1} ===")
        print(f"Template ID: {template_id}")
        print(f"Atomic Entry: {atomic_entry['topic']} - {atomic_entry['which']}")
        print(f"Prompt: {prompt[:200]}...")

    # Save the batch requests
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")

    print(f"\nSaved {len(batch_requests)} batch requests to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
