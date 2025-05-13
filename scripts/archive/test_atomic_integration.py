#!/usr/bin/env python3
"""
Test script for the atomic10x integration.
This script generates a single prompt using the atomic10x data and prints it.
"""

import json
import random
import argparse
from pathlib import Path
from jinja2 import Template


# Load resources
def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def load_resources(lang_code="en-GB"):
    """Load resources for the specified language."""
    # Load templates
    template_file = "templates/atomic10x/als_template.json"
    templates = load_file(template_file)

    # Load language-specific substitutions
    subs_file = Path(f"templates/substitutions/{lang_code}.json")
    if not subs_file.exists():
        subs_file = Path("templates/atomic10x/als_substitutions.json")
    substitutions = load_file(subs_file)

    # Load language-specific atomic data
    role_map_file = Path(f"templates/atomic10x/atomic10x_als_subset_{lang_code}.json")
    if not role_map_file.exists():
        role_map_file = Path("templates/atomic10x/atomic10x_als_subset.json")
    atomic_entries = load_file(role_map_file)

    return templates, substitutions, atomic_entries


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test the atomic10x integration")
    parser.add_argument(
        "--lang",
        type=str,
        default="en-GB",
        help="Language code (e.g., en-GB, fr-FR, es-ES)",
    )
    args = parser.parse_args()

    # Load resources for the specified language
    templates, substitutions, atomic_entries = load_resources(args.lang)

    # Select a random template and atomic entry
    template = random.choice(templates)
    atomic_entry = random.choice(atomic_entries)

    # Expand the template
    prompt = expand_template(template, substitutions, atomic_entry)

    # Print the results
    print("\n=== ATOMIC ENTRY ===")
    print(json.dumps(atomic_entry, indent=2))
    print("\n=== GENERATED PROMPT ===")
    print(prompt)


if __name__ == "__main__":
    main()
