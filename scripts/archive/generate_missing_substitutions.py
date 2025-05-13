#!/usr/bin/env python3
"""
Generate culturally appropriate substitution files for languages listed in the README.
This script creates localized substitution files for each language, ensuring content
is culturally appropriate and in the local language where possible.
"""
import json
import os
from pathlib import Path
import argparse
from openai import OpenAI
import time

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Constants
SUBSTITUTIONS_DIR = Path("templates/substitutions")
TEMPLATE_FILE = SUBSTITUTIONS_DIR / "en-GB.json"

# List of all languages from README
ALL_LANGUAGES = [
    "af-ZA",
    "ar-SA",
    "eu-ES",
    "ca-ES",
    "hr-HR",
    "cs-CZ",
    "da-DK",
    "nl-BE",
    "nl-NL",
    "en-AU",
    "en-CA",
    "en-NZ",
    "en-ZA",
    "en-GB",
    "en-US",
    "fo-FO",
    "fi-FI",
    "fr-CA",
    "fr-FR",
    "de-AT",
    "de-DE",
    "el-GR",
    "he-IL",
    "it-IT",
    "nb-NO",
    "pl-PL",
    "pt-BR",
    "pt-PT",
    "ru-RU",
    "sk-SK",
    "sl-SI",
    "es-ES",
    "es-US",
    "sv-SE",
    "uk-UA",
    "cy-GB",
    "zh-CN",
    "ja-JP",
    "ko-KR",
]

# Language names for reference
LANGUAGE_NAMES = {
    "af-ZA": "Afrikaans (South Africa)",
    "ar-SA": "Arabic (Saudi Arabia)",
    "eu-ES": "Basque (Spain)",
    "ca-ES": "Catalan (Spain)",
    "hr-HR": "Croatian (Croatia)",
    "cs-CZ": "Czech (Czechia)",
    "da-DK": "Danish (Denmark)",
    "nl-BE": "Dutch (Belgium)",
    "nl-NL": "Dutch (Netherlands)",
    "en-AU": "English (Australia)",
    "en-CA": "English (Canada)",
    "en-NZ": "English (New Zealand)",
    "en-ZA": "English (South Africa)",
    "en-GB": "English (United Kingdom)",
    "en-US": "English (United States)",
    "fo-FO": "Faroese (Faroe Islands)",
    "fi-FI": "Finnish (Finland)",
    "fr-CA": "French (Canada)",
    "fr-FR": "French (France)",
    "de-AT": "German (Austria)",
    "de-DE": "German (Germany)",
    "el-GR": "Greek (Greece)",
    "he-IL": "Hebrew (Israel)",
    "it-IT": "Italian (Italy)",
    "nb-NO": "Norwegian Bokmål (Norway)",
    "pl-PL": "Polish (Poland)",
    "pt-BR": "Portuguese (Brazil)",
    "pt-PT": "Portuguese (Portugal)",
    "ru-RU": "Russian (Russia)",
    "sk-SK": "Slovak (Slovakia)",
    "sl-SI": "Slovenian (Slovenia)",
    "es-ES": "Spanish (Spain)",
    "es-US": "Spanish (United States)",
    "sv-SE": "Swedish (Sweden)",
    "uk-UA": "Ukrainian (Ukraine)",
    "cy-GB": "Welsh (United Kingdom)",
    "zh-CN": "Chinese (China)",
    "ja-JP": "Japanese (Japan)",
    "ko-KR": "Korean (Korea)",
}

# Fields that need localization
FIELDS_TO_LOCALIZE = [
    "time_of_day",
    "setting",
    "tone",
    "relationship",
    "aac_system",
    "topic",
    "aac_user",
    "partner",
]

# Fields that should remain in English
FIELDS_TO_KEEP_ENGLISH = [
    "aac_mlu_length",  # These are technical terms
    "writing_style",  # These are instructions for the model
]


def get_existing_languages():
    """Get list of languages that already have substitution files."""
    existing_languages = []
    for file_path in SUBSTITUTIONS_DIR.glob("*.json"):
        lang_code = file_path.stem
        if "-" in lang_code:  # Only include locale-specific codes
            existing_languages.append(lang_code)
    return existing_languages


def get_missing_languages():
    """Get list of languages that don't have substitution files yet."""
    existing_languages = get_existing_languages()
    return [lang for lang in ALL_LANGUAGES if lang not in existing_languages]


def load_template():
    """Load the template substitution file."""
    with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_localized_content(lang_code, template):
    """Generate culturally appropriate content for the language using OpenAI."""
    language_name = LANGUAGE_NAMES.get(lang_code, lang_code)

    # Create a prompt that includes examples from the template
    prompt = f"""
    I need culturally appropriate content for {language_name} for an AAC (Augmentative and Alternative Communication) dataset.

    Please localize the following content categories to be authentic and culturally appropriate for {language_name}-speaking regions.
    All responses should be IN THE LOCAL LANGUAGE of {language_name}, not in English.

    For each category, I'll provide examples in English, but I need your responses to be in the local language and culturally relevant.

    1. time_of_day: Common times of day (examples in English: morning, afternoon, evening, etc.)

    2. setting: Common places and settings that would be familiar in {language_name}-speaking regions.
       Include local places, cultural venues, religious places relevant to the culture, etc.

    3. tone: Words describing tone of conversation (examples in English: gentle, serious, supportive, etc.)

    4. relationship: Types of relationships between people (examples in English: spouse, partner, friend, etc.)

    5. aac_system: Terms for AAC devices and systems. Translate "AAC" to the local equivalent term if one exists.
       (examples in English: keyboard-based AAC device, eye-gaze device, etc.)

    6. topic: Common conversation topics that would be culturally relevant in {language_name}-speaking regions.
       Include local cultural events, traditions, foods, media, etc.

    7. aac_user: 10 common first names for people in {language_name}-speaking regions who might use AAC

    8. partner: 10 common first names for caregivers/partners in {language_name}-speaking regions

    Format your response as a JSON object with arrays for each category. For example:
    {{
        "time_of_day": ["local term 1", "local term 2", ...],
        "setting": ["local setting 1", "local setting 2", ...],
        ...and so on for each category
    }}

    Make sure all text is in the local language of {language_name}, not English translations.
    Ensure the content is culturally authentic and would be immediately recognizable to locals.

    IMPORTANT: Your response must be valid JSON that can be parsed with json.loads().
    """

    try:
        # First try with GPT-4 if available
        try:
            response = client.chat.completions.create(
                model="gpt-4",  # Using GPT-4 for better cultural understanding
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cultural expert and translator with deep knowledge of languages and cultural contexts worldwide. Respond with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
        except Exception as e:
            # Fall back to GPT-3.5-turbo if GPT-4 is not available
            print(f"Falling back to GPT-3.5-turbo: {e}")
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cultural expert and translator with deep knowledge of languages and cultural contexts worldwide. Respond with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )

        # Extract JSON from the response
        content = response.choices[0].message.content

        # Sometimes the model might include markdown code blocks, so we need to extract the JSON
        if "```json" in content:
            # Extract JSON from markdown code block
            json_content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            # Extract JSON from generic code block
            json_content = content.split("```")[1].split("```")[0].strip()
        else:
            # Use the whole content
            json_content = content

        # Parse the JSON
        result = json.loads(json_content)
        return result
    except Exception as e:
        print(f"Error generating localized content for {lang_code}: {e}")
        # If there's an error, we'll return None and handle it in the calling function
        return None


def create_substitution_file(lang_code, template):
    """Create a culturally appropriate substitution file for the language."""
    print(f"Generating localized content for {lang_code}...")

    # Generate culturally appropriate content
    localized_content = generate_localized_content(lang_code, template)

    if not localized_content:
        print(
            f"Failed to generate localized content for {lang_code}. Using names only."
        )
        # Fall back to just generating names
        try:
            # Create a simpler prompt just for names
            language_name = LANGUAGE_NAMES.get(lang_code, lang_code)

            prompt = f"""
            I need culturally appropriate names for {language_name}.
            Please provide:
            1. 10 common first names for AAC users (people with disabilities) in {language_name}
            2. 10 common first names for caregivers/partners in {language_name}

            Format your response as a JSON object with two arrays:
            {{
                "aac_user": ["Name1", "Name2", ...],
                "partner": ["Name1", "Name2", ...]
            }}

            The names should be authentic and commonly used in {language_name}-speaking regions.
            IMPORTANT: Your response must be valid JSON that can be parsed with json.loads().
            """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant with knowledge of names from different cultures. Respond with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )

            # Extract JSON from the response
            content = response.choices[0].message.content

            # Sometimes the model might include markdown code blocks, so we need to extract the JSON
            if "```json" in content:
                # Extract JSON from markdown code block
                json_content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                # Extract JSON from generic code block
                json_content = content.split("```")[1].split("```")[0].strip()
            else:
                # Use the whole content
                json_content = content

            names = json.loads(json_content)

            # Update only the name fields in the template
            template["aac_user"] = names["aac_user"]
            template["partner"] = names["partner"]
        except Exception as e:
            print(f"Error generating names for {lang_code}: {e}")
            # Keep the template as is if we can't even get names
    else:
        # Update the template with all localized content
        for field in FIELDS_TO_LOCALIZE:
            if field in localized_content and localized_content[field]:
                template[field] = localized_content[field]

    # Save the file
    output_file = SUBSTITUTIONS_DIR / f"{lang_code}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    print(f"Created substitution file for {lang_code}: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate culturally appropriate substitution files"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force regeneration of all files"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of languages to process in each batch",
    )
    parser.add_argument(
        "--delay", type=int, default=10, help="Delay in seconds between batches"
    )
    args = parser.parse_args()

    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        return

    # Create substitutions directory if it doesn't exist
    SUBSTITUTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Get languages to process
    if args.force:
        languages_to_process = ALL_LANGUAGES
    else:
        languages_to_process = get_missing_languages()

    print(
        f"Found {len(languages_to_process)} languages to process: {', '.join(languages_to_process)}"
    )

    # Load template
    template = load_template()

    # Process languages in batches to avoid rate limits
    for i in range(0, len(languages_to_process), args.batch_size):
        batch = languages_to_process[i : i + args.batch_size]
        print(
            f"\nProcessing batch {i//args.batch_size + 1}/{(len(languages_to_process)-1)//args.batch_size + 1}..."
        )

        for lang_code in batch:
            print(
                f"\nProcessing {lang_code} ({LANGUAGE_NAMES.get(lang_code, lang_code)})..."
            )
            create_substitution_file(lang_code, template.copy())

        # Wait between batches to avoid rate limits
        if i + args.batch_size < len(languages_to_process):
            print(f"Waiting {args.delay} seconds before next batch...")
            time.sleep(args.delay)

    print(f"\nCompleted processing {len(languages_to_process)} languages.")


if __name__ == "__main__":
    main()
