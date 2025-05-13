#!/usr/bin/env python3
"""
Translate the aac_mlu_length and writing_style fields in all substitution files to the local language.
This script processes existing substitution files and updates these specific fields with translations.
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


def get_all_languages():
    """Get list of all languages that have substitution files."""
    languages = []
    for file_path in SUBSTITUTIONS_DIR.glob("*.json"):
        lang_code = file_path.stem
        if "-" in lang_code:  # Only include locale-specific codes
            languages.append(lang_code)
    return languages


def load_substitution_file(lang_code):
    """Load a substitution file."""
    file_path = SUBSTITUTIONS_DIR / f"{lang_code}.json"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_substitution_file(lang_code, data):
    """Save a substitution file."""
    file_path = SUBSTITUTIONS_DIR / f"{lang_code}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def translate_writing_styles(lang_code, data):
    """Translate aac_mlu_length and writing_style to the local language."""
    language_name = LANGUAGE_NAMES.get(lang_code, lang_code)

    # Skip English variants as they're already in English
    if lang_code.startswith("en-"):
        print(f"Skipping {lang_code} as it's already in English")
        return data

    # Extract current values
    current_mlu_length = data.get("aac_mlu_length", ["short", "medium", "long"])
    current_writing_style = data.get("writing_style", [])

    # Create a prompt for translation
    prompt = f"""
    I need to translate the following terms and descriptions related to AAC (Augmentative and Alternative Communication)
    into {language_name}. In many languages, "AAC" might not be a common term, so please use the local equivalent
    for "communication aid user" or "assistive communication device user" where appropriate.

    1. First, translate these simple terms for message length:
    - short
    - medium
    - long

    2. Then, translate these writing style descriptions:

    a) "The AAC user's messages should be shown as they would appear on their AAC device message bar — short phrases, possibly telegraphic or ungrammatical. Avoid using ellipses ... or texting-style punctuation. AAC messages should use direct, clear language with minimal words (2-4 words per message)."

    b) "The AAC user's messages should be shown as they would appear on their AAC device message bar — abbreviated sentences with some grammar but missing articles or conjunctions. Messages should be somewhat condensed but still maintain basic sentence structure (4-6 words per message)."

    c) "The AAC user's messages should be shown as complete, grammatically correct sentences. These messages represent an AAC user who is computer literate and takes time to construct full sentences with proper grammar and punctuation, similar to typical written communication."

    d) "The AAC user's messages should show a mix of communication styles — sometimes using short telegraphic phrases, other times using more complete sentences. This represents how AAC users might vary their communication style based on energy levels, urgency, or complexity of the message."

    e) "Messages are short, direct, often missing small words. Sentences are 2-5 words long. No texting-style abbreviations or ellipses."

    f) "Messages are a mix of telegraphic phrases and short full sentences. Adjust style naturally based on the message purpose."

    Format your response as a JSON object with two arrays:
    {{
        "aac_mlu_length": ["translated short", "translated medium", "translated long"],
        "writing_style": ["translated style a", "translated style b", "translated style c", "translated style d", "translated style e", "translated style f"]
    }}

    Make sure all translations are in the local language of {language_name} and culturally appropriate.
    IMPORTANT: Your response must be valid JSON that can be parsed with json.loads().
    """

    try:
        # Try with GPT-4 if available
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional translator with expertise in assistive technology terminology. Respond with valid JSON.",
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
                        "content": "You are a professional translator with expertise in assistive technology terminology. Respond with valid JSON.",
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
        translations = json.loads(json_content)

        # Update the data with translations
        if "aac_mlu_length" in translations and translations["aac_mlu_length"]:
            data["aac_mlu_length"] = translations["aac_mlu_length"]

        if "writing_style" in translations and translations["writing_style"]:
            data["writing_style"] = translations["writing_style"]

        return data
    except Exception as e:
        print(f"Error translating writing styles for {lang_code}: {e}")
        # Return the original data if translation fails
        return data


def main():
    parser = argparse.ArgumentParser(
        description="Translate writing styles in substitution files"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        help="Specific language codes to process (default: all)",
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

    # Get languages to process
    if args.languages:
        languages_to_process = args.languages
    else:
        languages_to_process = get_all_languages()

    print(
        f"Found {len(languages_to_process)} languages to process: {', '.join(languages_to_process)}"
    )

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

            # Load the substitution file
            data = load_substitution_file(lang_code)

            # Translate writing styles
            updated_data = translate_writing_styles(lang_code, data)

            # Save the updated file
            save_substitution_file(lang_code, updated_data)

            print(f"Updated substitution file for {lang_code}")

        # Wait between batches to avoid rate limits
        if i + args.batch_size < len(languages_to_process):
            print(f"Waiting {args.delay} seconds before next batch...")
            time.sleep(args.delay)

    print(f"\nCompleted processing {len(languages_to_process)} languages.")


if __name__ == "__main__":
    main()
