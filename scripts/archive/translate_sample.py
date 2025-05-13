#!/usr/bin/env python3
"""
Translate a small sample of the atomic10x_als_subset.json file to French.
This script translates only the 'topic' and 'which' fields for 20 entries.
"""
import json
import os
import time
from pathlib import Path
from openai import OpenAI
import random

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Constants
SAMPLE_SIZE = 20  # Number of entries to translate
TARGET_LANG = "fr-FR"  # French


def translate_batch(texts, source_lang="en", target_lang="fr-FR"):
    """
    Translate a batch of texts using OpenAI.

    Args:
        texts: List of texts to translate
        source_lang: Source language code
        target_lang: Target language code

    Returns:
        List of translated texts
    """
    if not texts:
        return []

    # Prepare the prompt
    prompt = f"Translate the following texts from {source_lang} to {target_lang}. Keep the meaning intact but make it sound natural in the target language. Return only the translations, one per line, with the same index numbers:\n\n"

    for i, text in enumerate(texts):
        prompt += f"{i+1}. {text}\n"

    try:
        print(f"Sending batch of {len(texts)} texts to OpenAI...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        # Parse the response to extract translations
        translation_text = response.choices[0].message.content
        print(f"Received translation response:\n{translation_text[:200]}...")

        translations = []

        # Parse line by line
        for line in translation_text.strip().split("\n"):
            if line and line[0].isdigit() and ". " in line:
                # Extract the translation part after the index
                translations.append(line.split(". ", 1)[1])

        # Ensure we have the same number of translations as input texts
        if len(translations) != len(texts):
            print(
                f"Warning: Got {len(translations)} translations for {len(texts)} input texts"
            )
            # Fill in missing translations if needed
            while len(translations) < len(texts):
                translations.append(texts[len(translations)])

        return translations

    except Exception as e:
        print(f"Error during translation: {e}")
        # Return original texts as fallback
        return texts


def translate_sample():
    """
    Translate a sample of the atomic subset to French.
    """
    input_file = "templates/atomic10x/atomic10x_als_subset.json"
    output_file = f"templates/atomic10x/atomic10x_als_subset_sample_{TARGET_LANG}.json"

    # Load the atomic subset
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries from {input_file}")

    # Select a random sample
    if len(data) > SAMPLE_SIZE:
        sample_data = random.sample(data, SAMPLE_SIZE)
    else:
        sample_data = data

    print(f"Selected {len(sample_data)} entries for translation")

    # Extract topics and which fields
    topics = [entry["topic"] for entry in sample_data]
    which_fields = [entry["which"] for entry in sample_data]

    print(f"Translating {len(topics)} topics to {TARGET_LANG}...")
    translated_topics = translate_batch(topics, "en", TARGET_LANG)

    print(f"Translating {len(which_fields)} which fields to {TARGET_LANG}...")
    translated_which = translate_batch(which_fields, "en", TARGET_LANG)

    # Create translated data
    translated_data = []
    for i, entry in enumerate(sample_data):
        translated_entry = entry.copy()
        translated_entry["topic"] = translated_topics[i]
        translated_entry["which"] = translated_which[i]
        translated_data.append(translated_entry)

    # Save translated data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(translated_data)} translated entries to {output_file}")

    # Print a few examples for verification
    print("\nExample translations:")
    for i in range(min(5, len(translated_data))):
        print(f"\nOriginal topic: {sample_data[i]['topic']}")
        print(f"Translated topic: {translated_data[i]['topic']}")
        print(f"Original which: {sample_data[i]['which']}")
        print(f"Translated which: {translated_data[i]['which']}")


if __name__ == "__main__":
    translate_sample()
