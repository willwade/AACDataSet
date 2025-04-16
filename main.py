# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "jinja2",
#     "llm",
#     "llm-gemini",
#     "llm-ollama",  # Added Ollama plugin
#     "tqdm",
#     # Note: OpenAI support is built into llm, no separate plugin needed
# ]
# ///


# This script is designed to create a corpora of AAC-like conversations. We are using Google's Gemini free tier and as such have rate limiting
# Right now the aac prompts is really the key
# We will create these in different languages going forward


import time
import json
import random
import llm
import argparse  # Import argparse
from tqdm import tqdm
from pathlib import Path
from jinja2 import Template

# --- Configuration --- (Consider moving more constants here or to a config file)
DEFAULT_LANGUAGE = "en"
MAX_REQUESTS_PER_DAY = 1500
MAX_REQUESTS_PER_MINUTE = 15
SLEEP_BETWEEN_REQUESTS = 60 / MAX_REQUESTS_PER_MINUTE
OUTPUT_DIR = Path("output")  # Create an output directory
PROMPT_TEMPLATES_DIR = Path("prompt_templates")  # Directory for prompt templates
SUBSTITUTIONS_DIR = Path("substitutions")  # Directory for substitutions

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Generate AAC-like conversations using LLM."
)
parser.add_argument(
    "--lang",
    type=str,
    default="all",  # Changed default to 'all' to process all languages by default
    help=f"Language code for prompts and substitutions (e.g., 'en', 'es'). Use 'all' to process all available languages (default).",
)
parser.add_argument(
    "--num_variations",
    type=int,
    default=3,
    help="Number of variations to generate per template.",
)
parser.add_argument(
    "--model",
    type=str,
    default="gemini-2.0-flash",
    help="LLM model to use. Options: 'gemini-2.0-flash', 'mistral-7b', 'mistral-medium', 'llama3', etc.",
)
parser.add_argument(
    "--provider",
    type=str,
    default="gemini",
    choices=["gemini", "ollama", "openai"],
    help="LLM provider to use. Options: 'gemini', 'ollama', 'openai'.",
)
args = parser.parse_args()

# --- Detect available languages ---
def get_available_languages():
    """Detect all available language files in the prompt_templates directory."""
    available_langs = []
    for template_file in PROMPT_TEMPLATES_DIR.glob("*.json"):
        lang_code = template_file.stem  # Get filename without extension
        # Check if corresponding substitution file exists
        if (SUBSTITUTIONS_DIR / f"{lang_code}.json").exists():
            available_langs.append(lang_code)
    return available_langs

# --- Setup Paths based on Language ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

# Handle locale-specific language codes (e.g., en-GB, es-ES)
def get_language_paths(lang_code):
    """Get file paths for a given language code, handling locale variants."""
    # Check if this is a locale-specific code (e.g., en-GB, es-ES)
    if '-' in lang_code:
        base_lang, locale = lang_code.split('-', 1)
        # Try locale-specific files first, fall back to base language if not found
        output_file = f"aac_conversations_{lang_code}.jsonl"
        template_paths = [
            PROMPT_TEMPLATES_DIR / f"{lang_code}.json",  # Try en-GB.json first
            PROMPT_TEMPLATES_DIR / f"{base_lang}.json"    # Fall back to en.json
        ]
        substitution_paths = [
            SUBSTITUTIONS_DIR / f"{lang_code}.json",     # Try en-GB.json first
            SUBSTITUTIONS_DIR / f"{base_lang}.json"       # Fall back to en.json
        ]
    else:
        # For non-locale codes, use the standard path
        output_file = f"aac_conversations_{lang_code}.jsonl"
        template_paths = [PROMPT_TEMPLATES_DIR / f"{lang_code}.json"]
        substitution_paths = [SUBSTITUTIONS_DIR / f"{lang_code}.json"]

    return output_file, template_paths, substitution_paths

# --- Load templates and substitutions for a specific language ---
def load_language_data(lang_code):
    """Load templates and substitutions for a specific language."""
    output_file, template_paths, substitution_paths = get_language_paths(lang_code)
    output_path = OUTPUT_DIR / output_file

    # --- Load Prompts ---
    # Try each template path in order until one works
    template_loaded = False
    templates = []
    for template_path in template_paths:
        if template_path.exists():
            with open(template_path, "r", encoding="utf-8") as f:
                templates = json.load(f)
            print(f"Loaded {len(templates)} prompt templates from {template_path}")
            template_loaded = True
            break

    if not template_loaded:
        print(f"Warning: Prompt templates file not found for language '{lang_code}' in any of these locations: {template_paths}")
        return None, None, None

    # --- Load Substitutions ---
    # Try each substitution path in order until one works
    substitution_loaded = False
    substitutions = {}
    for substitution_path in substitution_paths:
        if substitution_path.exists():
            with open(substitution_path, "r", encoding="utf-8") as f:
                substitutions = json.load(f)
            print(f"Loaded substitutions from {substitution_path}")
            substitution_loaded = True
            break

    if not substitution_loaded:
        print(f"Warning: Substitutions file not found for language '{lang_code}' in any of these locations: {substitution_paths}")
        return None, None, None

    return templates, substitutions, output_path

# --- Setup Jinja Environment --- (More robust way to handle templates)
# Consider putting templates in separate .jinja2 files if they get complex
# env = Environment(loader=FileSystemLoader(PROMPT_TEMPLATES_DIR))


# --- Generate concrete prompts --- (Updated to handle potentially missing keys and writing style)
def expand_prompt(
    template, template_id="unknown", substitutions=None
):  # Add template_id for better context and pass substitutions
    if substitutions is None:
        print(f"Error: No substitutions provided for template {template_id}")
        return None, None

    sub_values = {}
    # Randomly select values for keys present in BOTH template and substitutions
    for key in substitutions.keys():
        # Basic check if the template likely uses the key
        if f"{{{{ {key} }}}}" in template:
            sub_values[key] = random.choice(substitutions[key])

    # Special handling for writing style - replace { writing_style } with the actual writing style
    if "{ writing_style }" in template and "writing_style" in sub_values:
        template = template.replace("{ writing_style }", sub_values["writing_style"])

    # Render the template
    try:
        rendered = Template(template).render(**sub_values)
    except Exception as e:
        print(
            f"Error rendering template {template_id} with substitutions {sub_values}: {e}"
        )
        return None, None  # Indicate error

    # Prepare metadata including chosen substitutions
    metadata_subs = {k: v for k, v in sub_values.items() if v is not None}

    return rendered.strip(), metadata_subs  # Return rendered prompt and chosen subs


# This section has been moved to the generate_conversations_for_language function

# --- Get LLM model based on provider ---
def get_llm_model(provider, model_name):
    """Get LLM model based on provider and model name."""
    try:
        if provider == "gemini":
            print(f"Using Gemini model: {model_name}")
            return llm.get_model(model_name)
        elif provider == "ollama":
            # For Ollama, we just use the model name as registered in llm
            # If a specific version is not specified, use the model name directly
            if ":" not in model_name:
                # Try with just the model name first
                try:
                    print(f"Using Ollama model: {model_name}")
                    return llm.get_model(model_name)
                except llm.UnknownModelError:
                    # If that fails, try with :latest suffix
                    model_name = f"{model_name}:latest"
                    print(f"Using Ollama model: {model_name}")
                    return llm.get_model(model_name)
            else:
                # Model name already has a version specified
                print(f"Using Ollama model: {model_name}")
                return llm.get_model(model_name)
        elif provider == "openai":
            # For OpenAI, we don't need to prefix the model name
            # Just use the model name directly
            print(f"Using OpenAI model: {model_name}")
            return llm.get_model(model_name)
        else:
            print(f"Error: Unsupported provider '{provider}'")
            exit(1)
    except llm.UnknownModelError:
        print(f"Error: Model '{model_name}' not found for provider '{provider}'.")
        print(f"Ensure the llm-{provider} plugin is installed and configured.")
        exit(1)
    except Exception as e:
        print(f"Error getting LLM model: {e}")
        exit(1)


# --- Rate limiting ---
request_timestamps = []


def rate_limit():
    # (Rate limiting logic remains the same)
    now = time.time()
    request_timestamps.append(now)
    # --- snip --- (same as before)
    recent = [t for t in request_timestamps if now - t < 60]
    request_timestamps[:] = recent
    if len(recent) >= MAX_REQUESTS_PER_MINUTE:
        wait = max(0, 60 - (now - recent[0]))  # Ensure wait is not negative
        print(f"Rate limit hit. Sleeping for {wait:.2f}s...")
        time.sleep(wait)


# --- Generate conversations for a specific language ---
def generate_conversations_for_language(lang_code, num_variations):
    """Generate conversations for a specific language."""
    # Get LLM model for this generation run
    model = get_llm_model(args.provider, args.model)

    # Load language-specific data
    templates, substitutions, output_path = load_language_data(lang_code)
    if templates is None or substitutions is None or output_path is None:
        print(f"Skipping language '{lang_code}' due to missing files.")
        return 0

    # --- Load existing results --- (Adjust path)
    # We'll use a set of (template_id, substitutions) tuples to avoid duplicates
    existing_templates = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "metadata" in data and "template_id" in data["metadata"] and "substitutions_used" in data["metadata"]:
                        # Create a tuple of template_id and a frozenset of substitution items
                        template_id = data["metadata"]["template_id"]
                        subs = data["metadata"]["substitutions_used"]
                        # Convert dict to a frozenset of tuples for hashability
                        subs_tuple = frozenset((k, v) for k, v in subs.items())
                        existing_templates.add((template_id, subs_tuple))
                except Exception:
                    continue
    print(f"Loaded {len(existing_templates)} existing template combinations to avoid duplicates.")

    # --- Main generation loop ---
    generated_count = 0
    max_to_generate = min(MAX_REQUESTS_PER_DAY, len(templates) * num_variations)
    print(f"Attempting to generate up to {max_to_generate} new conversations for language '{lang_code}'...")

    # Use enumerate to get an index for templates if needed (e.g., for template_id)
    template_list = [(idx, tpl) for idx, tpl in enumerate(templates)]

    with open(output_path, "a", encoding="utf-8") as out_file:
        # Loop goal is to generate 'max_to_generate' *new* items
        # We might need more attempts if many prompts are duplicates or fail
        pbar = tqdm(total=max_to_generate, desc=f"Generating {lang_code} Conversations")
        attempts = 0
        max_attempts = max_to_generate * 3  # Try up to 3x times to find non-duplicates

        while generated_count < max_to_generate and attempts < max_attempts:
            attempts += 1
            template_id, template = random.choice(
                template_list
            )  # Get template and its index/ID
            prompt, chosen_substitutions = expand_prompt(template, template_id, substitutions)

            if prompt is None:  # Skip if rendering failed
                continue

            # Check if this template_id + substitutions combination already exists
            subs_tuple = frozenset((k, v) for k, v in chosen_substitutions.items())
            if (template_id, subs_tuple) in existing_templates:
                continue  # Skip duplicate

            rate_limit()
            try:
                # Call the LLM with retry logic for quota/rate limit errors
                max_retries = 3
                raw_response_text = None

                for attempt in range(max_retries):
                    try:
                        response = model.prompt(prompt)
                        raw_response_text = response.text()
                        break  # Success, exit retry loop
                    except Exception as e:
                        error_message = str(e).lower()
                        if "quota" in error_message and attempt < max_retries - 1:
                            # Quota error, wait longer
                            wait_time = 60 * (2 ** attempt)  # 1min, 2min, 4min
                            print(f"Quota limit hit. Waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                        elif "rate limit" in error_message and attempt < max_retries - 1:
                            # Rate limit error, wait
                            wait_time = 30 * (2 ** attempt)  # 30s, 1min, 2min
                            print(f"Rate limit hit. Waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                        else:
                            # Other error or last attempt, re-raise
                            raise

                if raw_response_text is None:
                    raise Exception("Failed to get response after retries")

                # Clean the response text by removing markdown code blocks
                cleaned_response = raw_response_text.strip()
                # Handle various markdown code block formats
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]
                elif cleaned_response.startswith("```"):
                    cleaned_response = cleaned_response[3:]
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()

                # Debug the response
                print(f"Cleaned response (first 100 chars): {cleaned_response[:100]}...")

                # Now parse the cleaned JSON with error handling for different formats
                try:
                    parsed_data = json.loads(cleaned_response)

                    # Handle array format (Mistral sometimes returns an array instead of an object)
                    if isinstance(parsed_data, list) and len(parsed_data) > 0:
                        parsed_data = parsed_data[0]  # Take the first item in the array

                    # Handle role/speaker field inconsistency
                    if "conversation" in parsed_data:
                        for turn in parsed_data["conversation"]:
                            if "role" in turn and "speaker" not in turn:
                                turn["speaker"] = turn.pop("role")
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    raise

                # Add metadata
                parsed_data["metadata"] = {
                    "template_id": template_id,  # Store template index/ID
                    # 'original_prompt_template': template, # Maybe too verbose? Store ID instead.
                    # Removed rendered_prompt to save space
                    "substitutions_used": chosen_substitutions,  # Use dict returned by expand_prompt
                    "language": lang_code,
                    "model": args.model,
                    "provider": args.provider
                }

                out_file.write(json.dumps(parsed_data) + "\n")
                out_file.flush()
                # Add this template_id + substitutions combination to our set
                subs_tuple = frozenset((k, v) for k, v in chosen_substitutions.items())
                existing_templates.add((template_id, subs_tuple))
                generated_count += 1
                pbar.update(1)  # Increment progress bar

            except json.JSONDecodeError as json_e:
                print(
                    f"\nError parsing JSON response for prompt (template ID: {template_id})"
                )
                print(f"Error: {json_e}")
                print(f"Raw Response (first 500 chars): {raw_response_text[:500]}...")
                # Save the failed response to a separate error file
                try:
                    with open(
                        f"error_response_{lang_code}_{template_id}_{int(time.time())}.txt", "w", encoding="utf-8"
                    ) as error_file:
                        error_file.write(f"Prompt: {prompt}\n\nResponse:\n{raw_response_text}")
                except Exception as write_error:
                    print(f"Could not write error file: {write_error}")
                # Add a small sleep to avoid overwhelming the API
                time.sleep(2)
            except Exception as e:
                print(f"\nError processing prompt (template ID: {template_id}): {e}")
                # Add a small sleep in case of API errors
                time.sleep(5)
            finally:
                # Optional: Short sleep even on success to be nice to the API
                time.sleep(SLEEP_BETWEEN_REQUESTS / 5)  # Shorter sleep than rate limit wait

        pbar.close()
        print(f"\nFinished generation for language '{lang_code}'. Generated {generated_count} new conversations.")
        if generated_count < max_to_generate:
            print(
                f"Note: Target was {max_to_generate}, generated {generated_count}. Could be due to duplicates or errors."
            )

        return generated_count

# --- Main execution ---
# Process languages
total_generated = 0
if args.lang.lower() == "all":
    # Process all available languages
    available_languages = get_available_languages()
    print(f"Found {len(available_languages)} available languages: {', '.join(available_languages)}")

    for lang in available_languages:
        print(f"\n{'='*50}\nProcessing language: {lang}\n{'='*50}")
        lang_generated = generate_conversations_for_language(lang, args.num_variations)
        total_generated += lang_generated

    print(f"\nCompleted processing all languages. Total generated: {total_generated} conversations.")
else:
    # Process a single language
    total_generated = generate_conversations_for_language(args.lang, args.num_variations)
