# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "jinja2",
#     "llm",
#     "llm-gemini",
#     "tqdm",
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
    default=DEFAULT_LANGUAGE,
    help=f"Language code for prompts and substitutions (e.g., 'en', 'es'). Default: {DEFAULT_LANGUAGE}",
)
parser.add_argument(
    "--num_variations",
    type=int,
    default=3,
    help="Number of variations to generate per template.",
)
args = parser.parse_args()

# --- Setup Paths based on Language ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
output_path = OUTPUT_DIR / f"aac_conversations_{args.lang}.jsonl"
prompt_templates_path = (
    PROMPT_TEMPLATES_DIR / f"{args.lang}.json"
)  # Assumes templates are named like 'en.json', 'es.json'
substitutions_path = SUBSTITUTIONS_DIR / f"{args.lang}.json"

# --- Load Prompts ---
if prompt_templates_path.exists():
    with open(prompt_templates_path, "r", encoding="utf-8") as f:
        templates = json.load(f)
    print(f"Loaded {len(templates)} prompt templates from {prompt_templates_path}")
else:
    raise FileNotFoundError(
        f"Prompt templates file not found for language '{args.lang}' at: {prompt_templates_path}"
    )

# --- Load Substitutions --- REMOVED THE HARDCODED DICTIONARY
if substitutions_path.exists():
    with open(substitutions_path, "r", encoding="utf-8") as f:
        substitutions = json.load(f)
    print(f"Loaded substitutions for language '{args.lang}' from {substitutions_path}")
else:
    raise FileNotFoundError(
        f"Substitutions file not found for language '{args.lang}' at: {substitutions_path}"
    )

# --- Setup Jinja Environment --- (More robust way to handle templates)
# Consider putting templates in separate .jinja2 files if they get complex
# env = Environment(loader=FileSystemLoader(PROMPT_TEMPLATES_DIR))


# --- Generate concrete prompts --- (Updated to handle potentially missing keys)
def expand_prompt(
    template, template_id="unknown"
):  # Add template_id for better context
    sub_values = {}
    missing_keys = []
    # Randomly select values for keys present in BOTH template and substitutions
    for key in substitutions.keys():
        # Basic check if the template likely uses the key
        if f"{{{{ {key} }}}}" in template:
            sub_values[key] = random.choice(substitutions[key])
        # Keep track even if not found, maybe log later if needed

    # Store the chosen values for metadata (ensure keys exist before accessing)
    chosen_time = sub_values.get("time_of_day")
    chosen_setting = sub_values.get("setting")
    chosen_tone = sub_values.get("tone")
    chosen_relationship = sub_values.get("relationship")
    chosen_aac_system = sub_values.get("aac_system")
    chosen_topic = sub_values.get("topic")

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


# --- Load existing results --- (Adjust path)
existing_prompts = set()
if output_path.exists():
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                # Adjust based on your FINAL desired JSONL structure
                # This assumes 'rendered_prompt' will be in the metadata
                data = json.loads(line)
                if "metadata" in data and "rendered_prompt" in data["metadata"]:
                    existing_prompts.add(data["metadata"]["rendered_prompt"])
            except Exception:
                continue
print(f"Loaded {len(existing_prompts)} existing prompts to avoid duplicates.")

# --- Get Gemini model ---
try:
    model = llm.get_model("gemini-2.0-flash")  # Use model alias if configured in llm
    # Or use the full name if needed: model = llm.get_model("gemini-1.5-flash")
except llm.UnknownModelError:
    print(
        "Error: Gemini model not found. Ensure llm-gemini plugin is installed and configured."
    )
    exit()  # Exit if model not found
except Exception as e:
    print(f"Error getting LLM model: {e}")
    exit()


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


# --- Main generation loop ---
generated_count = 0
max_to_generate = min(MAX_REQUESTS_PER_DAY, len(templates) * args.num_variations)
print(f"Attempting to generate up to {max_to_generate} new conversations...")

# Use enumerate to get an index for templates if needed (e.g., for template_id)
template_list = [(idx, tpl) for idx, tpl in enumerate(templates)]


with open(output_path, "a", encoding="utf-8") as out_file:
    # Loop goal is to generate 'max_to_generate' *new* items
    # We might need more attempts if many prompts are duplicates or fail
    pbar = tqdm(total=max_to_generate, desc="Generating Conversations")
    attempts = 0
    max_attempts = max_to_generate * 3  # Try up to 3x times to find non-duplicates

    while generated_count < max_to_generate and attempts < max_attempts:
        attempts += 1
        template_id, template = random.choice(
            template_list
        )  # Get template and its index/ID
        prompt, chosen_substitutions = expand_prompt(template, template_id)

        if prompt is None:  # Skip if rendering failed
            continue

        if prompt in existing_prompts:
            continue  # Skip duplicate

        rate_limit()
        try:
            # *** IMPORTANT: You need to actually CALL the LLM here! ***
            # The previous version had the LLM call commented out inside the try block.
            response = model.prompt(prompt)
            raw_response_text = response.text()
            # **********************************************************

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

            # Now parse the cleaned JSON
            parsed_data = json.loads(cleaned_response)

            # Add metadata
            parsed_data["metadata"] = {
                "template_id": template_id,  # Store template index/ID
                # 'original_prompt_template': template, # Maybe too verbose? Store ID instead.
                "rendered_prompt": prompt,
                "substitutions_used": chosen_substitutions,  # Use dict returned by expand_prompt
                "language": args.lang,
            }

            out_file.write(json.dumps(parsed_data) + "\n")
            out_file.flush()
            existing_prompts.add(prompt)  # Add newly generated prompt
            generated_count += 1
            pbar.update(1)  # Increment progress bar

        except json.JSONDecodeError as json_e:
            print(
                f"\nError parsing JSON response for prompt (template ID: {template_id})"
            )
            print(f"Error: {json_e}")
            print(f"Raw Response (first 500 chars): {raw_response_text[:500]}...")
            # Save the failed response to a separate error file
            with open(
                f"error_response_{template_id}_{int(time.time())}.txt", "w"
            ) as error_file:
                error_file.write(f"Prompt: {prompt}\n\nResponse:\n{raw_response_text}")
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
    print(f"\nFinished generation. Generated {generated_count} new conversations.")
    if generated_count < max_to_generate:
        print(
            f"Note: Target was {max_to_generate}, generated {generated_count}. Could be due to duplicates or errors."
        )
