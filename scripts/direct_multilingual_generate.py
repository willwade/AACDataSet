import json
import random
import argparse
import asyncio
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import aiohttp
import backoff
import uuid

from google.generativeai import GenerativeModel
import google.generativeai as genai

# Constants
OUTPUT_DIR = Path("output")
CHECKPOINT_DIR = Path("checkpoints")
BATCH_DIR = Path("batch_files")
DEFAULT_MODEL_GEMINI = "gemini-1.5-pro"
DEFAULT_MODEL_OPENAI = "gpt-4o-mini"
DEFAULT_BATCH_SIZE = 5
MAX_BATCH_SIZE = 10
TOKEN_LIMIT = 30000  # Adjust based on the model you're using

# Define a dictionary to map language codes to their language names
LANGUAGE_CODES = {
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

# Setup directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR.mkdir(parents=True, exist_ok=True)


def load_json_file(file_path: Path) -> Any:
    """Load a JSON file and return its contents."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data: Any, file_path: Path, use_jsonl: bool = False) -> None:
    """Save data as JSON or JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        if use_jsonl and isinstance(data, list):
            # Write as JSONL (one JSON object per line)
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            # Write as formatted JSON
            json.dump(data, f, ensure_ascii=False, indent=2)


def get_supported_languages() -> List[str]:
    """Get list of supported language codes based on available template files."""
    langs = []
    for file in Path("templates/instructions").glob("*.json"):
        lang_code = file.stem
        if lang_code != "en":  # Only include non-English languages
            langs.append(lang_code)
    return langs


def ensure_output_dir(lang_code: str) -> Path:
    """Create and return output directory for the language."""
    lang_dir = OUTPUT_DIR / lang_code
    lang_dir.mkdir(parents=True, exist_ok=True)
    return lang_dir


def get_checkpoint_path(lang_code: str) -> Path:
    """Get the path for the checkpoint file."""
    return CHECKPOINT_DIR / f"{lang_code}_checkpoint.json"


def save_checkpoint(
    lang_code: str, completed_items: List[Dict], total_items: int
) -> None:
    """Save progress checkpoint."""
    checkpoint_data = {
        "lang_code": lang_code,
        "completed_items": completed_items,
        "total_items": total_items,
        "timestamp": datetime.now().isoformat(),
    }
    save_json_file(checkpoint_data, get_checkpoint_path(lang_code))
    print(
        f"Checkpoint saved: {len(completed_items)}/{total_items} items completed for {lang_code}"
    )


def load_checkpoint(lang_code: str) -> Tuple[List[Dict], int]:
    """Load checkpoint if it exists, return empty list otherwise."""
    checkpoint_path = get_checkpoint_path(lang_code)
    if checkpoint_path.exists():
        checkpoint_data = load_json_file(checkpoint_path)
        completed_items = checkpoint_data.get("completed_items", [])
        total_items = checkpoint_data.get("total_items", 0)
        print(
            f"Resuming from checkpoint: {len(completed_items)}/{total_items} items already completed for {lang_code}"
        )
        return completed_items, total_items
    return [], 0


@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def rate_limit() -> None:
    """Simple rate limiting with exponential backoff on exception."""
    await asyncio.sleep(1)  # Adjust as needed based on API limits


def load_resources(lang_code: str) -> Tuple[List[str], Dict, List[Dict]]:
    """Load language-specific resources."""
    # Load templates based on language
    template_file = Path(f"templates/instructions/{lang_code}.json")
    if not template_file.exists():
        template_file = Path("templates/instructions/en.json")

    templates_data = load_json_file(template_file)

    # Extract templates from the structure
    if isinstance(templates_data, dict) and "templates" in templates_data:
        # If templates are in a dictionary under the "templates" key
        templates = templates_data["templates"]
    elif isinstance(templates_data, list):
        # If templates are directly a list
        templates = templates_data
    else:
        # Fallback
        print(f"Warning: Unexpected template format for {lang_code}. Using empty list.")
        templates = []

    # Load language-specific substitutions
    subs_file = Path(f"templates/substitutions/{lang_code}.json")
    if not subs_file.exists():
        subs_file = Path("templates/substitutions/en.json")
    substitutions = load_json_file(subs_file)

    # Always use the English atomic data file for consistency
    # This avoids the need to translate all atomic data files
    role_map_file = Path("templates/atomic10x/atomic10x_als_subset.json")
    role_map = load_json_file(role_map_file)

    return templates, substitutions, role_map


def expand_template(
    template_str: str, substitutions: Dict, atomic_entry: Dict, lang_code: str
) -> str:
    """Expand a template with substitutions and atomic entry data."""
    # Create a new dictionary for substitutions
    subs = {}

    # Use language-specific names from substitutions file
    aac_user_name = random.choice(substitutions.get("aac_user", ["Alex"]))
    partner_name = random.choice(substitutions.get("partner", ["Taylor"]))

    # Process the topic from atomic_entry
    topic = atomic_entry.get("topic", "daily care")

    # Handle language-specific person placeholders
    person_placeholders = {
        "en": {"x": "PersonX", "y": "PersonY"},
        "sl-SI": {"x": "OsebaX", "y": "oseboY"},
        # Add more languages as needed
    }

    # Get the appropriate placeholders for this language
    lang_key = lang_code if lang_code in person_placeholders else "en"
    person_x = person_placeholders[lang_key]["x"]
    person_y = person_placeholders[lang_key]["y"]

    # Replace the placeholders with actual names
    topic = topic.replace(person_x, aac_user_name).replace(person_y, partner_name)

    # Replace any blank placeholders with partner name
    if "___" in topic:
        topic = topic.replace("___", partner_name)
    elif "___ " in topic:
        topic = topic.replace("___ ", f"{partner_name} ")
    elif " ___" in topic:
        topic = topic.replace(" ___", f" {partner_name}")

    # Get the relation and which fields from atomic_entry
    relation = atomic_entry.get("relation", "")
    which = atomic_entry.get("which", "")

    # Replace person placeholders in the which field too
    which = which.replace(person_x, aac_user_name).replace(person_y, partner_name)

    # Replace any blank placeholders in the which field too
    if "___" in which:
        which = which.replace("___", partner_name)
    elif "___ " in which:
        which = which.replace("___ ", f"{partner_name} ")
    elif " ___" in which:
        which = which.replace(" ___", f" {partner_name}")

    # Set key fields with atomic entry values and processed data
    subs["topic"] = topic
    subs["relationship"] = atomic_entry.get("relationship", "caregiver")
    subs["setting"] = atomic_entry.get("setting", "home")
    subs["aac_user"] = aac_user_name
    subs["partner"] = partner_name
    subs["which"] = which
    subs["relation"] = relation

    # Enhance with rich atomic relationship context
    # Map relation codes to human-readable descriptions
    relation_descriptions = {
        "xIntent": f"what {aac_user_name} intends by this",
        "xNeed": f"what {aac_user_name} needs for this",
        "xEffect": f"the effect this has on {aac_user_name}",
        "xReact": f"how {aac_user_name} reacts to this",
        "oEffect": f"the effect this has on {partner_name}",
        "oReact": f"how {partner_name} reacts to this",
        "xWant": f"what {aac_user_name} wants in this situation",
        "oWant": f"what {partner_name} wants in this situation",
        "xAttr": f"attributes of {aac_user_name} in this context",
        "oAttr": f"attributes of {partner_name} in this context",
    }

    # Add rich context based on the relation
    relation_description = relation_descriptions.get(relation, "")
    if relation_description:
        subs["relation_context"] = f"{relation_description}: {which}"
    else:
        subs["relation_context"] = which

    # Add context about the AAC user's role
    subs["aac_user_role"] = atomic_entry.get("aac_user_role", "person with ALS")

    # Create a rich context description that combines all the atomic knowledge
    context_elements = []
    if topic:
        context_elements.append(f"The conversation is about '{topic}'")
    if which:
        context_elements.append(f"where {which}")
    if relation_description:
        context_elements.append(f"{relation_description}")

    subs["atomic_context"] = ". ".join(context_elements)

    # Randomly select other substitution values
    for key in ["time_of_day", "tone", "aac_system", "aac_mlu_length", "writing_style"]:
        if key in substitutions:
            subs[key] = random.choice(substitutions[key])

    # Use string formatting to render the template
    try:
        from jinja2 import Template

        template = Template(template_str)
        return template.render(**subs)
    except Exception as e:
        # Fallback simple replacement if Jinja2 fails
        result = template_str
        for key, value in subs.items():
            result = result.replace("{{ " + key + " }}", str(value))
        return result


async def generate_conversation_with_gemini(
    prompt: str,
    lang_code: str,
    model_name: str = DEFAULT_MODEL_GEMINI,
    system_prompt: str = None,
) -> Dict:
    """Generate a conversation using Google's Gemini model."""
    try:
        # Configure the model
        model = GenerativeModel(model_name)

        # Generate content
        if system_prompt is None:
            system_prompt = get_system_prompt(lang_code)

        response = await asyncio.to_thread(
            model.generate_content,
            [
                {"role": "system", "parts": [system_prompt]},
                {"role": "user", "parts": [prompt]},
            ],
        )

        # Process the response to extract JSON
        try:
            # Extract the response text
            response_text = response.text

            # Check if the response contains JSON
            import re

            json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text

            # Try to parse the JSON
            conversation_data = json.loads(json_str)
            return conversation_data

        except json.JSONDecodeError:
            print(f"Failed to parse JSON from response: {response_text[:100]}...")
            # Return a simple error object
            return {
                "error": "Failed to parse JSON from response",
                "raw_response": response_text[
                    :500
                ],  # Include partial response for debugging
            }

    except Exception as e:
        print(f"Error generating with Gemini: {str(e)}")
        return {"error": str(e)}


async def generate_conversation_with_openai(
    prompt: str,
    lang_code: str,
    model_name: str = DEFAULT_MODEL_OPENAI,
    api_key: str = None,
    system_prompt: str = None,
) -> Dict:
    """Generate a conversation using OpenAI API."""
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required but not provided")

    if system_prompt is None:
        system_prompt = get_system_prompt(lang_code)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 1500,
                "response_format": {"type": "json_object"},
            },
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"OpenAI API error: {response.status} - {error_text}")
                return {"error": f"API error: {response.status}"}

            result = await response.json()
            try:
                # Extract the content from the completion
                content = result["choices"][0]["message"]["content"]
                # Parse the JSON content
                conversation_data = json.loads(content)
                return conversation_data
            except (KeyError, json.JSONDecodeError) as e:
                print(f"Failed to parse OpenAI response: {str(e)}")
                return {"error": str(e), "raw_response": str(result)}


def get_system_prompt(lang_code: str) -> str:
    """Get the appropriate system prompt for the given language."""
    # Get language name for clearer instructions
    language_names = {
        code: name.split(" (")[0] for code, name in LANGUAGE_CODES.items()
    }

    language_name = language_names.get(lang_code, lang_code)

    # Check for language-specific system prompt
    prompt_template_file = Path(f"templates/prompt_templates/{lang_code}.json")
    if not prompt_template_file.exists():
        prompt_template_file = Path("templates/prompt_templates/en.json")

    try:
        templates = load_json_file(prompt_template_file)
        base_prompt = random.choice(templates)

        # Add atomic context information to enrich the prompt
        atomic_context_instruction = (
            "\n\nAdditional context about the conversation:"
            "\nThe conversation should incorporate the following atomic knowledge:"
            "\n- The topic is about '{{ topic }}'."
            "\n- The relation type is '{{ relation }}', which means {{ relation_context }}."
            "\n- The specific context is: {{ atomic_context }}"
        )

        # Add explicit language instruction for all languages to use colloquial speech
        # and to translate English phrases in the topic
        language_instruction = (
            f"\n\nIMPORTANT: You MUST generate ALL content in {language_name} language ONLY, "
            f"using COLLOQUIAL, EVERYDAY spoken language - NOT formal or literary language. "
            f"Use natural spoken language as people would use in real conversations. "
            f"The output conversation must be entirely in colloquial {language_name}, "
            f"including all utterances, scene descriptions, and any other text content. "
            f"Do not use any other language or formal expressions that wouldn't be used in everyday speech."
            f"\n\nIf the topic contains English phrases like 'calls a repairman', "
            f"'accuses of cheating', etc., TRANSLATE these phrases into natural, "
            f"colloquial {language_name} in your response. Do not keep any English phrases."
            f"\n\nIf you see placeholders like '___' in phrases (e.g., 'treats ___ in patients'), "
            f"these are meant to be filled with names or appropriate content. "
            f"Replace these placeholders with appropriate words in {language_name} "
            f"that make sense in the context of the conversation."
            f"\n\nFormat your response as a JSON object with the following structure:"
            f"\n{{"
            f'\n  "conversation": ['
            f'\n    {{"speaker": "Name", "utterance": "Text", "is_aac_user": boolean}},'
            f"\n    ..."
            f"\n  ],"
            f'\n  "scene": "Description of the scene"'
            f"\n}}"
        )

        return base_prompt + atomic_context_instruction + language_instruction
    except Exception:
        # Fallback generic system prompt
        return (
            f"You are a helpful assistant generating AAC-like conversations for adults with ALS/MND. "
            f"Follow this JSON schema exactly:\n"
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
            "Keep AAC utterances realistic for users with physical limitations.\n\n"
            "Additional context about the conversation:\n"
            "The conversation should incorporate atomic knowledge about the topic, "
            "including the relationship between the participants and their reactions, "
            "intentions, and needs in this specific context.\n\n"
            f"IMPORTANT: You MUST generate ALL content in {language_name} language ONLY, "
            f"using COLLOQUIAL, EVERYDAY spoken language - NOT formal or literary language. "
            f"Use natural spoken language as people would use in real conversations. "
            f"The output conversation must be entirely in colloquial {language_name}, "
            f"including all utterances, scene descriptions, and any other text content. "
            f"Do not use any other language or formal expressions that wouldn't be used in everyday speech."
            f"\n\nIf the topic contains English phrases like 'calls a repairman', "
            f"'accuses of cheating', etc., TRANSLATE these phrases into natural, "
            f"colloquial {language_name} in your response. Do not keep any English phrases."
            f"\n\nIf you see placeholders like '___' in phrases (e.g., 'treats ___ in patients'), "
            f"these are meant to be filled with names or appropriate content. "
            f"Replace these placeholders with appropriate words in {language_name} "
            f"that make sense in the context of the conversation."
            f"\n\nMake sure to format your response as a valid JSON object as specified above."
        )


async def process_batch(
    batch: List[Tuple[str, int]],
    lang_code: str,
    provider: str,
    model_name: str,
    openai_api_key: str = None,
) -> List[Dict]:
    """Process a batch of prompts and return generated conversations."""
    results = []

    # Process each prompt in the batch
    for item in batch:
        # Handle both old and new format tuples
        if len(item) == 3:
            user_prompt, system_prompt, template_id = item
        else:
            user_prompt, template_id = item
            # Get system prompt (this is the old way, should be phased out)
            system_prompt_template = get_system_prompt(lang_code)
            system_prompt = system_prompt_template  # No expansion

        await rate_limit()  # Apply rate limiting

        try:
            # Choose provider
            if provider.lower() == "gemini":
                result = await generate_conversation_with_gemini(
                    user_prompt, lang_code, model_name, system_prompt
                )
            else:  # Default to OpenAI
                result = await generate_conversation_with_openai(
                    user_prompt, lang_code, model_name, openai_api_key, system_prompt
                )

            # Add template_id if not present in the result
            if "template_id" not in result:
                result["template_id"] = template_id

            # Add metadata
            result["metadata"] = {
                "lang_code": lang_code,
                "generated_at": datetime.now().isoformat(),
                "provider": provider,
                "model": model_name,
            }

            results.append(result)
            print(f"Generated conversation for template_id {template_id}")

        except Exception as e:
            print(f"Error processing prompt for template_id {template_id}: {str(e)}")
            results.append(
                {
                    "template_id": template_id,
                    "error": str(e),
                    "metadata": {
                        "lang_code": lang_code,
                        "generated_at": datetime.now().isoformat(),
                        "provider": provider,
                        "model": model_name,
                    },
                }
            )

    return results


async def generate_conversations(
    lang_code: str,
    num_conversations: int,
    provider: str = "openai",
    model_name: str = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    openai_api_key: str = None,
) -> None:
    """Generate multiple conversations in the specified language."""
    # Set default model name based on provider
    if not model_name:
        model_name = (
            DEFAULT_MODEL_OPENAI
            if provider.lower() == "openai"
            else DEFAULT_MODEL_GEMINI
        )

    # Load resources for the language
    templates, substitutions, role_map = load_resources(lang_code)

    # Check if we should resume from checkpoint
    completed_items, total_items = load_checkpoint(lang_code)

    # Update total_items if the requested number is greater
    if num_conversations > total_items:
        total_items = num_conversations
    # If starting new, set total_items
    elif total_items == 0:
        total_items = num_conversations

    # Calculate how many more items we need to generate
    remaining = total_items - len(completed_items)

    if remaining <= 0:
        print(
            f"Already completed {len(completed_items)} conversations for {lang_code}. No more to generate."
        )
        return

    print(
        f"Generating {remaining} conversations for language: {lang_code} (Total target: {total_items})"
    )

    # Prepare prompts for all conversations
    all_prompts = []
    for _ in range(remaining):
        template_id = random.randint(0, len(templates) - 1)
        template = templates[template_id]
        atomic_entry = random.choice(role_map)

        # Expand the user prompt template
        user_prompt = expand_template(template, substitutions, atomic_entry, lang_code)

        # Get and expand the system prompt template
        system_prompt_template = get_system_prompt(lang_code)
        system_prompt = expand_template(
            system_prompt_template, substitutions, atomic_entry, lang_code
        )

        # Store both prompts
        all_prompts.append((user_prompt, system_prompt, template_id))

    # Process in batches
    for i in range(0, len(all_prompts), batch_size):
        batch = all_prompts[i : i + batch_size]
        print(
            f"Processing batch {i//batch_size + 1}/{(len(all_prompts) + batch_size - 1)//batch_size}"
        )

        batch_results = await process_batch(
            batch, lang_code, provider, model_name, openai_api_key
        )
        completed_items.extend(batch_results)

        # Save checkpoint after each batch
        save_checkpoint(lang_code, completed_items, total_items)

        # Save the current results to output directory
        output_file = (
            ensure_output_dir(lang_code)
            / f"{lang_code}_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        save_json_file(batch_results, output_file, use_jsonl=True)
        print(f"Saved batch results to {output_file}")

    # Save final output
    final_output_file = (
        ensure_output_dir(lang_code) / f"{lang_code}_all_conversations.jsonl"
    )
    save_json_file(completed_items, final_output_file, use_jsonl=True)
    print(f"All {len(completed_items)} conversations saved to {final_output_file}")


def prepare_batch_files(
    lang_code: str,
    num_conversations: int,
    provider: str = "openai",
    model_name: str = None,
) -> None:
    """Prepare batch files with prompts for later API processing."""
    # Set default model name based on provider
    if not model_name:
        model_name = (
            DEFAULT_MODEL_OPENAI
            if provider.lower() == "openai"
            else DEFAULT_MODEL_GEMINI
        )

    # Load resources for the language
    templates, substitutions, role_map = load_resources(lang_code)

    # Prepare batch directory for this language
    batch_lang_dir = BATCH_DIR / lang_code
    batch_lang_dir.mkdir(parents=True, exist_ok=True)

    # Generate a unique batch ID
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

    # Prepare batch requests
    batch_requests = []
    for i in range(num_conversations):
        template_id = random.randint(0, len(templates) - 1)
        template = templates[template_id]
        atomic_entry = random.choice(role_map)

        # Expand the user prompt template with substitutions and atomic data
        user_prompt = expand_template(template, substitutions, atomic_entry, lang_code)

        # Get system prompt template
        system_prompt_template = get_system_prompt(lang_code)
        # Expand the system prompt template with the same substitutions and atomic data
        system_prompt = expand_template(
            system_prompt_template, substitutions, atomic_entry, lang_code
        )

        # Format the request based on provider
        if provider.lower() == "openai":
            # Store metadata for our reference
            metadata_entry = {
                "template_id": template_id,
                "lang_code": lang_code,
                "created_at": datetime.now().isoformat(),
            }

            # OpenAI Batch API compatible format
            request = {
                "custom_id": f"{batch_id}_{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1500,
                    "response_format": {"type": "json_object"},
                },
                "metadata": metadata_entry,  # Add metadata as a non-standard field
            }
        else:  # Gemini
            # Standard format for our own processing
            request = {
                "id": f"{batch_id}_{i}",
                "template_id": template_id,
                "provider": provider,
                "model": model_name,
                "lang_code": lang_code,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "created_at": datetime.now().isoformat(),
                "parameters": {"temperature": 0.7, "max_output_tokens": 1500},
                "metadata": metadata_entry,  # For consistency with OpenAI format
            }

        batch_requests.append(request)

    # Save appropriate batch files based on provider
    if provider.lower() == "openai":
        # Main OpenAI Batch API compatible file - remove metadata field before saving
        openai_batch_filename = f"openai_batch_{lang_code}_{batch_id}.jsonl"
        openai_batch_path = batch_lang_dir / openai_batch_filename

        with open(openai_batch_path, "w", encoding="utf-8") as f:
            for req in batch_requests:
                # Create a copy without the metadata field for the official batch file
                api_req = req.copy()
                if "metadata" in api_req:
                    del api_req["metadata"]
                f.write(json.dumps(api_req) + "\n")

        print(
            f"Created OpenAI batch file with {len(batch_requests)} requests at: {openai_batch_path}"
        )

        # Also save a metadata version with all information for our reference
        meta_batch_filename = f"batch_{lang_code}_{provider}_{batch_id}_metadata.jsonl"
        meta_batch_path = batch_lang_dir / meta_batch_filename

        with open(meta_batch_path, "w", encoding="utf-8") as f:
            for req in batch_requests:
                f.write(json.dumps(req) + "\n")

        print(f"Created metadata reference file at: {meta_batch_path}")
        return openai_batch_path
    else:
        # Regular batch file for other providers
        batch_filename = f"batch_{lang_code}_{provider}_{batch_id}.jsonl"
        batch_file_path = batch_lang_dir / batch_filename

        with open(batch_file_path, "w", encoding="utf-8") as f:
            for req in batch_requests:
                f.write(json.dumps(req) + "\n")

        print(
            f"Created batch file with {len(batch_requests)} requests at: {batch_file_path}"
        )
        return batch_file_path


async def main():
    """Main function to parse arguments and run generation."""
    parser = argparse.ArgumentParser(
        description="Generate AAC conversations directly in target languages"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="all",
        help="Language code (or 'all' for all supported languages)",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="Number of conversations to generate per language",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "gemini"],
        help="Model provider",
    )
    parser.add_argument(
        "--model", type=str, help="Model name (defaults based on provider)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for processing",
    )
    parser.add_argument("--openai_api_key", type=str, help="OpenAI API key")
    parser.add_argument("--gemini_api_key", type=str, help="Google API key for Gemini")
    parser.add_argument(
        "--batch-prepare",
        action="store_true",
        help="Only prepare batch files without making API calls",
    )

    args = parser.parse_args()

    # Set API keys from arguments or environment variables if not in batch prepare mode
    if not args.batch_prepare:
        if args.provider == "openai" and args.openai_api_key:
            os.environ["OPENAI_API_KEY"] = args.openai_api_key
        elif args.provider == "gemini" and args.gemini_api_key:
            os.environ["GOOGLE_API_KEY"] = args.gemini_api_key
            genai.configure(api_key=args.gemini_api_key)

    # Determine which languages to process
    languages = [args.lang]
    if args.lang == "all":
        languages = get_supported_languages()

    # Process each language
    for lang in languages:
        print(f"Starting generation for language: {lang}")

        if args.batch_prepare:
            # Only prepare batch files without making API calls
            prepare_batch_files(
                lang_code=lang,
                num_conversations=args.num,
                provider=args.provider,
                model_name=args.model,
            )
        else:
            # Generate conversations using API calls
            await generate_conversations(
                lang_code=lang,
                num_conversations=args.num,
                provider=args.provider,
                model_name=args.model,
                batch_size=args.batch_size,
                openai_api_key=(
                    args.openai_api_key if args.provider == "openai" else None
                ),
            )

    if args.batch_prepare:
        print(
            "All batch preparation tasks completed! Files saved in the 'batch_files' directory."
        )
    else:
        print("All language generation tasks completed!")


if __name__ == "__main__":
    asyncio.run(main())
