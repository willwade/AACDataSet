# Direct Multilingual Conversation Generation

This tool allows you to generate AAC conversations directly in multiple languages without the need for translation. It's designed to be more efficient and cost-effective than translating existing English conversations.

## Features

- Generate AAC conversations directly in target languages
- Support for both OpenAI and Google Gemini models
- Resume from checkpoints if generation is interrupted
- Batch processing to optimize API usage
- Language-specific substitution and template support

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Generate 10 conversations in French:

```bash
python direct_multilingual_generate.py --lang fr-FR --num 10
```

### Generate for All Supported Languages

```bash
python direct_multilingual_generate.py --lang all --num 5
```

### Using Google Gemini

```bash
python direct_multilingual_generate.py --lang es-ES --provider gemini --gemini_api_key YOUR_API_KEY
```

### Using OpenAI with a Specific Model

```bash
python direct_multilingual_generate.py --lang de-DE --provider openai --model gpt-4-turbo --openai_api_key YOUR_API_KEY
```

### Customizing Batch Size

```bash
python direct_multilingual_generate.py --lang it-IT --batch_size 3
```

## API Keys

You can provide API keys either:
1. As command-line arguments (`--openai_api_key` or `--gemini_api_key`)
2. As environment variables (`OPENAI_API_KEY` or `GOOGLE_API_KEY`)

## Output

Generated conversations are saved in the `output/{lang_code}/` directory:
- Individual batch results as timestamped JSON files
- Final combined results as `{lang_code}_all_conversations.json`

## Checkpoints

The script creates checkpoints after each batch in the `checkpoints/` directory, allowing you to resume generation if interrupted.

## How It Works

1. Loads language-specific templates, substitutions, and English atomic data
2. Generates prompts by expanding templates with appropriate substitutions
3. Instructs the LLM to translate English phrases and handle placeholders in the target language
4. Uses the selected LLM to generate conversations directly in the target language
5. Saves results with metadata for tracking

### Batch Preparation and Processing

To prepare batch files for OpenAI processing:

```bash
python direct_multilingual_generate.py --lang [language-code] --batch-prepare --num [count]
```

This will create batch files in the `batch_files/[language-code]/` directory that can be processed with OpenAI's batch API.

After processing the batch files with OpenAI, you'll need to transform the output to the format expected by the augmentation script:

```bash
python transform_batch_output.py path/to/batch_output.jsonl
```

Then augment the data with realistic AAC errors:

```bash
python augment_aac_data.py --input path/to/batch_output_transformed.jsonl --lang [language-code]
```

## Advantages Over Translation

- More natural conversation flow in the target language
- Culturally appropriate references and language use
- Reduced computational and cost overhead
- Simplified workflow with fewer potential failure points
- No need to translate atomic data files for each language
- LLM handles translation of English phrases and placeholders
- Better handling of cultural nuances and colloquial expressions