# JSON Translation Workflow for Atomic Data

This document explains the simplest approach for translating the atomic data used in the AAC dataset generation process.

## Overview

Instead of translating individual strings or entries, this approach translates the entire JSON file at once while preserving its structure. This is the most straightforward and efficient method for translating the atomic data.

## Benefits

- **Simplest Approach**: Translate the entire file in one go
- **Structure Preservation**: Maintains the JSON structure during translation
- **Efficient**: No need to extract, translate, and reconstruct individual strings
- **Fast**: Translation completes in minutes rather than hours
- **No API Key Required**: Uses the free tier of Google Translate via deep_translator

## Prerequisites

This approach uses the `deep_translator` library, which provides a free interface to Google Translate:

1. Install the required packages:

```bash
pip install deep-translator tqdm
```

## Workflow

### Step 1: Translate the JSON File

You can process a single language:

```bash
python scripts/translate_json_file.py --lang fr-FR
```

Or process multiple languages at once:

```bash
python scripts/translate_json_file.py --languages fr-FR es-ES de-DE
```

Or process all supported languages:

```bash
python scripts/translate_json_file.py --all
```

This script:
- Loads the atomic data from `templates/atomic10x/atomic10x_als_subset.json`
- Extracts all strings that need to be translated
- Translates them in batches using deep_translator
- Preserves the JSON structure
- Saves the translated files as `templates/atomic10x/atomic10x_als_subset_fr-FR.json`, etc.

### Step 2: Continue with the Standard Workflow

After creating the translated atomic data files, you can continue with the standard workflow:

```bash
# Generate batch requests using the translated atomic data
python scripts/atomic10x_batch_wrapper.py --languages fr-FR es-ES de-DE --prepare_only --skip translate
```

The `--skip translate` flag tells the script to skip the translation step since you've already created the translated files.

## Customization Options

The script provides several options for customization:

- `--fields`: Specify which fields to translate (default: "topic" and "which")
- `--skip-fields`: Specify which fields to skip (default: "aac_user", "partner", "aac_user_role", "relation")
- `--input`: Specify a different input file (default: "templates/atomic10x/atomic10x_als_subset.json")
- `--batch-size`: Number of strings to translate in each batch (default: 25)
- `--min-delay`: Minimum delay between batches in seconds (default: 1)
- `--max-delay`: Maximum delay between batches in seconds (default: 3)

Example:

```bash
python scripts/translate_json_file.py --lang fr-FR --fields topic which --skip-fields aac_user partner relation --batch-size 50 --min-delay 2 --max-delay 5
```

## Performance Comparison

| Approach | Complexity | API Calls | Approximate Time | Cost |
|----------|------------|-----------|-----------------|------|
| Standard (atomic10x_batch_wrapper.py) | High | ~3,900 | 8-10 hours | $$$$ |
| Efficient (batch_translate_unique_atomic.py) | Medium | ~50 | 30-60 minutes | $$ |
| JSON Translation (translate_json_file.py) | Low | Batched | 10-30 minutes | Free |

The JSON Translation approach is by far the simplest and most efficient method for translating the atomic data.

## Troubleshooting

### Rate Limiting

If you encounter rate limiting issues, you can adjust the batch size and delays:

```bash
# Smaller batches with longer delays
python scripts/translate_json_file.py --lang fr-FR --batch-size 10 --min-delay 3 --max-delay 6
```

### Translation Quality

Google Translate is generally high quality, but you may want to review the translations for specific terms or phrases. You can always manually edit the translated files if needed.

### Special Placeholders

The script is designed to preserve special placeholders like "PersonX", "PersonY", and "___". If you notice any issues with these placeholders, please check the translated files and make manual corrections if needed.
