# Efficient Translation Workflow for Atomic Data

This document explains a more efficient approach for translating the atomic data used in the AAC dataset generation process.

## Overview

The standard translation process in `atomic10x_batch_wrapper.py` translates each entry in the atomic dataset individually, which can be inefficient since many entries share the same "topic" or "which" values. This new approach:

1. Extracts unique strings from the atomic data
2. Creates a batch file for translating only these unique strings
3. Processes the results to reconstruct the full translated dataset

## Benefits

- **Significantly Reduced Translation Volume**: Instead of translating 48,710 entries (97,420 fields), you only translate the unique strings (typically a few thousand)
- **Cost Savings**: Fewer API calls means lower costs
- **Faster Processing**: The entire translation process completes much more quickly
- **Same Quality**: The final output is identical to the standard approach

## Workflow

### Step 1: Extract Unique Strings and Create Batch File

You can process a single language:

```bash
python scripts/batch_translate_unique_atomic.py --lang fr-FR
```

Or process multiple languages at once:

```bash
python scripts/batch_translate_unique_atomic.py --languages fr-FR es-ES de-DE
```

Or process all supported languages:

```bash
python scripts/batch_translate_unique_atomic.py --all
```

This script:
- Loads the atomic data from `templates/atomic10x/atomic10x_als_subset.json`
- Extracts all unique "topic" and "which" values
- Creates a mapping file to track the original strings
- Generates a batch file for OpenAI's batch processing system

Output files (for each language):
- `batch_output/batch_translate_unique_fr-FR_TIMESTAMP.jsonl`: Batch file for OpenAI
- `batch_output/atomic_mapping_fr-FR_TIMESTAMP.json`: Mapping file for reconstruction

### Step 2: Process with OpenAI's Batch System

1. Upload the batch files to OpenAI's batch processing system
2. Wait for processing to complete
3. Download the results

### Step 3: Process the Results

For each language, process the results:

```bash
python scripts/process_batch_unique_translations.py --input downloaded_results_fr-FR.json --mapping batch_output/atomic_mapping_fr-FR_TIMESTAMP.json
```

This script:
- Loads the batch results and mapping file
- Extracts the translations for each unique string
- Creates a lookup dictionary to map original strings to translations
- Reconstructs the full translated dataset
- Saves the result to `templates/atomic10x/atomic10x_als_subset_fr-FR.json`

## Example Usage

### Single Language

```bash
# Step 1: Create batch file for French
python scripts/batch_translate_unique_atomic.py --lang fr-FR

# Step 2: Upload batch_output/batch_translate_unique_fr-FR_20250503_120000.jsonl to OpenAI's batch system

# Step 3: Process the results
python scripts/process_batch_unique_translations.py --input downloaded_results_fr-FR.json --mapping batch_output/atomic_mapping_fr-FR_20250503_120000.json
```

### Multiple Languages

```bash
# Step 1: Create batch files for multiple languages
python scripts/batch_translate_unique_atomic.py --languages fr-FR es-ES de-DE

# Step 2: Upload all batch files to OpenAI's batch system

# Step 3: Process the results for each language
python scripts/process_batch_unique_translations.py --input downloaded_results_fr-FR.json --mapping batch_output/atomic_mapping_fr-FR_20250503_120000.json
python scripts/process_batch_unique_translations.py --input downloaded_results_es-ES.json --mapping batch_output/atomic_mapping_es-ES_20250503_120000.json
python scripts/process_batch_unique_translations.py --input downloaded_results_de-DE.json --mapping batch_output/atomic_mapping_de-DE_20250503_120000.json
```

### All Languages

```bash
# Step 1: Create batch files for all supported languages
python scripts/batch_translate_unique_atomic.py --all

# Step 2: Upload all batch files to OpenAI's batch system

# Step 3: Process the results for each language
# (Repeat for each language)
```

## Integration with Existing Workflow

After creating the translated atomic data files, you can continue with the standard workflow:

```bash
# Generate batch requests using the translated atomic data
python scripts/atomic10x_batch_wrapper.py --languages fr-FR es-ES de-DE --prepare_only --skip translate
```

The `--skip translate` flag tells the script to skip the translation step since you've already created the translated files.

## Performance Comparison

For a typical atomic dataset with 48,710 entries:

| Approach | Items to Translate | API Calls | Approximate Time |
|----------|-------------------|-----------|-----------------|
| Standard | 97,420 | ~3,900 | 8-10 hours |
| Efficient | ~5,000 | ~50 | 30-60 minutes |

The exact numbers will vary depending on the dataset, but the efficient approach is typically 10-20x faster and requires far fewer API calls.
