# Atomic10x Workflow for AAC Dataset

This document outlines the workflow for using the Atomic10x knowledge graph data to generate diverse, realistic conversations for adults with ALS in multiple languages.

## Overview

The Atomic10x approach uses the ATOMIC2020 knowledge graph to create diverse conversation scenarios. The workflow consists of five main steps:

1. **Prepare Atomic Subset**: Filter the ATOMIC2020 dataset to create a subset relevant to AAC users with ALS
2. **Translate Atomic Data**: Translate the atomic data components to target languages
3. **Generate Batch Requests**: Create batch requests for OpenAI using the translated atomic data
4. **Process Batch Responses**: Send the batch requests to OpenAI and process the responses
5. **Augment AAC Data**: Add noisy variations to the AAC utterances (using the existing augmentation script)

## Quick Start: Using the Wrapper Script

The easiest way to run the entire workflow is to use the wrapper script:

```bash
# Process a single language with default settings (100 requests)
python scripts/atomic10x_batch_wrapper.py --languages en-GB

# Process multiple languages
python scripts/atomic10x_batch_wrapper.py --languages en-GB fr-FR es-ES

# Process all supported languages
python scripts/atomic10x_batch_wrapper.py --all

# Generate 1000 conversations per language (10 batches of 100)
python scripts/atomic10x_batch_wrapper.py --languages en-GB fr-FR --requests_per_batch 100 --num_batches 10

# Skip certain steps (useful for resuming after errors)
python scripts/atomic10x_batch_wrapper.py --languages en-GB --skip prepare translate

# Limit the number of atomic entries (useful for testing)
python scripts/atomic10x_batch_wrapper.py --languages en-GB --atomic_limit 100

# Use a different model
python scripts/atomic10x_batch_wrapper.py --languages en-GB --model gpt-4o
```

The wrapper script:
- Processes multiple languages in sequence
- Creates multiple batches per language
- Handles errors and provides a summary
- Skips steps that have already been completed
- Saves a summary of the run with timing information

## Detailed Workflow

If you prefer to run the steps individually, here's how to do it:

### Step 1: Prepare Atomic Subset

```bash
python scripts/prepare_atomic_subset.py

# Limit the number of entries (for testing)
python scripts/prepare_atomic_subset.py --limit 100

# Specify a custom output file
python scripts/prepare_atomic_subset.py --output templates/custom_atomic_subset.json
```

This script filters the ATOMIC2020 dataset to create a subset relevant to AAC users with ALS. The output is saved to `templates/atomic10x/atomic10x_als_subset.json`.

**Key files:**
- Input: `atomic2020_data-feb2021/train.tsv`, `atomic2020_data-feb2021/dev.tsv`, `atomic2020_data-feb2021/test.tsv`
- Output: `templates/atomic10x/atomic10x_als_subset.json`

### Step 2: Translate Atomic Data

```bash
# Translate to a specific language
python scripts/translate_atomic_data.py --lang fr-FR

# Translate to all supported languages
python scripts/translate_atomic_data.py --all

# Translate a limited number of entries (for testing)
python scripts/translate_atomic_data.py --lang fr-FR --limit 100
```

This script translates the atomic data components (topic and which fields) to target languages. The translations are done in batches to minimize API calls.

**Key files:**
- Input: `templates/atomic10x/atomic10x_als_subset.json`
- Output: `templates/atomic10x/atomic10x_als_subset_{lang}.json` (e.g., `templates/atomic10x/atomic10x_als_subset_fr-FR.json`)

### Step 3: Generate Batch Requests

```bash
# Generate batch requests for a specific language
python scripts/batch_openai_prepare.py --lang fr-FR --num_requests 100 --model gpt-4-turbo

# Generate batch requests for English
python scripts/batch_openai_prepare.py --lang en --num_requests 100 --model gpt-4-turbo
```

This script generates batch requests for OpenAI using the translated atomic data. The requests are saved as JSONL files in the `batch_output` directory.

**Key files:**
- Input:
  - `templates/atomic10x/als_template.json`
  - `templates/atomic10x/als_substitutions.json`
  - `templates/atomic10x/atomic10x_als_subset_{lang}.json`
- Output: `batch_output/batch_requests_{lang}_{timestamp}.jsonl`

### Step 4: Process Batch Responses

```bash
# Process a batch file
python scripts/process_batch.py batch_output/batch_requests_{lang}_{timestamp}.jsonl
```

This script sends the batch requests to OpenAI and processes the responses. The responses are saved as JSON files in the `batch_output` directory.

**Key files:**
- Input: `batch_output/batch_requests_{lang}_{timestamp}.jsonl`
- Output: `batch_output/batch_requests_{lang}_{timestamp}_responses.json`

### Step 5: Augment AAC Data

```bash
# Augment the AAC data
python scripts/augment_aac_data.py --input batch_output/batch_requests_{lang}_{timestamp}_responses.json

# Append to an existing file
python scripts/augment_aac_data.py --input batch_output/batch_requests_{lang}_{timestamp}_responses.json --append
```

This script adds noisy variations to the AAC utterances, simulating different typing errors and keyboard layouts.

**Key files:**
- Input: `batch_output/batch_requests_{lang}_{timestamp}_responses.json`
- Output: `data/output/augmented_aac_conversations_{lang}.jsonl`

## Utility Scripts

### Display Conversations

```bash
# Display conversations from a responses file
python scripts/display_conversations.py batch_output/batch_requests_{lang}_{timestamp}_responses.json --num 5 --random --show-prompt
```

This script displays conversations from a responses file in a readable format.

### Test Atomic Integration

```bash
# Test the atomic integration
python scripts/test_atomic_integration.py
```

This script tests the integration of atomic data into templates.

### Single Language Workflow

```bash
# Run the entire workflow for a single language
python scripts/atomic10x_workflow.py --lang fr-FR --num_requests 100

# Skip certain steps
python scripts/atomic10x_workflow.py --lang fr-FR --num_requests 100 --skip_prepare --skip_translate
```

This script runs the entire workflow for a single language.

## Complete Example Workflow

Here's a complete example workflow for generating French conversations:

```bash
# Step 1: Prepare atomic subset (if not already done)
python scripts/prepare_atomic_subset.py

# Step 2: Translate atomic data to French
python scripts/translate_atomic_data.py --lang fr-FR

# Step 3: Generate batch requests for French
python scripts/batch_openai_prepare.py --lang fr-FR --num_requests 100 --model gpt-4-turbo

# Step 4: Process batch responses
python scripts/process_batch.py batch_output/batch_requests_fr-FR_20230501_120000.jsonl

# Step 5: Augment AAC data
python scripts/augment_aac_data.py --input batch_output/batch_requests_fr-FR_20230501_120000_responses.json

# Optional: Display some conversations
python scripts/display_conversations.py batch_output/batch_requests_fr-FR_20230501_120000_responses.json --num 5 --random
```

## Generating Large Datasets

To generate a large dataset (e.g., 20,000 conversations per language), you can use the wrapper script:

```bash
# Generate 20,000 conversations for English-GB and French
python scripts/atomic10x_batch_wrapper.py --languages en-GB fr-FR --requests_per_batch 1000 --num_batches 20
```

This will:
1. Prepare the atomic subset (if not already done)
2. Translate the atomic data to French (if not already done)
3. Generate 20 batches of 1,000 requests each for both languages
4. Process each batch and augment the data
5. Save a summary of the run

## Notes

- The atomic10x approach provides more diverse and realistic conversations compared to standard templates
- The translation approach is efficient because it only translates the atomic data components once per language
- The batch processing approach allows for generating large numbers of conversations efficiently
- The augmentation step adds realistic variations to the AAC utterances, making the dataset more useful for training AAC prediction models
- The wrapper script makes it easy to generate large datasets for multiple languages
