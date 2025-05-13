# OpenAI Batch Processing Workflow

This document explains how to use the `atomic10x_batch_wrapper.py` script with OpenAI's batch processing system for more cost-effective dialogue generation.

## Overview

The workflow consists of three main steps:

1. **Prepare batch files** - Generate JSONL files with batch requests
2. **Process with OpenAI's batch system** - Upload these files to OpenAI's batch processing system
3. **Process the results** - Download the results and process them with our script

## Step 1: Prepare Batch Files

Use the `--prepare_only` flag to generate batch files without processing them:

```bash
python scripts/atomic10x_batch_wrapper.py --languages en-GB fr-FR --num_batches 10 --requests_per_batch 100 --prepare_only
```

This will:
- Generate batch request files in the `batch_output` directory
- Skip the processing and augmentation steps
- Provide instructions for the next steps

The batch files will be named like: `batch_requests_en-GB_20250501_191835.jsonl`

## Step 2: Process with OpenAI's Batch System

1. Go to the [OpenAI API Playground](https://platform.openai.com/playground) or use the OpenAI API directly
2. Upload the batch files generated in Step 1
3. Start the batch processing job
4. Wait for the processing to complete
5. Download the results

## Step 3: Process the Results

Use the `--process_from_batch` flag to process the results from OpenAI's batch system:

```bash
python scripts/atomic10x_batch_wrapper.py --process_from_batch path/to/downloaded_results.json
```

This will perform two steps:

1. **Process batch results** - Uses `process_batch_results.py` to:
   - Parse the downloaded results
   - Extract conversation data
   - Save the processed conversations to `data/output/aac_conversations_{lang}.jsonl`

2. **Augment AAC data** - Uses `augment_aac_data.py` to:
   - Add noisy utterances for different keyboard layouts (QWERTY, ABC, frequency)
   - Add minimally and fully corrected versions
   - Save the augmented data to `data/output/augmented_aac_conversations_{lang}.jsonl`

## Benefits

Using this workflow provides several benefits:

1. **Cost-effective** - OpenAI's batch processing system is more cost-effective than making individual API calls
2. **Efficient** - Batch processing is faster for large numbers of requests
3. **Resumable** - You can process the results at any time after the batch processing is complete

## Command Line Options

The script supports the following command line options:

- `--languages` - Language codes to process (e.g., en-GB, fr-FR)
- `--all` - Process all supported languages
- `--requests_per_batch` - Number of requests per batch (default: 100)
- `--num_batches` - Number of batches to create per language (default: 1)
- `--model` - OpenAI model to use (default: gpt-4-turbo)
- `--atomic_limit` - Limit the number of atomic entries to include
- `--skip` - Steps to skip (prepare, translate, batch, process, augment)
- `--verbose` - Print verbose output
- `--prepare_only` - Only prepare batch files without processing them
- `--process_from_batch` - Process results from a batch file downloaded from OpenAI

## Example Workflow

```bash
# Step 1: Prepare batch files for English and French
python scripts/atomic10x_batch_wrapper.py --languages en-GB fr-FR --num_batches 10 --requests_per_batch 100 --prepare_only

# Step 2: Upload the batch files to OpenAI's batch processing system and download the results

# Step 3: Process the results
python scripts/atomic10x_batch_wrapper.py --process_from_batch batch_output/downloaded_results_en-GB.json
python scripts/atomic10x_batch_wrapper.py --process_from_batch batch_output/downloaded_results_fr-FR.json
```
