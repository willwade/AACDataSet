# Using OpenAI's Batch API with the AAC Dataset Generator

This guide provides instructions for using OpenAI's Batch API to efficiently generate multilingual AAC conversations at scale.

## Prerequisites

- An OpenAI API key with access to the models you want to use
- The AAC Dataset Generator scripts
- Basic understanding of terminal/command line usage

## Overview of the Workflow

1. Generate batch files with our script
2. Upload batch files to OpenAI's Batch API
3. Process the responses
4. Augment the data with our augmentation script

## Step 1: Generate Batch Files

Our `direct_multilingual_generate.py` script can create batch files compatible with OpenAI's Batch API:

```bash
# Generate for a single language (e.g., 100 English conversations)
python scripts/direct_multilingual_generate.py --lang en-GB --num 100 --batch-prepare --provider openai

# Generate for all supported languages (e.g., 20 conversations each)
python scripts/direct_multilingual_generate.py --lang all --num 20 --batch-prepare --provider openai
```

This creates two files for each language:
- `batch_files/{lang}/openai_batch_{lang}_{timestamp}.jsonl` - OpenAI Batch API compatible file
- `batch_files/{lang}/batch_{lang}_openai_{timestamp}_metadata.jsonl` - Reference file with metadata

The OpenAI-compatible files contain all required parameters for the Batch API:
- `custom_id`: A unique identifier for each request
- `method`: Set to "POST"
- `url`: Set to "/v1/chat/completions"
- `body`: Contains all request parameters including:
  - `model`: The OpenAI model to use (e.g., "gpt-4-turbo")
  - `messages`: The system and user prompts
  - Other parameters like temperature and response format

## Step 2: Upload to OpenAI's Batch API

You can upload the batch files to OpenAI's Batch API in multiple ways:

### Using the OpenAI Web Interface

1. Log in to the [OpenAI Platform](https://platform.openai.com/)
2. Navigate to the Batch API section
3. Upload your `openai_batch_{lang}_{timestamp}.jsonl` file
4. Start the batch processing
5. Download the results when complete

### Using the OpenAI CLI

```bash
# Install the OpenAI CLI if you haven't already
pip install openai

# Upload and process a batch file
openai api batch create \
  --api-key YOUR_API_KEY \
  --file batch_files/en-GB/openai_batch_en-GB_20250512_132744_bec33911.jsonl \
  --output-format jsonl \
  --output batch_files/en-GB/responses_en-GB_20250512.jsonl
```

### Using the OpenAI API Directly

You can also use the API directly with curl or other HTTP clients:

```bash
# First upload your batch file
curl -X POST https://api.openai.com/v1/files \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "purpose=batch" \
  -F "file=@batch_files/en-GB/openai_batch_en-GB_20250512_132744_bec33911.jsonl"

# Create the batch job
curl -X POST https://api.openai.com/v1/batches \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "file_id": "file-abc123",  
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h"
  }'
```

Note: You'll need to replace `file-abc123` with the actual file ID from the upload response.

## Step 3: Process the Responses

After receiving the responses from OpenAI's Batch API, use our script to process them:

```bash
# Process a single language
python scripts/process_openai_batch_responses.py \
  --responses batch_files/en-GB/batch_output.jsonl \
  --metadata batch_files/en-GB/batch_en-GB_openai_20250512_132744_bec33911_metadata.jsonl \
  --output output/en-GB/en-GB_all_conversations.jsonl

# Process all languages in a directory
python scripts/process_openai_batch_responses.py \
  --responses batch_files/responses/ \
  --metadata batch_files/ \
  --output output/
```

This script extracts the assistant's responses from the nested structure of the batch API responses, combines them with the metadata, and creates our standard format JSONL files.

Note that OpenAI's batch API returns responses in a different format than regular API calls, with multiple levels of nesting. Our processing script handles this automatically.

## Step 4: Augment the Data

Once the responses are processed, you can run the augmentation script:

```bash
# Augment a single language
python scripts/augment_aac_data.py --input output/en-GB/en-GB_all_conversations.jsonl --lang en-GB

# Augment all languages
python scripts/augment_aac_data.py --lang all --dir output
```

This adds various noisy and corrected versions of each AAC utterance, creating a rich dataset for training.

## Troubleshooting

### Common Batch API Errors

- **Missing required parameter: 'custom_id'** - Make sure you're using the OpenAI-compatible batch file (openai_batch_{lang}_{timestamp}.jsonl)
- **Missing required parameter: 'method'** - The script should add this automatically; check the batch file
- **Missing required parameter: 'url'** - The script should add this automatically; check the batch file
- **Missing required parameter: 'body'** - Make sure your batch file uses the correct format with a body object
- **Rate limit exceeded** - Space out your batch submissions or request a rate limit increase
- **Processing errors** - If you see "processed 0 conversations" when running the processing script, use the detailed logging to identify the issue. The common problem is that the response format is different than expected.

### File Format Issues

If you get errors about the file format:
1. Check that each line is a valid JSON object
2. Verify there are no blank lines in the file
3. Make sure all required parameters are present

If needed, you can manually fix the batch file before uploading.

## Cost Considerations

- Batch API uses the same pricing as regular API calls
- Processing in batches is more efficient but costs the same per token
- You can check estimated costs before submitting a batch job
- Consider starting with a small batch to test the process

## Advanced Usage

### Customizing Model Parameters

You can edit the script to change parameters like temperature, top_p, etc.:

```python
# In direct_multilingual_generate.py
request["body"]["temperature"] = 0.8  # Default is 0.7
request["body"]["max_tokens"] = 2000  # Default is 1500
```

### Validating API Response Format

If you have trouble with the response format, you can examine the batch output file:

```bash
# View the first line of the batch output
head -n 1 batch_files/en-GB/batch_output.jsonl | python -m json.tool

# Look at the nested structure
python -c "import json; data = json.loads(open('batch_files/en-GB/batch_output.jsonl').readline()); print(json.dumps(data.get('response', {}).get('body', {}), indent=2))"
```

### Processing Partial Results

If a batch job is interrupted or partially completes, you can still process the available responses:

```bash
python scripts/process_openai_batch_responses.py \
  --responses partial_responses.jsonl \
  --metadata batch_files/en-GB/batch_en-GB_openai_20250512_132744_bec33911_metadata.jsonl \
  --output output/en-GB/partial_results.jsonl
```

Then merge the results when the full job completes.

### Batch Job Monitoring

You can monitor batch jobs through the OpenAI dashboard or API:

```bash
# Get batch job status
curl -X GET https://api.openai.com/v1/batches/batch_123 \
  -H "Authorization: Bearer YOUR_API_KEY"

# Cancel a batch job
curl -X POST https://api.openai.com/v1/batches/batch_123/cancel \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Additional Resources

- [OpenAI Batch API Documentation](https://platform.openai.com/docs/api-reference/batch)
- [OpenAI Batch Request Format](https://platform.openai.com/docs/guides/batch-api)
- [OpenAI Rate Limits](https://platform.openai.com/docs/guides/rate-limits)
- [JSONL Format Specification](https://jsonlines.org/)
- [OpenAI Cookbook: Batch Processing](https://github.com/openai/openai-cookbook/blob/main/examples/batch_processing/)