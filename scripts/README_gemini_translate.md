# Gemini Translation Script Documentation

## Overview

The `gemini_translate_templates.py` script translates JSON templates using Google's Gemini API. This approach is much faster than using Google Translate for each string and is currently **free of charge** under Gemini's pricing model.

## Features

- **Free Translation**: Uses Google's Gemini API which is currently free for both input and output
- **Adaptive Batch Processing**: Dynamically adjusts batch size for optimal performance
- **Token Estimation**: Monitors token usage to stay within model limits
- **Rate Limiting**: Respects Gemini API rate limits to avoid errors
- **Checkpointing**: Saves progress regularly and can resume from where it left off
- **Progress Tracking**: Shows detailed progress information with time estimates

## Prerequisites

1. Install required dependencies:
   ```bash
   pip install google-generativeai tqdm python-dotenv
   ```

2. Get a Gemini API key from https://ai.google.dev/

3. Create a `.env` file in the project root with your API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

Basic usage:

```bash
python scripts/gemini_translate_templates.py --lang fr-FR
```

Translate to multiple languages:

```bash
python scripts/gemini_translate_templates.py --languages fr-FR de-DE it-IT
```

Translate to all supported languages:

```bash
python scripts/gemini_translate_templates.py --all
```

### Common Options

- `--skip-english`: Skip all English variants (en-*)
- `--skip-unsupported`: Skip languages not supported by Gemini
- `--skip-completed`: Skip languages that already have completed output files
- `--resume`: Resume from checkpoints if available
- `--estimate-only`: Only estimate time without performing translations

### Advanced Options

- `--batch-size`: Initial number of items to translate in one API call (default: 15)
- `--rate-limit`: Maximum requests per minute (default: 30)

## Examples

### Estimate time for all languages

```bash
python scripts/gemini_translate_templates.py --all --skip-english --skip-unsupported --estimate-only
```

### Resume a previously interrupted translation

```bash
python scripts/gemini_translate_templates.py --all --skip-english --skip-unsupported --skip-completed --resume
```

### Translate a specific language with custom settings

```bash
python scripts/gemini_translate_templates.py --lang fr-FR --batch-size 10 --rate-limit 20
```

## Adaptive Batch Sizing

The script uses an adaptive approach to batch sizing:

1. **Initial Batch Size**: Starts with a batch size of 15 items
2. **Token Monitoring**: Estimates token usage for each batch
3. **Dynamic Adjustment**:
   - Increases batch size if token usage is low
   - Decreases batch size if token usage is high
   - Reduces batch size on API errors
4. **Minimum/Maximum Limits**: Ensures batch size stays between 3 and 25 items

This approach maximizes throughput while staying within model limits.

## Checkpointing

The script creates checkpoints in the `checkpoints` directory. If a translation is interrupted, you can resume it by running the script again with the `--resume` flag.

Checkpoints are automatically removed when a language is successfully completed.

## Time Estimates

The script provides time estimates based on the rate limit and adaptive batch sizing:
- Default rate limit: 60 requests per minute (using Gemini 1.5 Flash)
- Initial batch size: 15 items per request (adjusts dynamically)

For a full run of all 32 languages with 48,710 entries:
- Estimated average batch size: ~12 items
- Estimated total batches: ~130,000
- Estimated time: ~36 hours (~1.5 days) at 60 requests per minute

This is significantly faster than the Google Translate approach (36 days) and completely free!

> **Note**: Gemini 1.5 Flash is used instead of Gemini 1.5 Pro because it has higher rate limits and is more cost-effective for this task. The quality is still excellent for translation purposes.

## Tips

1. Run the script on a server that can operate continuously
2. Use the `--skip-completed` and `--resume` options to handle interruptions
3. Use `screen` or `tmux` to keep the script running in the background
4. Check partial results periodically to monitor translation quality
