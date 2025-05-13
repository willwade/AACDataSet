# Language Support for AAC Dataset

This document explains how to add support for new languages to the AAC Dataset.

## Overview

The AAC Dataset supports multiple languages through:

1. **Substitution Files**: Language-specific substitutions for names, settings, etc.
2. **Atomic Data Translation**: Translating atomic data entries for each language
3. **Language-Specific Keyboard Layouts**: For generating realistic typing errors

## Adding Support for a New Language

### 1. Generate Substitution Files

The `generate_missing_substitutions.py` script can create substitution files for all languages listed in the README:

```bash
python scripts/generate_missing_substitutions.py
```

This will:
- Check which languages are missing substitution files
- Generate culturally appropriate names for each language using OpenAI
- Create substitution files in `templates/substitutions/`

To regenerate all substitution files (even existing ones):

```bash
python scripts/generate_missing_substitutions.py --force
```

### 2. Translate Atomic Data

Once you have substitution files, you can translate the atomic data:

```bash
python scripts/translate_atomic_data.py --lang <language-code>
```

For example:
```bash
python scripts/translate_atomic_data.py --lang fr-FR
```

To translate for all supported languages:
```bash
python scripts/translate_atomic_data.py --all
```

### 3. Generate Batch Files

After translating the atomic data, you can generate batch files for OpenAI processing:

```bash
python scripts/atomic10x_batch_wrapper.py --languages <language-code> --prepare_only
```

For example:
```bash
python scripts/atomic10x_batch_wrapper.py --languages fr-FR --prepare_only
```

### 4. Process Batch Results

After processing the batch files with OpenAI's batch system, you can process the results:

```bash
python scripts/atomic10x_batch_wrapper.py --process_from_batch <batch-results-file>
```

## Supported Languages

The following languages are supported by the AAC Dataset:

| Language Code | Language | Status |
|--------------|----------|--------|
| af-ZA | Afrikaans (South Africa) | Planned |
| ar-SA | Arabic (Saudi Arabia) | Supported |
| eu-ES | Basque (Spain) | Planned |
| ca-ES | Catalan (Spain) | Planned |
| hr-HR | Croatian (Croatia) | Planned |
| cs-CZ | Czech (Czechia) | Planned |
| da-DK | Danish (Denmark) | Planned |
| nl-BE | Dutch (Belgium) | Planned |
| nl-NL | Dutch (Netherlands) | Supported |
| en-AU | English (Australia) | Planned |
| en-CA | English (Canada) | Planned |
| en-NZ | English (New Zealand) | Planned |
| en-ZA | English (South Africa) | Planned |
| en-GB | English (United Kingdom) | Supported |
| en-US | English (United States) | Supported |
| fo-FO | Faroese (Faroe Islands) | Planned |
| fi-FI | Finnish (Finland) | Planned |
| fr-CA | French (Canada) | Planned |
| fr-FR | French (France) | Supported |
| de-AT | German (Austria) | Planned |
| de-DE | German (Germany) | Supported |
| el-GR | Greek (Greece) | Supported |
| he-IL | Hebrew (Israel) | Supported |
| it-IT | Italian (Italy) | Supported |
| nb-NO | Norwegian Bokmål (Norway) | Planned |
| pl-PL | Polish (Poland) | Planned |
| pt-BR | Portuguese (Brazil) | Supported |
| pt-PT | Portuguese (Portugal) | Planned |
| ru-RU | Russian (Russia) | Supported |
| sk-SK | Slovak (Slovakia) | Planned |
| sl-SI | Slovenian (Slovenia) | Planned |
| es-ES | Spanish (Spain) | Supported |
| es-US | Spanish (United States) | Planned |
| sv-SE | Swedish (Sweden) | Planned |
| uk-UA | Ukrainian (Ukraine) | Planned |
| cy-GB | Welsh (United Kingdom) | Supported |
| zh-CN | Chinese (China) | Supported |
| ja-JP | Japanese (Japan) | Supported |
| ko-KR | Korean (Korea) | Planned |

## Language-Specific Considerations

### Right-to-Left Languages

For right-to-left languages (Arabic, Hebrew), the keyboard layouts and error generation are adapted to handle RTL text correctly.

### Non-Latin Scripts

For languages with non-Latin scripts (Chinese, Japanese, Korean, etc.), the keyboard layouts and error generation are adapted to handle the specific characteristics of these scripts.

### Locale-Specific Variations

For languages with multiple locale-specific variations (e.g., English, French, Spanish), each variation has its own substitution file with culturally appropriate names and settings.

## Troubleshooting

If you encounter issues with a specific language:

1. Check if the substitution file exists in `templates/substitutions/`
2. Check if the atomic data has been translated in `templates/atomic10x/atomic10x_als_subset_<lang>.json`
3. Try running the translation step manually:
   ```bash
   python scripts/translate_atomic_data.py --lang <language-code>
   ```
4. Check if the OpenAI API key is set in your environment variables
