# AAC Multilingual Dataset

This dataset contains augmented AAC (Augmentative and Alternative Communication) conversations in multiple languages. It is designed for training and evaluating AAC prediction models, error correction algorithms, and other NLP tasks related to AAC communication.

## Dataset Structure

The dataset is split into train and test sets:
- Train: ~19,640 examples
- Test: ~4,910 examples

Each example represents a single utterance from an AAC user and includes:

- `conversation_id`: Unique identifier for the conversation (prefixed with language code)
- `turn_number`: Position of the utterance in the conversation
- `language_code`: Language code (e.g., en-GB, fr-FR)
- `template_id`: ID of the template used to generate the conversation
- `scene`: Description of the conversation setting
- `context_speakers`: List of speakers for the previous turns (context)
- `context_utterances`: List of utterances for the previous turns (context)
- `speaker`: Speaker of the current utterance
- `utterance`: The actual AAC utterance
- `utterance_intended`: The intended meaning of the utterance
- `next_turn_speaker`: Speaker of the next turn
- `next_turn_utterance`: Utterance of the next turn
- `model`: Model used to generate the conversation
- `provider`: Provider of the model

Additionally, each example includes various noisy versions of the utterance:
- `noisy_qwerty_minimal`: Minimal noise based on QWERTY keyboard adjacency
- `noisy_abc_minimal`: Minimal noise based on ABC keyboard layout
- `noisy_frequency_minimal`: Minimal noise based on frequency keyboard layout
- `noisy_qwerty_light`: Light noise based on QWERTY keyboard adjacency
- `noisy_abc_light`: Light noise based on ABC keyboard layout
- `noisy_frequency_light`: Light noise based on frequency keyboard layout
- `noisy_qwerty_moderate`: Moderate noise based on QWERTY keyboard adjacency
- `noisy_abc_moderate`: Moderate noise based on ABC keyboard layout
- `noisy_frequency_moderate`: Moderate noise based on frequency keyboard layout
- `noisy_qwerty_severe`: Severe noise based on QWERTY keyboard adjacency
- `noisy_abc_severe`: Severe noise based on ABC keyboard layout
- `noisy_frequency_severe`: Severe noise based on frequency keyboard layout
- `minimally_corrected`: Minimally corrected version of the utterance
- `fully_corrected`: Fully corrected version of the utterance

## Languages

The dataset includes conversations in 40 languages:

| Language Code | Language |
|--------------|----------|
| af-ZA | Afrikaans (South Africa) |
| ar-SA | Arabic (Saudi Arabia) |
| ca-ES | Catalan (Spain) |
| cs-CZ | Czech (Czechia) |
| cy-GB | Welsh (United Kingdom) |
| da-DK | Danish (Denmark) |
| de-AT | German (Austria) |
| de-DE | German (Germany) |
| el-GR | Greek (Greece) |
| en-AU | English (Australia) |
| en-CA | English (Canada) |
| en-GB | English (United Kingdom) |
| en-NZ | English (New Zealand) |
| en-US | English (United States) |
| en-ZA | English (South Africa) |
| es-ES | Spanish (Spain) |
| es-US | Spanish (United States) |
| eu-ES | Basque (Spain) |
| fi-FI | Finnish (Finland) |
| fo-FO | Faroese (Faroe Islands) |
| fr-CA | French (Canada) |
| fr-FR | French (France) |
| he-IL | Hebrew (Israel) |
| hr-HR | Croatian (Croatia) |
| it-IT | Italian (Italy) |
| ja-JP | Japanese (Japan) |
| ko-KR | Korean (Korea) |
| nb-NO | Norwegian Bokmål (Norway) |
| nl-BE | Dutch (Belgium) |
| nl-NL | Dutch (Netherlands) |
| pl-PL | Polish (Poland) |
| pt-BR | Portuguese (Brazil) |
| pt-PT | Portuguese (Portugal) |
| ru-RU | Russian (Russia) |
| sk-SK | Slovak (Slovakia) |
| sl-SI | Slovenian (Slovenia) |
| sv-SE | Swedish (Sweden) |
| uk-UA | Ukrainian (Ukraine) |
| zh-CN | Chinese (China) |

## Usage

This dataset can be used for various NLP tasks related to AAC communication:

1. **AAC Prediction**: Predicting the next word or phrase in an AAC conversation
2. **Error Correction**: Correcting errors in AAC utterances
3. **Intent Recognition**: Understanding the intent behind AAC utterances
4. **Multilingual AAC**: Developing AAC systems for multiple languages
5. **Context-aware AAC**: Using conversation context to improve AAC prediction

## Example

Here's an example of how to load and use the dataset:

```python
from datasets import load_from_disk

# Load the dataset
dataset = load_from_disk("huggingface/data/aac_multilingual_dataset")

# Access the train and test splits
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Filter by language
en_gb_dataset = train_dataset.filter(lambda example: example["language_code"] == "en-GB")

# Access an example
example = en_gb_dataset[0]
print(f"Utterance: {example['utterance']}")
print(f"Intended: {example['utterance_intended']}")
print(f"Noisy (QWERTY): {example['noisy_qwerty_moderate']}")
print(f"Fully corrected: {example['fully_corrected']}")
```

## Citation

If you use this dataset in your research, please cite:

```
[Citation information will be added here]
```

## License

[License information will be added here]
