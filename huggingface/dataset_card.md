# AAC Conversations Dataset

## Dataset Description

The AAC Conversations Dataset is a collection of simulated conversations involving Augmentative and Alternative Communication (AAC) users across multiple languages. This dataset is designed to help researchers and developers build better assistive technologies for people who use AAC devices.

### Dataset Summary

This dataset contains conversations between AAC users and communication partners in various scenarios. Each conversation includes both the original utterances and various augmented versions that simulate different types of typing errors that commonly occur when using AAC devices. The dataset supports multiple languages, making it valuable for developing multilingual assistive technologies.

## Dataset Structure

### Data Instances

Each instance in the dataset represents a single utterance from an AAC user, along with context from the conversation and various augmented versions of the utterance.

Example:
```json
{
  "conversation_id": 42,
  "turn_number": 2,
  "language_code": "en-GB",
  "template_id": 15,
  "scene": "At a doctor's appointment",
  "context_speakers": ["Doctor", "Patient (AAC)"],
  "context_utterances": ["How have you been feeling lately?", "Not great"],
  "speaker": "Patient (AAC)",
  "utterance": "I've been having trouble sleeping",
  "utterance_intended": "I've been having trouble sleeping",
  "next_turn_speaker": "Doctor",
  "next_turn_utterance": "How long has this been going on?",
  "noisy_qwerty_minimal": "I've been having troubke sleeping",
  "noisy_qwerty_light": "I've been havng troble sleepng",
  "noisy_qwerty_moderate": "I've ben havin troble sleping",
  "noisy_qwerty_severe": "Ive ben havin trble slping",
  "noisy_abc_minimal": "I've been having troubke sleeping",
  "noisy_abc_light": "I've been havng troble sleepng",
  "noisy_abc_moderate": "I've ben havin troble sleping",
  "noisy_abc_severe": "Ive ben havin trble slping",
  "noisy_frequency_minimal": "I've been having troubke sleeping",
  "noisy_frequency_light": "I've been havng troble sleepng",
  "noisy_frequency_moderate": "I've ben havin troble sleping",
  "noisy_frequency_severe": "Ive ben havin trble slping",
  "minimally_corrected": "I've been having trouble sleeping.",
  "fully_corrected": "I've been having trouble sleeping."
}
```

### Data Fields

#### Conversation Structure Fields
- `conversation_id`: Unique identifier for each conversation
- `turn_number`: The position of this utterance in the conversation
- `language_code`: The language code of the conversation (e.g., "en-GB", "fr-FR")
- `template_id`: Identifier for the conversation template
- `scene`: Description of the conversation setting

#### Speaker and Utterance Fields
- `speaker`: The speaker of the current utterance
- `utterance`: The original utterance as typed by the AAC user
- `utterance_intended`: The intended utterance (what the user meant to type)

#### Context Fields (Flattened for Better Usability)
- `context_speakers`: List of speakers for the previous turns (up to 3)
- `context_utterances`: List of utterances for the previous turns (up to 3)
- `next_turn_speaker`: Speaker of the next turn in the conversation
- `next_turn_utterance`: Utterance of the next turn in the conversation

#### Augmented Utterance Fields
- `noisy_qwerty_minimal`: Utterance with minimal typing errors based on QWERTY keyboard layout
- `noisy_qwerty_light`: Utterance with light typing errors based on QWERTY keyboard layout
- `noisy_qwerty_moderate`: Utterance with moderate typing errors based on QWERTY keyboard layout
- `noisy_qwerty_severe`: Utterance with severe typing errors based on QWERTY keyboard layout
- `noisy_abc_minimal`: Utterance with minimal typing errors based on ABC keyboard layout
- `noisy_abc_light`: Utterance with light typing errors based on ABC keyboard layout
- `noisy_abc_moderate`: Utterance with moderate typing errors based on ABC keyboard layout
- `noisy_abc_severe`: Utterance with severe typing errors based on ABC keyboard layout
- `noisy_frequency_minimal`: Utterance with minimal typing errors based on frequency keyboard layout
- `noisy_frequency_light`: Utterance with light typing errors based on frequency keyboard layout
- `noisy_frequency_moderate`: Utterance with moderate typing errors based on frequency keyboard layout
- `noisy_frequency_severe`: Utterance with severe typing errors based on frequency keyboard layout
- `minimally_corrected`: Minimally corrected version of the utterance
- `fully_corrected`: Fully corrected version of the utterance

### Languages

The dataset includes conversations in multiple languages:
- English (en, en-GB, en-US)
- French (fr, fr-FR)
- German (de, de-DE)
- Spanish (es, es-ES)
- Italian (it, it-IT)
- Dutch (nl, nl-NL)
- Greek (el, el-GR)
- Russian (ru, ru-RU)
- Hebrew (he, he-IL)
- Arabic (ar, ar-SA)
- Portuguese (pt, pt-BR)
- Welsh (cy, cy-GB)
- Japanese (ja, ja-JP)
- Chinese (zh, zh-CN)

### Dataset Statistics

Here are some key statistics about the dataset:

| Language | Conversations | Total Turns | AAC Utterances | Non-AAC Utterances | Avg Turns/Conv | AAC MLU | Non-AAC MLU |
|----------|---------------|-------------|----------------|-------------------|---------------|---------|-------------|
| en-GB    | 48            | 384         | 192            | 192               | 8.00          | 7.25    | 11.50       |
| fr-FR    | 48            | 384         | 192            | 192               | 8.00          | 6.75    | 10.80       |
| de-DE    | 48            | 384         | 192            | 192               | 8.00          | 6.50    | 10.25       |
| ... (other languages) ... |
| TOTAL    | 576           | 4,608       | 2,304          | 2,304             | 8.00          | 6.85    | 10.75       |

*Note: These are example statistics. Actual numbers will vary based on the current dataset.*

**MLU**: Mean Length of Utterance (average number of words per utterance)

## Dataset Creation

### Curation Rationale

AAC users often experience challenges with text entry that can lead to typing errors. This dataset was created to help develop and evaluate technologies that can assist AAC users by correcting typing errors, predicting text, and improving communication efficiency across multiple languages.

### Source Data

The conversations in this dataset are simulated based on common scenarios that AAC users might encounter in daily life, including medical appointments, social interactions, educational settings, and more.

### Generation Process

The dataset was created through a multi-step process:

1. **Generation**: Conversations were generated using templates and LLM (Gemini) to create realistic AAC interactions
2. **Augmentation**: AAC utterances were augmented with various noise levels and keyboard layouts
3. **Correction**: Both minimal and full corrections were added to each AAC utterance
4. **Multilingual Expansion**: Templates were translated and adapted for multiple languages

### Annotations

The dataset includes several types of augmented utterances that simulate typing errors:

- **Error Rates**:
  - Minimal: 5% errors - very mild typing issues
  - Light: 15% errors - noticeable but clearly readable
  - Moderate: 25% errors - challenging but comprehensible
  - Severe: 35% errors - significant difficulty

- **Keyboard Layouts**:
  - QWERTY: Standard keyboard layout
  - ABC: Alphabetical keyboard layout
  - Frequency: Layout based on letter frequency

Each language uses appropriate keyboard layouts and letter frequencies for that language.

### Personal and Sensitive Information

This dataset does not contain any personal or sensitive information. All conversations are simulated and do not represent real individuals.

## Potential Uses

This dataset can be used for a variety of NLP tasks related to AAC:

1. **AAC Utterance Correction**: Train models to correct noisy AAC input
2. **Telegraphic Speech Expansion**: Expand telegraphic AAC utterances into grammatically complete sentences
3. **AAC Response Prediction**: Predict appropriate responses to AAC utterances
4. **AAC Interface Optimization**: Study error patterns across different keyboard layouts
5. **Multilingual Assistive Technology**: Develop assistive technologies that work across multiple languages
6. **Cross-lingual Transfer Learning**: Explore how models trained on one language can be adapted to others

## Considerations for Using the Data

### Social Impact of Dataset

This dataset aims to improve assistive technologies for people who use AAC devices, potentially enhancing their communication abilities and quality of life across multiple languages and cultures.

### Discussion of Biases

The dataset attempts to represent diverse scenarios and contexts, but may not capture all the nuances of real AAC user experiences. Users of this dataset should be aware of potential biases in the simulated conversations.

### Other Known Limitations

- The typing errors are generated algorithmically and may not perfectly represent the patterns of errors that real AAC users make
- Some languages have more comprehensive support than others
- The dataset focuses primarily on text-based communication and does not include symbol-based AAC

## Additional Information

### Dataset Curators

This dataset was curated by Will Wade and the team at Ace Centre.

### Licensing Information

This dataset is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

### Citation Information

If you use this dataset in your research, please cite:

```
@dataset{aac_conversations_dataset,
  author = {Wade, Will},
  title = {AAC Conversations Dataset},
  year = {2023},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/willwade/AACConversations}
}
```

### Contributions

Thanks to all who contributed to the creation of this dataset! Special thanks to the Ace Centre team and the AAC community for their insights and guidance.

## How to Use

Here's a simple example of how to load and explore the dataset:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("willwade/AACConversations")

# Print the first example
print(dataset['dataset'][0])

# Filter examples by language
english_examples = [ex for ex in dataset['dataset'] if ex['language_code'].startswith('en')]
print(f"Number of English examples: {len(english_examples)}")

# Example of a task: AAC utterance correction
for example in dataset['dataset'][:5]:
    print(f"Original: {example['noisy_qwerty_moderate']}")
    print(f"Corrected: {example['fully_corrected']}")
    print()

# Reconstruct a conversation
conversation_id = dataset['dataset'][0]['conversation_id']
conversation_turns = [ex for ex in dataset['dataset'] if ex['conversation_id'] == conversation_id]
conversation_turns.sort(key=lambda x: x['turn_number'])

print(f"Conversation {conversation_id}:")
for turn in conversation_turns:
    print(f"{turn['speaker']}: {turn['utterance']}")
    if turn['next_turn_speaker'] and turn['next_turn_utterance']:
        print(f"{turn['next_turn_speaker']}: {turn['next_turn_utterance']}")
```
