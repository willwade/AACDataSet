# AACDataSet

A comprehensive dataset of synthesized Augmentative and Alternative Communication (AAC) conversations for adults with ALS, focusing on realistic communication scenarios using eye-gaze, direct selection, and headmouse with keyboard-based systems.

## Overview

This dataset contains simulated conversations between AAC users with ALS and various communication partners (healthcare providers, family members, strangers, etc.). The conversations are designed to represent realistic communication scenarios and include both the actual AAC utterances and their intended meanings, along with various augmented versions to simulate typing errors and corrections.

## Dataset Focus

- **Primary User Group**: Adults with ALS
- **Access Methods**:
  - Eye-gaze tracking
  - Direct selection
  - Headmouse
- **Interface Types**:
  - QWERTY keyboard layouts (majority)
  - ABC layouts (primarily for older adults)
  - Phrase-based systems (minority)

## Data Structure

The dataset is stored in JSONL format (JSON Lines), with each line containing a complete conversation. Each conversation includes:

- **template_id**: The template used to generate the conversation
- **scene**: A brief description of the setting and participants
- **conversation**: An array of conversation turns
- **metadata**: Information about how the conversation was generated

For AAC user turns, the following fields are included:

- **utterance**: The original AAC message as it would appear on the device
- **utterance_intended**: The full intended meaning of the message
- **noisy_utterance**: A version with random typing errors
- **noisy_utterance_qwerty**: Errors based on QWERTY keyboard adjacency
- **noisy_utterance_abc**: Errors based on ABC keyboard layout
- **noisy_utterance_frequency**: Errors based on frequency keyboard layout
- **minimally_corrected**: Basic capitalization and punctuation correction
- **fully_corrected**: Complete grammatical correction

## Example Data

```json
{
  "utterance": "Need water please",
  "utterance_intended": "I need some water, please.",
  "noisy_utterance": "Nee watr plese",
  "noisy_utterance_qwerty": "Nerd wayer pkease",
  "noisy_utterance_abc": "Neef wate rplease",
  "noisy_utterance_frequency": "Need wate rplase",
  "minimally_corrected": "Need water please.",
  "fully_corrected": "I need some water, please."
}
```

## Files

- `output/aac_conversations_en.jsonl`: Original generated conversations
- `output/augmented_aac_conversations_en.jsonl`: Conversations with augmented AAC utterances
- `prompt_templates/en.json`: Templates used to generate conversations
- `substitutions/en.json`: Substitution values for conversation generation
- `scanning_library.py`: Utilities for simulating different keyboard layouts
- `metrics.py`: Metrics for evaluating AAC communication
- `main.py`: Script for generating conversations
- `augment_aac_data.py`: Script for augmenting AAC utterances

## Usage

### Generating Conversations

```bash
python main.py --lang en --num_variations 3
```

This will generate new conversations using the templates in `prompt_templates/en.json` and save them to `output/aac_conversations_en.jsonl`.

### Augmenting Conversations

```bash
python augment_aac_data.py --input output/aac_conversations_en.jsonl --output output/augmented_aac_conversations_en.jsonl
```

This will read the conversations from the input file, augment the AAC utterances with noisy versions and corrections, and save the result to the output file.

## Applications

This dataset can be used for:

1. Training AAC prediction models
2. Developing error correction algorithms
3. Studying patterns in AAC communication
4. Testing the robustness of AAC interfaces
5. Simulating AAC communication for training healthcare providers
6. Developing more efficient AAC interfaces

## License

[Insert appropriate license information here]

## Citation

If you use this dataset in your research, please cite:

```
[Insert citation information here]
```

## Acknowledgments

This dataset was created to support research and development in the field of Augmentative and Alternative Communication, with a focus on improving communication options for adults with ALS.
