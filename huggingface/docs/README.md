# AAC Conversations Dataset

## Dataset Description

The AAC Conversations Dataset contains simulated conversations involving users of Augmentative and Alternative Communication (AAC) devices, particularly focusing on adults with ALS (Amyotrophic Lateral Sclerosis). The dataset includes both the original AAC utterances and various augmented versions, such as noisy inputs and corrected outputs.

### Dataset Structure

The dataset is organized into three splits:
- `train`: Training set (80% of the data)
- `validation`: Validation set (10% of the data)
- `test`: Test set (10% of the data)

Each example in the dataset contains:

- **Basic Information**:
  - `template_id`: The template used to generate the conversation
  - `scene`: A brief description of the setting and participants
  - `speaker`: The speaker of the current utterance
  - `utterance`: The original AAC utterance
  - `utterance_intended`: The intended meaning of the utterance

- **Context**:
  - `context`: Previous turns in the conversation (up to 3)
  - `next_turn`: The next turn in the conversation (if available)

- **Augmented Versions**:
  - `noisy_qwerty_minimal`, `noisy_qwerty_light`, `noisy_qwerty_moderate`, `noisy_qwerty_severe`: Noisy versions of the utterance simulating errors on a QWERTY keyboard
  - `noisy_abc_minimal`, `noisy_abc_light`, `noisy_abc_moderate`, `noisy_abc_severe`: Noisy versions of the utterance simulating errors on an ABC keyboard
  - `noisy_frequency_minimal`, `noisy_frequency_light`, `noisy_frequency_moderate`, `noisy_frequency_severe`: Noisy versions of the utterance simulating errors on a frequency-based keyboard
  - `minimally_corrected`: A minimally corrected version of the utterance (e.g., fixing spelling errors)
  - `fully_corrected`: A fully corrected version of the utterance (e.g., expanding telegraphic speech)

### Dataset Creation

The dataset was created through a multi-step process:

1. **Generation**: Conversations were generated using templates and LLM (Gemini) to create realistic AAC interactions
2. **Augmentation**: AAC utterances were augmented with various noise levels and keyboard layouts
3. **Correction**: Both minimal and full corrections were added to each AAC utterance

### Source Data

The source data consists of simulated conversations involving AAC users, particularly adults with ALS. The conversations cover various settings, relationships, and topics.

## Potential Uses

This dataset can be used for a variety of NLP tasks related to AAC:

1. **AAC Utterance Correction**: Train models to correct noisy AAC input
2. **Telegraphic Speech Expansion**: Expand telegraphic AAC utterances into grammatically complete sentences
3. **AAC Response Prediction**: Predict appropriate responses to AAC utterances
4. **AAC Interface Optimization**: Study error patterns across different keyboard layouts
5. **Assistive Technology Research**: Develop and evaluate new assistive technologies for AAC users

## Ethical Considerations

While this dataset contains simulated conversations rather than real user data, it's important to consider:

- **Privacy**: Even simulated data should be handled with care
- **Representation**: The dataset focuses on adults with ALS and may not represent all AAC users
- **Bias**: The generated conversations may contain biases from the underlying LLM

## Citation Information

If you use this dataset in your research, please cite:

```
@misc{aac-conversations-dataset,
  title={AAC Conversations Dataset},
  author={AACDataSet Contributors},
  year={2025},
  howpublished={\url{https://github.com/willwade/AACDataSet}},
}
```

## Licensing Information

This dataset is released under [LICENSE].

## Contact Information

For questions or feedback about the dataset, please contact [CONTACT_INFO].
