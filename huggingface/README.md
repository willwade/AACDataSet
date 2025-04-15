# Hugging Face Dataset Preparation

This directory contains scripts and documentation for preparing the AAC Conversations Dataset for Hugging Face.

## Directory Structure

- `scripts/`: Python scripts for preparing and uploading the dataset
- `data/`: Directory where the prepared dataset will be stored
- `docs/`: Documentation for the dataset

## Usage

### 1. Prepare the Dataset

To convert the augmented AAC conversations to Hugging Face format:

```bash
cd scripts
python prepare_huggingface_dataset.py --input ../../output/augmented_aac_conversations_en.jsonl --output_dir ../data --split_ratio 0.8,0.1,0.1
```

For locale-specific language codes (e.g., en-GB, es-ES):

```bash
cd scripts
python prepare_huggingface_dataset.py --input ../../output/augmented_aac_conversations_en-GB.jsonl
```

If you don't specify an output directory, it will automatically create one based on the language code (e.g., `../data/en-GB`).

This will:
- Load the augmented conversations from the JSONL file
- Flatten the data to create one entry per AAC utterance
- Split the data into train, validation, and test sets
- Save the data in Hugging Face format

### 2. Explore Example Usage

To see examples of how the dataset can be used:

```bash
cd scripts
python example_usage.py --data_dir ../data
```

This will demonstrate:
- AAC utterance correction (noisy to clean)
- AAC utterance expansion (telegraphic to full)
- AAC response prediction

### 3. Upload to Hugging Face (Optional)

To upload the dataset to Hugging Face:

```bash
cd scripts
python upload_to_huggingface.py --input_dir ../data --repo_id your-username/aac-conversations
```

Note: You need to have a Hugging Face account and be logged in via the Hugging Face CLI.

## Dataset Documentation

See the [dataset documentation](docs/README.md) for details about the dataset structure, potential uses, and ethical considerations.
