# AAC Conversations Dataset Statistics

This directory contains statistics about the AAC Conversations Dataset, including:

- Number of conversations per language
- Average conversation length (turns)
- Mean length of utterance (MLU) for AAC users and non-AAC users
- Total utterances by speaker type
- Overall totals across all languages

## Files

- `dataset_statistics.csv`: CSV file containing all statistics
- `dataset_statistics.md`: Markdown file with formatted statistics table

## Metrics Explanation

- **Language**: Language code of the conversations
- **Conversations**: Number of complete conversations in the dataset
- **Total Turns**: Total number of dialogue turns across all conversations
- **AAC Utterances**: Number of utterances from AAC users
- **Non-AAC Utterances**: Number of utterances from communication partners
- **AAC Words**: Total word count from AAC users
- **Non-AAC Words**: Total word count from communication partners
- **Avg Turns/Conv**: Average number of turns per conversation
- **AAC MLU**: Mean Length of Utterance for AAC users (average words per utterance)
- **Non-AAC MLU**: Mean Length of Utterance for non-AAC users (average words per utterance)

## Generating Statistics

To regenerate these statistics, run:

```bash
cd huggingface/scripts
python calculate_dataset_stats.py --input_dir ../../output --output_dir ../stats
```

This will analyze all augmented conversation files in the output directory and save the statistics to this directory.

## Using Statistics in Dataset Documentation

These statistics can be included in the dataset documentation to provide users with a better understanding of the dataset's composition and characteristics. They are particularly useful for:

1. Showing the distribution of data across languages
2. Highlighting differences in utterance length between AAC users and communication partners
3. Providing context about the typical structure of conversations in the dataset
