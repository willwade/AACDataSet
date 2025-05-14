# AAC Conversation Viewer

A web-based viewer for browsing and exploring AAC conversations from the AAC Dataset project.

## Features

- Browse conversations by language
- View conversation details including:
  - Scene descriptions
  - Speaker turns
  - AAC user's intended vs. actual utterances
  - Metadata information
- Navigate between conversations
- Links to source files

## Getting Started

### Prerequisites

- Python 3.6 or higher

### Running the Viewer

1. Navigate to the project root directory
2. Run the server script:

```bash
python web/server.py
```

3. Open your web browser and go to: http://localhost:8080/web/

### Usage

1. Select a language from the dropdown menu
2. Use the "Previous" and "Next" buttons to navigate between conversations
3. View conversation details in the main panel
4. See metadata and file links in the side panel

## File Structure

The conversation viewer works with the following file structure:

```
batch_files/
  ├── [language-code]/
  │   ├── augmented_batch_conversations_transformed.jsonl
  │   └── (other files)
  └── (other language directories)
web/
  ├── index.html
  ├── styles.css
  ├── script.js
  ├── server.py
  └── README.md
```

## Customization

You can customize the viewer by modifying the following files:

- `styles.css` - Change the appearance of the viewer
- `script.js` - Modify the functionality or add new features
- `index.html` - Change the structure of the viewer

## Batches Info


Batch 1. 100x Conversations per lang. Had paper based which led to senteces like H e l l o w or ld. No utterance_intended
Batch 2. 100x Conversations per lang.Fixed Batch 1. But realised the atomic data was very small. 
Batch 3. 100x Conversations per lang.Added way more richer atomic data subset. 

All batches are in the parquet - but you may want to filter them. 

## Troubleshooting

If you encounter issues:

1. Make sure the server is running
2. Check that the file paths are correct
3. Verify that the JSONL files are properly formatted
4. Check the browser console for any JavaScript errors

## License

This project is part of the AAC Dataset project.
