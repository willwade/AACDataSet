# Archived Scripts

This directory contains scripts that are no longer part of the main workflow but are kept for reference.

## Why These Scripts Are Archived

1. **atomic10x_workflow.py**
   - Superseded by `atomic10x_batch_wrapper.py` which has more features and better progress tracking
   - The batch wrapper script provides a more comprehensive workflow with better error handling and progress reporting

2. **generate_batch_20.py**
   - Redundant with `batch_openai_prepare.py` and `test_batch_generation.py`
   - These newer scripts provide more flexibility and better integration with the overall workflow

3. **local_generate.py**
   - Uses a different approach (llm library) than the current OpenAI batch workflow
   - The current workflow focuses on using OpenAI's batch processing system for cost-effectiveness

4. **test_atomic_integration.py**
   - Test script that's no longer part of the main workflow
   - Testing functionality is now integrated into other scripts

5. **extract_prompts.py**
   - Simple utility script that's not part of the main workflow
   - Functionality can be achieved with other tools or scripts

6. **pretty_print_conversations.py**
   - Redundant with `display_conversations.py`
   - The display_conversations.py script provides more features and better formatting

## How to Use These Scripts

If you need to use any of these scripts, you can still run them from this directory. However, be aware that they may not be compatible with the latest workflow or data formats.

## Current Workflow

The current workflow uses the following scripts:

1. **atomic10x_batch_wrapper.py** - Main workflow script
2. **batch_openai_prepare.py** - Prepares batch requests
3. **process_batch_results.py** - Processes results from OpenAI's batch system
4. **augment_aac_data.py** - Augments AAC data

For more information on the current workflow, see the documentation in the `docs` directory.
