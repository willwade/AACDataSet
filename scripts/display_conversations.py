#!/usr/bin/env python3
"""
Display conversations from a responses file in a readable format.
"""
import json
import argparse
import random

def display_conversation(conversation, index, show_prompt=False, prompt=""):
    """
    Display a conversation in a readable format.
    
    Args:
        conversation: The conversation to display
        index: The index of the conversation
        show_prompt: Whether to show the prompt
        prompt: The prompt used to generate the conversation
    """
    print(f"\n{'='*80}")
    print(f"CONVERSATION {index}")
    print(f"{'='*80}")
    
    if show_prompt:
        print(f"\nPROMPT:\n{prompt}\n")
        print(f"{'-'*80}")
    
    scene = conversation.get("scene", "")
    if scene:
        print(f"SCENE: {scene}\n")
    
    for turn in conversation.get("conversation", []):
        speaker = turn.get("speaker", "Unknown")
        utterance = turn.get("utterance", "")
        utterance_intended = turn.get("utterance_intended", "")
        is_aac_user = turn.get("is_aac_user", False)
        
        if is_aac_user:
            print(f"{speaker} (AAC User):")
            print(f"  Intended: {utterance_intended}")
            print(f"  Actual: {utterance}")
        else:
            print(f"{speaker}:")
            print(f"  {utterance}")
        print()

def main():
    parser = argparse.ArgumentParser(description="Display conversations from a responses file")
    parser.add_argument("file", help="Responses file")
    parser.add_argument("--num", type=int, default=5, help="Number of conversations to display")
    parser.add_argument("--show-prompt", action="store_true", help="Show the prompt used to generate the conversation")
    parser.add_argument("--random", action="store_true", help="Select random conversations")
    
    args = parser.parse_args()
    
    with open(args.file, "r") as f:
        responses = json.load(f)
    
    print(f"Loaded {len(responses)} conversations from {args.file}")
    
    # Select conversations to display
    if args.random:
        selected = random.sample(responses, min(args.num, len(responses)))
    else:
        selected = responses[:args.num]
    
    # Display each conversation
    for i, response in enumerate(selected):
        if "response" in response and isinstance(response["response"], dict):
            display_conversation(
                response["response"], 
                i+1, 
                args.show_prompt, 
                response.get("prompt", "")
            )
        else:
            print(f"\n{'='*80}")
            print(f"CONVERSATION {i+1} - ERROR")
            print(f"{'='*80}")
            print(response.get("error", "Unknown error"))

if __name__ == "__main__":
    main()
