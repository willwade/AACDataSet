import json
import argparse
from pathlib import Path


def pretty_print_conversation(convo, show_metadata=False):
    print(f"\n{'='*60}")
    print(f"Scene: {convo.get('scene', 'N/A')}")
    print(f"Template ID: {convo.get('template_id', 'N/A')}")
    print("Conversation:")
    for turn in convo.get('conversation', []):
        speaker = turn.get('speaker', 'Unknown')
        utterance = turn.get('utterance', '')
        intended = turn.get('utterance_intended', '')
        is_aac = turn.get('is_aac_user', False)
        aac_tag = " [AAC User]" if is_aac else ""
        print(f"  {speaker}{aac_tag}: {utterance}")
        if is_aac and intended and intended != utterance:
            print(f"    → Intended: {intended}")
    if show_metadata and 'metadata' in convo:
        print("\nMetadata:")
        for k, v in convo['metadata'].items():
            print(f"  {k}: {v}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Pretty print AAC conversations from a JSONL file.")
    parser.add_argument("file", help="Path to the .jsonl file to pretty print")
    parser.add_argument("--max", type=int, default=5, help="Maximum number of conversations to print (default: 5)")
    parser.add_argument("--metadata", action="store_true", help="Show metadata for each conversation")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {args.file}")
        return

    with open(path, "r", encoding="utf-8") as f:
        count = 0
        for line in f:
            if args.max and count >= args.max:
                break
            try:
                convo = json.loads(line)
                pretty_print_conversation(convo, show_metadata=args.metadata)
                count += 1
            except Exception as e:
                print(f"Error parsing line {count+1}: {e}")

if __name__ == "__main__":
    main()
