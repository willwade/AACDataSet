import json
import os
import random
import re
import sys
import pandas as pd
from pathlib import Path
from jinja2 import Template

# ---
# This script requires ATOMIC10x data in Parquet format.
# Download from: https://allenai.org/data/atomic-2020
# Place as: data/atomic10x_processed/ATOMIC10X_with_literals.parquet
# ---

# Paths
ATOMIC_PATH = "../data/atomic10x_processed/ATOMIC10X_with_literals.parquet"
SUBSTITUTIONS_PATH = "../substitutions/en-GB.json"
OUTPUT_DIR = Path("./")
ENRICHED_SUBSTITUTIONS_PATH = OUTPUT_DIR / "en-GB.enriched.json"
ROLE_MAP_PATH = OUTPUT_DIR / "aac_user_role_map.json"
BATCH_OUTPUT_PATH = OUTPUT_DIR / "aac_conversations_en-GB.jsonl"
TEMPLATE_PATH = "../prompt_templates/en-GB.json"

# Check for ATOMIC10x data file
if not os.path.exists(ATOMIC_PATH):
    print(f"ERROR: ATOMIC10x data file not found at {ATOMIC_PATH}\n"
          f"Download from: https://allenai.org/data/atomic-2020\n"
          f"Place the file as: {ATOMIC_PATH}\n")
    sys.exit(1)

# --- Enrichment ---
def atomic_phrase_to_topic(phrase, x_name, y_name):
    topic = phrase.replace("PersonX", x_name)
    if y_name:
        topic = topic.replace("PersonY", y_name)
    topic = topic.strip()
    if topic.startswith("to "):
        topic = topic[3:]
    topic = topic.rstrip('. ')
    if topic.split() and topic.split()[0] in {"becomes", "is", "has", "joins", "calls"}:
        topic = "Discussing " + topic
    if len(topic.split()) < 2:
        return None
    return topic

def enrich_substitutions():
    df = pd.read_parquet(ATOMIC_PATH)
    with open(SUBSTITUTIONS_PATH, "r") as f:
        substitutions = json.load(f)
    atomic_topics = set()
    aac_user_role_map = []
    atomic_topic_relation_map = []
    for idx, row in df.iterrows():
        x_name = row.get('x', 'Alex') or 'Alex'
        y_name = row.get('y', 'Taylor') or 'Taylor'
        # Randomly assign AAC user to PersonX or PersonY
        if random.random() < 0.5:
            aac_user = x_name
            partner = y_name
            aac_user_role = 'PersonX'
        else:
            aac_user = y_name if y_name else 'Taylor'
            partner = x_name
            aac_user_role = 'PersonY'
        for which, phrase in zip(['head','tail'], [row['head'], row['tail']]):
            if isinstance(phrase, str):
                topic = atomic_phrase_to_topic(phrase, x_name, y_name)
                if topic:
                    atomic_topics.add(topic)
                    rel = row.get('relation', None)
                    atomic_topic_relation_map.append({
                        'topic': topic,
                        'relation': rel,
                        'aac_user': aac_user,
                        'partner': partner,
                        'aac_user_role': aac_user_role,
                        'which': which
                    })
    orig_topics = set(substitutions.get('topic', []))
    new_topics = orig_topics.union(atomic_topics)
    substitutions['topic'] = sorted(new_topics)
    substitutions['atomic_relation'] = sorted(set(df['relation'].dropna().unique()))
    with open(ENRICHED_SUBSTITUTIONS_PATH, "w") as f:
        json.dump(substitutions, f, indent=2, ensure_ascii=False)
    if not ROLE_MAP_PATH.exists():
        with open(ROLE_MAP_PATH, "w") as f:
            json.dump(atomic_topic_relation_map, f, indent=2, ensure_ascii=False)
        print(f"Role map saved to {ROLE_MAP_PATH}")
    else:
        print(f"Role map already exists at {ROLE_MAP_PATH}, not overwritten.")
    print(f"Enriched substitutions saved.")
    return substitutions, atomic_topic_relation_map

# --- Batch Generation ---
def load_templates():
    with open(TEMPLATE_PATH) as f:
        return json.load(f)

def expand_prompt(template, substitutions):
    jinja_template = Template(template)
    context = {}
    for key in substitutions:
        context[key] = random.choice(substitutions[key]) if substitutions[key] else ""
    return jinja_template.render(**context)

def relation_to_scene_phrase(topic, relation, aac_user, partner, aac_user_role, which):
    # Map ATOMIC10x relation type to a natural scene phrase
    # x = AAC user, y = partner (role may be swapped)
    if not relation:
        return f"about {topic}"
    rel_map = {
        'xIntent': f"what {aac_user} was trying to do when {topic}",
        'xNeed': f"what {aac_user} needed before {topic}",
        'xEffect': f"what happened to {aac_user} as a result of {topic}",
        'xReact': f"how {aac_user} felt after {topic}",
        'xWant': f"what {aac_user} wanted to do next after {topic}",
        'xAttr': f"how {aac_user} is described when {topic}",
        'oEffect': f"what happened to {partner} as a result of {topic}",
        'oReact': f"how {partner} felt after {topic}",
        'oWant': f"what {partner} wanted to do next after {topic}",
        'oNeed': f"what {partner} needed before {topic}",
        'isAfter': f"what happens after {topic}",
        'isBefore': f"what happens before {topic}",
        'HinderedBy': f"what could prevent {aac_user if aac_user_role=='PersonX' else partner} from {topic}",
    }
    return rel_map.get(relation, f"about {topic}")

def generate_batch(num_examples=10):
    templates = load_templates()
    with open(ENRICHED_SUBSTITUTIONS_PATH) as f:
        substitutions = json.load(f)
    with open(ROLE_MAP_PATH) as f:
        atomic_topic_relation_map = json.load(f)
    batch = []
    for _ in range(num_examples):
        template_id = random.randint(0, len(templates) - 1)
        template = templates[template_id]
        # Pick a random topic+relation entry
        rel_entry = random.choice(atomic_topic_relation_map)
        scene_phrase = relation_to_scene_phrase(
            rel_entry['topic'],
            rel_entry['relation'],
            rel_entry['aac_user'],
            rel_entry['partner'],
            rel_entry['aac_user_role'],
            rel_entry['which']
        )
        # Replace {topic} or similar in template with scene_phrase, or append
        scene = expand_prompt(template, substitutions)
        # Try to inject the scene_phrase more naturally
        if '{topic}' in scene:
            scene = scene.replace('{topic}', scene_phrase)
        else:
            scene = scene.rstrip('.') + f' The conversation is {scene_phrase}.'
        batch.append({
            "template_id": template_id,
            "scene": scene,
            "conversation": [
                {"speaker": "AAC User", "utterance": "...", "utterance_intended": "...", "is_aac_user": True},
                {"speaker": "Partner", "utterance": "...", "utterance_intended": "...", "is_aac_user": False}
            ]
        })
    with open(BATCH_OUTPUT_PATH, "w") as f:
        for req in batch:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    print(f"Batch saved to {BATCH_OUTPUT_PATH}")
    return batch

if __name__ == "__main__":
    enrich_substitutions()
    batch = generate_batch(10)
    for ex in batch[:3]:
        print(json.dumps(ex, indent=2, ensure_ascii=False))
