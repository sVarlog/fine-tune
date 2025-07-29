import os
import json

BASE_DIR = os.path.dirname(__file__)
TOPIC_DIRS = [
    os.path.join(BASE_DIR, d) for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d)) and not d.startswith('__')
]

all_data = []
for topic_dir in TOPIC_DIRS:
    jsonl_path = os.path.join(topic_dir, 'dataset.jsonl')
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            all_data.extend([json.loads(line) for line in f if line.strip()])

output_path = os.path.join(BASE_DIR, 'data.jsonl')
with open(output_path, 'w', encoding='utf-8') as out_file:
    for entry in all_data:
        out_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"[INFO] Merged {len(all_data)} samples from {len(TOPIC_DIRS)} folders.")
