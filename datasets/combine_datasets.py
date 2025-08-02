import os
import json
import re

# It was needed for the old structure for transforming the dataset from jsonl to a single json file
def transform_folder_to_json(folder_path: str):
    """
    Read `dataset.jsonl` in folder_path (if it exists),
    transform each record into {id, topic, question, think, output},
    then write a single JSON array to `combined_dataset.json` in folder_path.
    """
    input_file = os.path.join(folder_path, "dataset.jsonl")
    output_file = os.path.join(folder_path, "combined_dataset.json")

    if not os.path.isfile(input_file):
        print(f" ↳ no dataset.jsonl in {folder_path}, skipping")
        return

    topic = os.path.basename(folder_path)
    print(f"\n🔄 Processing topic `{topic}` → {input_file}")

    records = []
    with open(input_file, "r", encoding="utf-8") as inf:
        for idx, line in enumerate(inf, start=1):
            rec = json.loads(line)
            q = rec.get("question", "").strip()
            r = rec.get("response", "").strip()

            # Extract <think>…</think> and <output>…</output>
            think_m  = re.search(r"<think>(.*?)</think>",  r, re.DOTALL)
            output_m = re.search(r"<output>(.*?)</output>", r, re.DOTALL)

            think  = think_m.group(1).strip() if think_m  else None
            output = output_m.group(1).strip() if output_m else None

            records.append({
                "id":       idx,
                "topic":    topic,
                "question": q,
                "think":    think,
                "output":   output
            })

    # Write the full list out as pretty JSON
    with open(output_file, "w", encoding="utf-8") as outf:
        json.dump(records, outf, ensure_ascii=False, indent=2)

    print(f" ✅ Wrote {len(records)} records to {output_file}")

def main():
    # Assumes this script lives in the `datasets/` directory
    base_dir = os.path.dirname(__file__)
    for entry in sorted(os.listdir(base_dir)):
        topic_dir = os.path.join(base_dir, entry)
        if os.path.isdir(topic_dir):
            transform_folder_to_json(topic_dir)

if __name__ == "__main__":
    main()