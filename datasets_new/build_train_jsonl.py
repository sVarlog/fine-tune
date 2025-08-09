import os
import json
import argparse

STRUCTURE_FILE = "structure.enriched.json"
TOPICS_DIR = "topics"

def load_structure(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Missing dataset file: {path}")
    except json.JSONDecodeError:
        print(f"⚠️ Invalid JSON, skipping: {path}")
    return []

def pick(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def normalize_entry(
    entry, *,
    category, subcat, ctype, meta_tags, source_version, fallback_topic, next_id
):
    # Core fields (tolerate minor schema variants)
    q = pick(entry, "question", "q", default=None)
    think = pick(entry, "think", "rationale", "reasoning", default=None)
    out = pick(entry, "output", "answer", "a", default=None)

    # If any are missing, warn and skip by returning None
    if not q or not out:
        print(f"⚠️ Skipping sample without required fields (question/output) in {category}/{subcat}/{ctype}")
        return None

    row = {
        "id": entry.get("id", next_id),
        "category": category,
        "subcategory": subcat,
        "content_type": ctype,
        "topic": entry.get("topic", fallback_topic),
        "difficulty": entry.get("difficulty", 3),
        "question": q,
        "think": think if think is not None else "",
        "output": out,
        # Use metadata tags; if entry has tags, union them with meta tags (de-dup, keep order).
        "tags": list(dict.fromkeys((entry.get("tags") or []) + (meta_tags or []))),
        "source_version": source_version,
    }
    return row

def export_jsonl(output_path, rows):
    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Flatten topics/* into JSONL with per-sample metadata.")
    ap.add_argument("--structure", default=STRUCTURE_FILE, help="Path to structure.enriched.json")
    ap.add_argument("--topics-dir", default=TOPICS_DIR, help="Path to topics directory")
    ap.add_argument("--output", default="train_data.jsonl", help="Output JSONL filename")
    ap.add_argument("--source-version", default="v1.0", help="Version stamp to include on each row")
    ap.add_argument("--start-id", type=int, default=10_000, help="Start ID if a sample lacks id")
    args = ap.parse_args()

    structure = load_structure(args.structure)

    rows = []
    next_id = args.start_id
    seen_ids = set()

    for category, subcats in structure.items():
        for subcat, meta in subcats.items():
            content_types = meta.get("content_type", [])
            if isinstance(content_types, str):
                content_types = [content_types]
            meta_tags = meta.get("tags", [])
            subcat_dir = os.path.join(args.topics_dir, category, subcat)

            for ctype in content_types:
                filename = f"{category}.{subcat}.{ctype}.json"
                file_path = os.path.join(subcat_dir, filename)
                data = read_json_file(file_path)

                if not isinstance(data, list):
                    print(f"⚠️ Expected a list in {file_path}; skipping.")
                    continue

                fallback_topic = f"{category}.{subcat}"

                for entry in data:
                    row = normalize_entry(
                        entry,
                        category=category,
                        subcat=subcat,
                        ctype=ctype,
                        meta_tags=meta_tags,
                        source_version=args.source_version,
                        fallback_topic=fallback_topic,
                        next_id=next_id
                    )
                    if row is None:
                        continue

                    # Ensure unique IDs; if clash, reassign sequentially
                    rid = row["id"]
                    if rid in seen_ids:
                        rid = next_id
                        row["id"] = rid
                    seen_ids.add(rid)
                    next_id = max(next_id + 1, rid + 1)

                    rows.append(row)

    export_jsonl(args.output, rows)
    print(f"✅ Wrote {len(rows)} rows to {args.output}")

if __name__ == "__main__":
    main()