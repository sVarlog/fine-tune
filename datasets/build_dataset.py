import os
import json

import json
from pathlib import Path

def merge_json_to_jsonl(base_dir: Path, output_path: Path, input_name: str = "combined_dataset.json"):
    """
    Walks every subfolder of `base_dir`, looks for `input_name`,
    reads its JSON array, and writes each element as a line in `output_path`.
    """
    records = []
    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith("__"):
            continue

        src = folder / input_name
        if not src.is_file():
            print(f"⏭  Skipping {folder.name!r} (no {input_name})")
            continue

        print(f"✔  Loading {src}")
        with src.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    records.extend(data)
                else:
                    print(f"⚠️  Expected a list in {src}, got {type(data)}")
            except json.JSONDecodeError as e:
                print(f"❌  JSON error in {src}: {e}")

    print(f"\n📝 Writing {len(records)} records to {output_path}")
    with output_path.open("w", encoding="utf-8") as out_f:
        for rec in records:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    base = Path(__file__).parent
    output = base / "data.jsonl"
    merge_json_to_jsonl(base_dir=base, output_path=output)

if __name__ == "__main__":
    main()