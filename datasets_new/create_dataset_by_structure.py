import os
import json

STRUCTURE_FILE = "structure.enriched.json"
TOPICS_DIR = "topics"

def create_structure_from_json():
    # Load structure.enriched.json
    with open(STRUCTURE_FILE, "r", encoding="utf-8") as f:
        structure = json.load(f)

    # Ensure topics folder exists
    os.makedirs(TOPICS_DIR, exist_ok=True)

    for category, subcategories in structure.items():
        category_path = os.path.join(TOPICS_DIR, category)
        os.makedirs(category_path, exist_ok=True)

        for subcat, meta in subcategories.items():
            subcat_path = os.path.join(category_path, subcat)
            os.makedirs(subcat_path, exist_ok=True)

            content_types = meta.get("content_type", [])
            if isinstance(content_types, str):
                content_types = [content_types]

            for ctype in content_types:
                # Naming format: category.subcategory.contenttype.json
                filename = f"{category}.{subcat}.{ctype}.json"
                file_path = os.path.join(subcat_path, filename)

                if not os.path.exists(file_path):
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump([], f, ensure_ascii=False, indent=2)
                    print(f"Created: {file_path}")
                else:
                    print(f"Skipped existing: {file_path}")

if __name__ == "__main__":
    create_structure_from_json()