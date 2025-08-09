# Datasets: Structure and Scripts (`datasets_new`)

This folder contains all resources, scripts, and schemas for building, structuring, and exporting training datasets for fine-tuning language models. Below is a detailed guide to the directory's structure, its main files, and how to use them.

---

## Directory Structure

```
datasets_new/
├── build_train_jsonl.py
├── create_dataset_by_structure.py
├── structure.enriched.json
├── train_data.jsonl
├── schemas/
│   ├── schema_qa.json
│   └── schema_reasoning.json
└── topics/
    ├── ai/
    ├── business/
    ├── ethics/
    ├── finance/
    ├── geography/
    ├── marketing/
    ...
```

---

## Main Files and Their Roles

### 1. `structure.enriched.json`

-   **Purpose:**
    -   Defines the full taxonomy of dataset topics, subcategories, and their metadata.
    -   Each top-level key is a category (e.g., `ai`, `business`, `ethics`), containing subcategories with descriptions, example questions, content types, and tags.
-   **Usage:**
    -   Used by scripts to determine which topics and subcategories to process, and to enrich each data sample with consistent metadata.
-   **Example Entry:**
    ```json
    "ai": {
      "business_applications": {
        "description": "AI applied to business operations, decision-making, automation, and ROI.",
        "example_questions": [ ... ],
        "content_type": ["qa", "reasoning"],
        "tags": ["automation", "ops", ...]
      },
      ...
    }
    ```

### 2. `schemas/`

-   **Purpose:**
    -   Contains JSON schema files that define the expected structure for different types of dataset entries.
    -   `schema_qa.json`: Schema for question-answer pairs.
    -   `schema_reasoning.json`: Schema for reasoning-based samples.
-   **Usage:**
    -   Use these schemas to validate dataset files or as a reference for formatting new data.

### 3. `topics/`

-   **Purpose:**
    -   Contains all raw data files, organized by category and subcategory.
    -   Each subcategory folder contains files named as `{category}.{subcategory}.{content_type}.json` (e.g., `ai.business_applications.qa.json`).
-   **Usage:**
    -   These files are the source data for dataset building scripts.

### 4. `build_train_jsonl.py`

-   **Purpose:**
    -   Main script to flatten and export all topic data into a single JSONL file (`train_data.jsonl`), with per-sample metadata.
-   **How it works:**
    -   Loads the structure from `structure.enriched.json`.
    -   Iterates over all categories, subcategories, and content types.
    -   Reads each corresponding topic file, normalizes entries, and enriches them with metadata (category, subcategory, tags, etc.).
    -   Ensures unique IDs and consistent fields.
    -   Outputs a single `train_data.jsonl` file, ready for training or further processing.
-   **Usage:**
    ```sh
    python build_train_jsonl.py --structure structure.enriched.json --topics-dir topics --output train_data.jsonl
    ```
    -   All arguments are optional; defaults are set for typical usage.

### 5. `create_dataset_by_structure.py`

-   **Purpose:**
    -   (If present) Used for generating or organizing topic files according to the structure defined in `structure.enriched.json`.
    -   May help automate the creation of empty or template files for new topics/subcategories.
-   **Usage:**
    -   Refer to the script's help or source for details.

### 6. `train_data.jsonl`

-   **Purpose:**
    -   The final, flattened dataset file produced by `build_train_jsonl.py`.
    -   Each line is a JSON object with all required fields and metadata.

---

## Workflow: Building a Training Dataset

1. **Edit or add topic data:**
    - Place new or updated data files in the appropriate `topics/{category}/{subcategory}/` folder, following the naming convention.
2. **Update structure:**
    - If adding new topics or subcategories, update `structure.enriched.json` accordingly.
3. **Validate data (optional):**
    - Use the schemas in `schemas/` to check your data files for correctness.
4. **Build the dataset:**
    - Run `build_train_jsonl.py` to generate `train_data.jsonl`.
5. **Use `train_data.jsonl` for training or further processing.**

---

## Tips & Best Practices

-   Always keep `structure.enriched.json` in sync with the actual topics and subcategories present in `topics/`.
-   Use the schemas to ensure data consistency and avoid errors during export.
-   Each data entry should have at least a question and an output/answer; optional fields like `think` (rationale) are supported.
-   Tags and metadata are automatically merged and deduplicated by the export script.

---

## See Also

-   For more details on the data format, see the schema files in `schemas/`.
-   For advanced usage or troubleshooting, read the source code of the scripts or use the `--help` flag.
