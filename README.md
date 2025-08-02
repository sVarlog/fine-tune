# 🧠 LLM Fine-Tuning, Merging & GGUF Conversion

This repo shows how to:

1. **Fine-tune** a base LLM (e.g. Qwen/DeepSeek) with QLoRA
2. **Merge** the resulting LoRA adapter into the base model
3. **Convert** the merged model into GGUF (for `llama.cpp` / local inference)

---

## 📂 Project Layout

```
.
├── config/
│   ├── config.py            # Central paths & constants
│   └── lora_config.json     # LoRA hyperparameters
├── datasets/
│   ├── ai/
│   │   └── dataset.jsonl
│   ├── build_dataset.py     # Collate domain files into data.jsonl
│   └── data.jsonl           # Combined training data
├── merged-models/
│   └── deepseek-merged/     # Merged model outputs
├── output/
│   └── deepseek-ai/         # QLoRA training runs
├── scripts/
│   ├── train.py             # QLoRA fine-tuning
│   ├── merge_adapter.py     # Merge adapter → base model
│   └── convert_to_gguf.sh   # GGUF conversion wrapper
└── tools/
    └── llama/               # `transformers-to-gguf.py` & helpers
```

---

## 🚀 Quickstart

### 1. Build your dataset

```bash
python datasets/build_dataset.py
```

This pulls in every `dataset.jsonl` under `datasets/*` and writes `datasets/data.jsonl`.

### 2. Train with QLoRA

```bash
python scripts/train.py
```

Outputs checkpoints under `output/deepseek-ai/TRAINING-N/checkpoint-M/`.  
Special/chat tokens, `tokenizer.json`, `vocab.json`, `merges.txt`, and your `chat_template.jinja` are saved there.

### 3. Merge LoRA into the base

```bash
python scripts/merge_adapter.py
```

-   Picks the **last** `training-*` / `checkpoint-*`
-   Reads the adapter’s added embedding rows (via `adapter_model.safetensors`)
-   Resizes the HF base model to match
-   Merges & unloads LoRA weights
-   Saves under `merged-models/deepseek-merged/merging-K/`
-   Copies across your **full** trained-tokenizer artifacts:
    -   `tokenizer.json`
    -   `vocab.json`
    -   `merges.txt`
    -   `special_tokens_map.json`
    -   `chat_template.jinja`

### 4. Convert to GGUF

```bash
bash scripts/convert_to_gguf.sh --outtype q8_0
```

-   Locates the latest `merged-models/.../merging-K/`
-   Runs `transformers-to-gguf.py` → emits `*.gguf` in `merging-K/gguf-output/`

---

## 📝 Why copy _all_ tokenizer files?

When you added custom special/chat tokens and a Jinja template:

-   **`tokenizer.json`** holds your merges + special tokens + chat_template
-   **`vocab.json`** + **`merges.txt`** define your BPE vocabulary
-   **`special_tokens_map.json`** maps names → IDs
-   **`chat_template.jinja`** is your prompt-format template

By shipping them alongside the merged model, you preserve _exactly_ the same tokenization and chat layout your fine-tune used.

---

## 🛠 Fine-Tuning Tips

-   Use small batches (2–4) with gradient accumulation 16–32
-   Train for 3–5 epochs on ~2–3K samples to start
-   Monitor loss & generations via the built-in eval callback

---

## 🎉 Results

-   Adapter merging “just worked” once we resized embeddings and carried over the custom tokenizer.
-   Downstream GGUF conversion now sees the proper `tokenizer.model` alongside JSON/BPE files.

---
