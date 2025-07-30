# 🧠 LLM Fine-Tuning, Merging, and GGUF Conversion

This project allows you to fine-tune LLMs like DeepSeek/Qwen using QLoRA, merge LoRA adapters into base models, and convert the merged model to GGUF format for use with llama.cpp.

---

## 📁 Project Structure

```
scripts/
├── train.py              # Fine-tune using QLoRA
├── merge_adapter.py      # Merge LoRA into base model
├── convert_to_gguf.sh    # Convert to GGUF format
tools/llama/              # Local copy of llama.cpp Python
├── convert_hf_to_gguf_update.py
├── convert_llama_ggml_to_gguf.py
├── convert_lora_to_gguf.py
├── transformers-to-gguf.py
config/
├── config.py             # Centralized configuration for paths and settings
├── lora_config.json      # Configuration for LoRA fine-tuning
datasets/
├── ai/
│   ├── questions.json
│   └── dataset.jsonl
├── business/
├── finance/
├── ethics/
├── ...                   # Other domains
├── data.jsonl            # Combined dataset built from all folders
├── build_dataset.py      # Merges all datasets dynamically
merged-models/
├── deepseek-merged/      # Directory for merged models
output/
├── deepseek-ai/          # Directory for training outputs
```

---

## 🚀 Usage

### 1. Build the training dataset

```bash
python datasets/build_dataset.py
```

This will merge all `dataset.jsonl` files from domain folders into `datasets/data.jsonl`.

### 2. Train the model

```bash
docker-compose up trainer
```

Ensure `datasets/data.jsonl` and `config/lora_config.json` exist.

### 3. Merge adapter with base model

```bash
python scripts/merge_adapter.py
```

Paths like `BASE_MODEL_PATH`, `ADAPTER_PATH`, and `MERGED_MODEL_PATH` are managed in `config/config.py`.

### 4. Convert to GGUF

```bash
bash scripts/convert_to_gguf.sh
```

Output file will be saved to the latest merging directory:

```
merged-models/deepseek-merged/merging-n/gguf-output/deepseek-q4.gguf
```

If a file with the same name already exists, a unique suffix will be added to the filename to avoid overwriting.

---

## 👁️ Dataset Format

Each training sample follows a strict reasoning-output structure:

```json
{
    "question": "Why is transparency important in AI?",
    "response": "<think>Transparency helps detect bias, improve trust, and enable accountability...</think><output>Transparency is key to ethical and trustworthy AI systems.</output>"
}
```

-   `<think>` contains structured reasoning
-   `<output>` gives a short, clear final answer

This format enforces consistency and helps reduce hallucinations. It is designed for use in structured UI output and instruction-following agents.

### Domains Covered

-   AI
-   Business
-   Finance
-   Ethics
-   Global Trends
-   Marketing
-   Productivity
-   Psychology
-   Strategy
-   Tech

Each folder contains:

-   `questions.json` - raw questions
-   `dataset.jsonl` - generated training pairs

---

## 🛠️ Docker

This project uses `nvidia/cuda` and supports training inside Docker with GPU acceleration via Docker Compose.

### Volumes

Ensure the Hugging Face cache and project code are mounted correctly:

```yaml
volumes:
    - C:/Users/pc/.cache/huggingface:/root/.cache/huggingface
    - ./:/workspace
```

### If Model Is Not Found:

1. Check that `docker-compose.yml` maps your cache directory.
2. Share the host drive in Docker Desktop settings.
3. Manually download the model using:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.config import BASE_MODEL_PATH

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
```

---

## 📆 Recommended Fine-Tuning Config (RTX 5090)

| Model Size | Epochs | Batch Size | Accum. Steps | Learning Rate | Notes                      |
| ---------- | ------ | ---------- | ------------ | ------------- | -------------------------- |
| 7B         | 3-5    | 2-4        | 16           | 1e-4 – 2e-4   | Fast, fits with QLoRA      |
| 32B        | 2-4    | 1-2        | 16           | 1e-4          | Use `bf16` + checkpointing |

Make sure to monitor GPU memory and training loss with long context windows.

---

## 📆 Requirements (if running outside Docker)

```bash
pip install -r requirements.txt
```

If using local `transformers-to-gguf.py`, install:

```bash
pip install ./tools/llama/gguf-py
```

---

## ✨ Contributing

Feel free to extend domain coverage or add new response formats. This repo is designed to be modular and extensible for future agent-like assistants.

---

## 🎉 Results

Trained models show reduced hallucinations and maintain structure across multiple domains with `<think>` and `<output>` response blocks.

For structured UI generation, this format helps ensure alignment and response reliability.

---
