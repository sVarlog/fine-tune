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
```

---

## 🚀 Usage

### 1. Train the model

```bash
docker-compose run trainer
```

Ensure `data/train.jsonl` and `config/lora_config.json` exist.

### 2. Merge adapter with base model

```bash
python scripts/merge_adapter.py
```

### 3. Convert to GGUF

```bash
bash scripts/convert_to_gguf.sh
```

Output file will be saved to:

```
merged-models/deepseek-merged/gguf-output/deepseek-q4.gguf
```

---

## 🐳 Docker

This project uses `nvidia/cuda` and supports training inside Docker with GPU acceleration via Docker Compose. The container shares Hugging Face cache and project code from the host.

---

## 📦 Requirements (outside Docker)

If you're running locally, install:

```bash
pip install -r requirements.txt
```

If you use the local version of `transformers-to-gguf.py`, also install:

```bash
pip install ./tools/llama/gguf-py
```
