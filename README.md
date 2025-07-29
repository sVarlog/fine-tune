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
data/
├── train.jsonl           # Training dataset
├── fine_tune_examples_chunk1_250.jsonl # Example dataset
merged-models/
├── deepseek-merged/      # Directory for merged models
output/
├── deepseek-ai/          # Directory for training outputs
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

Paths like `BASE_MODEL_PATH`, `ADAPTER_PATH`, and `MERGED_MODEL_PATH` are managed in `config/config.py`. Update `config.py` to modify these paths.

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

### Volumes

-   **Model Volume**: Ensure the Hugging Face cache directory is mounted as a volume. Example:
    ```yaml
    volumes:
        - C:/Users/pc/.cache/huggingface:/root/.cache/huggingface
        - ./:/workspace
    ```

### What to Do If There's No Volume

If the model volume is not mounted:

1. Verify the `docker-compose.yml` file includes the correct volume mapping.
2. Check Docker Desktop settings to ensure the drive containing the cache directory is shared.
3. If the model is missing, download it manually using Hugging Face's `transformers` library:

    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from config.config import BASE_MODEL_PATH

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    ```

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
