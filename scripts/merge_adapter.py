import json
import shutil
from pathlib import Path

import safetensors.torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ───────────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────────
BASE_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ADAPTER_ROOT    = Path("output") / BASE_MODEL_NAME
OUTPUT_ROOT     = Path("merged-models") / "deepseek-merged"

# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────
def log(msg):
    print(f"\n🔧 {msg}\n{'=' * 60}")

def find_latest_checkpoint(root: Path) -> Path:
    """Return the path to the last ‘checkpoint-N’ under the last ‘training-M’ dir."""
    trainings = sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.startswith("training-")],
        key=lambda d: int(d.name.split("-", 1)[1])
    )
    if not trainings:
        raise FileNotFoundError(f"No training-* dirs under {root}")
    last_training = trainings[-1]

    checkpoints = sorted(
        [d for d in last_training.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-", 1)[1])
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* dirs under {last_training}")
    return checkpoints[-1]

def get_adapter_vocab_size(checkpoint_dir: Path) -> int:
    """Peek into the LoRA weights to infer how many embedding rows were added."""
    weights = safetensors.torch.load_file(
        str(checkpoint_dir / "adapter_model.safetensors"), device="cpu"
    )
    return weights["base_model.model.model.embed_tokens.weight"].shape[0]

def prepare_output_dir(base: Path) -> Path:
    """Make a new merging-N folder under OUTPUT_ROOT and return its path."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = len(list(OUTPUT_ROOT.glob("merging-*"))) + 1
    out_dir = OUTPUT_ROOT / f"merging-{run_id}"
    out_dir.mkdir()
    return out_dir

# ───────────────────────────────────────────────────────────────────────────────
# Main flow
# ───────────────────────────────────────────────────────────────────────────────
def main():
    log(f"Finding latest checkpoint in {ADAPTER_ROOT}")
    ckpt = find_latest_checkpoint(ADAPTER_ROOT)

    log(f"Loading base model `{BASE_MODEL_NAME}`")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
        device_map="auto"
    )

    log("Inspecting adapter vocabulary size")
    vocab_size = get_adapter_vocab_size(ckpt)
    log(f"Adapter added {vocab_size:,} token embeddings")

    log("Resizing base model embeddings")
    base.resize_token_embeddings(vocab_size)

    log("Loading & merging LoRA adapter")
    peft_model = PeftModel.from_pretrained(base, str(ckpt), is_trainable=False)
    merged     = peft_model.merge_and_unload()

    out_dir = prepare_output_dir(OUTPUT_ROOT)
    log(f"Saving merged model to {out_dir}")
    merged.save_pretrained(out_dir)

    log("Saving fresh base tokenizer")
    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
        use_fast=True
    )
    tok.save_pretrained(out_dir)

    log("Copying training-time tokenizer artifacts")
    for fn in ("vocab.json", "merges.txt", "special_tokens_map.json", "chat_template.jinja"):
        src = ckpt.parent / fn
        if src.exists():
            shutil.copy(src, out_dir / fn)
            log(f"  • {fn}")

    print(f"\n✅ Done! Merged model + tokenizer ready at: {out_dir}")

if __name__ == "__main__":
    main()