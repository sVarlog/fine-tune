from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path
import json, os
from config.config import BASE_MODEL_PATH, CHECKPOINT_STEP, MERGED_MODEL_PATH, ADAPTER_PATH, ALLOWED_KEYS

def log(title):
    print(f"\n🔧 {title}\n{'=' * 60}")

log("Cleaning adapter_config.json...")

config_path = ADAPTER_PATH / "adapter_config.json"
assert config_path.exists(), f"❌ adapter_config.json not found at {config_path}"
with open(config_path) as f:
    cfg = json.load(f)

cleaned_cfg = {k: v for k, v in cfg.items() if k in ALLOWED_KEYS}
with open(config_path, "w") as f:
    json.dump(cleaned_cfg, f, indent=2)

print("✅ adapter_config.json cleaned")

log("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

log("Loading adapter weights...")
model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))

log("Merging LoRA into base model...")
model = model.merge_and_unload()

log("Saving merged model...")
MERGED_MODEL_PATH.mkdir(parents=True, exist_ok=True)
model.save_pretrained(MERGED_MODEL_PATH)
AutoTokenizer.from_pretrained(BASE_MODEL_PATH).save_pretrained(MERGED_MODEL_PATH)
print(f"✅ Merged model saved to: {MERGED_MODEL_PATH.resolve()}")
