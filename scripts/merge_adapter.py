from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path
import json, os

def log(title):
    print(f"\n🔧 {title}\n{'=' * 60}")

base_model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
adapter_path = Path(f"output/{base_model_path}/checkpoint-100")
merged_path = Path("merged-models/deepseek-merged")

log("Cleaning adapter_config.json...")

config_path = adapter_path / "adapter_config.json"
assert config_path.exists(), f"❌ adapter_config.json not found at {config_path}"
with open(config_path) as f:
    cfg = json.load(f)

allowed_keys = {
    "peft_type", "base_model_name_or_path", "inference_mode", "r",
    "lora_alpha", "lora_dropout", "bias", "target_modules", "task_type",
    "modules_to_save", "rank_pattern", "alpha_pattern", "fan_in_fan_out",
    "init_lora_weights", "layers_to_transform", "layers_pattern",
    "auto_mapping", "revision", "use_dora", "use_rslora"
}

cleaned_cfg = {k: v for k, v in cfg.items() if k in allowed_keys}
with open(config_path, "w") as f:
    json.dump(cleaned_cfg, f, indent=2)

print("✅ adapter_config.json cleaned")

log("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)

log("Loading adapter weights...")
model = PeftModel.from_pretrained(base_model, str(adapter_path))

log("Merging LoRA into base model...")
model = model.merge_and_unload()

log("Saving merged model...")
merged_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(merged_path)
AutoTokenizer.from_pretrained(base_model_path).save_pretrained(merged_path)
print(f"✅ Merged model saved to: {merged_path.resolve()}")
