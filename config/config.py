from pathlib import Path

# Base model name
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" 
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# Base model path
BASE_MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Adapter path (update checkpoint step as needed)
CHECKPOINT_STEP = 80
ADAPTER_PATH = Path(f"output/{BASE_MODEL_PATH}/checkpoint-{CHECKPOINT_STEP}")

# Merged model path
MERGED_MODEL_PATH = Path("merged-models/deepseek-merged")

# Allowed keys for adapter configuration cleaning
ALLOWED_KEYS = {
    "peft_type", "base_model_name_or_path", "inference_mode", "r",
    "lora_alpha", "lora_dropout", "bias", "target_modules", "task_type",
    "modules_to_save", "rank_pattern", "alpha_pattern", "fan_in_fan_out",
    "init_lora_weights", "layers_to_transform", "layers_pattern",
    "auto_mapping", "revision", "use_dora", "use_rslora"
}
