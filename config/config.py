from pathlib import Path
import os

# Base model name
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" 
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 
# MODEL_NAME = "deepseek-ai/DeepSeek-LLM-7B-Base" 
# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Base model path
BASE_MODEL_PATH = MODEL_NAME

# Dynamically find the latest checkpoint
output_dir = Path(f"output/{BASE_MODEL_PATH}")
print(f"Checking directory: {output_dir}")  # Debugging output

if not output_dir.exists():
    print(f"Directory does not exist: {output_dir}. Creating it...")  # Debugging output
    output_dir.mkdir(parents=True, exist_ok=True)

if output_dir.exists():
    print(f"Files in directory: {[file.name for file in output_dir.iterdir()]}")  # Debugging output
    training_dirs = [step for step in output_dir.iterdir() if step.is_dir() and step.name.startswith("training-")]

    if training_dirs:
        latest_training_dir = max(training_dirs, key=lambda path: int(path.name.split('-')[1]))
        print(f"Latest training directory found: {latest_training_dir}")  # Debugging output
        checkpoints = [step for step in latest_training_dir.iterdir() if step.name.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda path: int(path.name.split('-')[1]))
            ADAPTER_PATH = latest_checkpoint
            print(f"Latest checkpoint found: {ADAPTER_PATH}")  # Debugging output
        else:
            print(f"No checkpoints found in {latest_training_dir}. Creating a base checkpoint...")
            (latest_training_dir / "checkpoint-1").mkdir(parents=True, exist_ok=True)
    else:
        print(f"No training directories found in {output_dir}. Creating a base training directory...")
        base_training_dir = output_dir / "training-1"
        base_training_dir.mkdir(parents=True, exist_ok=True)
        (base_training_dir / "checkpoint-1").mkdir(parents=True, exist_ok=True)
else:
    raise FileNotFoundError(f"Directory does not exist: {output_dir}")

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
