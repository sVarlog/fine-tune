import torch
import os, json
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from datasets import load_dataset
from accelerate import init_empty_weights, infer_auto_device_map
from config.config import MODEL_NAME

def log(msg):
    print(f"\n🔧 {msg}\n{'='*60}")

DATA_PATH = "datasets/data.jsonl"
output_dir = Path(f"output/{MODEL_NAME}")

# Find the next available training directory
existing_dirs = [d for d in os.listdir(output_dir) if d.startswith("training-")]
next_training_num = len(existing_dirs) + 1
OUTPUT_DIR = output_dir / f"training-{next_training_num}/"

# Create the directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log("Loading tokenizer...")
# Tokenizer + Dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

log("Loading dataset...")
assert os.path.exists(DATA_PATH), f"Data file not found at {DATA_PATH}" # Ensure the data file exists
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

log("Tokenizing dataset...")
def tokenize(example):
    prompt = example["question"]
    response = example["response"]

    SYSTEM_PROMPT = """
    You are a structured assistant. Respond in exactly two parts using the following format:

    <think>
    [Your internal reasoning here]
    </think>
    <output>
    [Your final answer here]
    </output>
    """.strip()
    
    text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n" + \
           f"<|im_start|>user\n{prompt}<|im_end|>\n" + \
           f"<|im_start|>assistant\n{response}<|im_end|>\n"

    tokenized = tokenizer(text, padding="max_length", max_length=2048, truncation=True)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(tokenize, remove_columns=["question", "response"])

config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)

log("Loading model with low_cpu_mem_usage + auto device map...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    llm_int8_threshold=6.0,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    config=config,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    quantization_config=bnb_config,
)

# Prepare model for QLoRA
log("Preparing model for QLoRA...")
model = prepare_model_for_kbit_training(model)
assert os.path.exists("config/lora_config.json") # Ensure the LoRA config file exists
with open("config/lora_config.json") as f:
    lora_config = LoraConfig(**json.load(f))
model = get_peft_model(model, lora_config)

# Training setup
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,  # Can increase to 4 for 7B if memory allows
    gradient_accumulation_steps=16,  # Simulates 32 batch size
    num_train_epochs=6,  # Recommended: 3–5 epochs for small/medium datasets
    # max_steps=1000,  # Set max steps for faster testing
    learning_rate=2e-4,  # Higher is better with LoRA adapters + QLoRA (1e-4 or 2e-4 works well)
    warmup_ratio=0.05,  # Scales with dataset size; 5–6% is smoother than fixed steps
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    save_strategy="epoch",  # Save once per epoch (to avoid overhead)
    save_total_limit=2,  # Limit checkpoints
    bf16=True,  # RTX 5090 supports bf16 — use it
    optim="paged_adamw_8bit",  # 8-bit optimizer + memory efficiency
    lr_scheduler_type="cosine",
    report_to="none",
    disable_tqdm=False,
    gradient_checkpointing=True,  # Saves memory (~30–40% lower)
    ddp_find_unused_parameters=False,  # Fine for single-GPU
)

for name, param in model.named_parameters():
    if param.device.type == "meta":
        raise RuntimeError(f"Parameter {name} is still on meta device!")

import torch.utils.checkpoint
torch.utils.checkpoint._use_reentrant = False

# Start training
log("Starting training loop...")
model.config.use_cache = False
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
