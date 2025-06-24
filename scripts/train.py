import torch
import os, json
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from datasets import load_dataset
from accelerate import init_empty_weights, infer_auto_device_map

def log(msg):
    print(f"\n🔧 {msg}\n{'='*60}")

# Paths
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" 
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 
DATA_PATH = "data/train.jsonl"
OUTPUT_DIR = f"output/{MODEL_NAME}/"

log("Loading tokenizer...")
# Tokenizer + Dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

log("Loading dataset...")
assert os.path.exists(DATA_PATH), f"Data file not found at {DATA_PATH}" # Ensure the data file exists
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

log("Tokenizing dataset...")
def tokenize(example):
    messages = example["messages"]
    text = "".join([f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages])
    tokenized = tokenizer(text, padding="max_length", max_length=2048, truncation=True)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(tokenize, remove_columns=["messages"])

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
    label_names=["labels"],
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=10,
    max_steps=100,
    learning_rate=2e-4,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    bf16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    report_to="none",                # ⬅️ disable wandb or other integrations
    disable_tqdm=False,              # ⬅️ ensure progress bar is visible
    gradient_checkpointing=True,
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
