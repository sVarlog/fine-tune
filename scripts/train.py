import torch
import os, json
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from datasets import load_dataset
from accelerate import init_empty_weights, infer_auto_device_map
from config.config import MODEL_NAME

def log(msg):
    print(f"\n🔧 {msg}\n{'='*60}")

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
    output_dir=OUTPUT_DIR,  # Directory to save model checkpoints and logs
    per_device_train_batch_size=2,  # Increase batch size for better training
    gradient_accumulation_steps=16,  # Increase accumulation steps to simulate larger batch sizes
    warmup_steps=200,  # Increase warmup steps for smoother learning rate adjustment
    max_steps=1000,  # Increase total steps for more effective training
    learning_rate=1e-4,  # Lower learning rate for stable convergence
    logging_dir=f"{OUTPUT_DIR}/logs",  # Directory for logging training metrics
    logging_steps=20,  # Log metrics less frequently for better monitoring
    save_steps=100,  # Save checkpoints less frequently to avoid interruptions
    save_total_limit=5,  # Keep up to 5 checkpoints to save disk space
    bf16=True,  # Use bfloat16 for faster computation and lower memory usage
    optim="paged_adamw_8bit",  # Use 8-bit optimizer for memory efficiency
    lr_scheduler_type="cosine",  # Use cosine scheduler for smooth learning rate decay
    report_to="none",  # Disable external reporting integrations
    disable_tqdm=False,  # Keep progress bar visible
    gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
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
