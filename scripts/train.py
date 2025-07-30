import os
import json
from time import time
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    logging, AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, TrainerCallback, BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from config.config import MODEL_NAME

DATA_PATH = "datasets/data.jsonl"
OUTPUT_BASE_DIR = Path(f"output/{MODEL_NAME}")
LORA_CONFIG_PATH = "config/lora_config.json"

def log(msg):
    print(f"\n🔧 {msg}\n{'=' * 60}")

def prepare_output_dir() -> Path:
    existing_dirs = [d for d in os.listdir(OUTPUT_BASE_DIR) if d.startswith("training-")]
    next_training_num = len(existing_dirs) + 1
    output_dir = OUTPUT_BASE_DIR / f"training-{next_training_num}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_and_prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.add_special_tokens({
    "additional_special_tokens": ["<think>", "</think>", "<output>", "</output>"],
    "bos_token": "<|im_start|>",
    "eos_token": "<|im_end|>",
    "pad_token": "<|im_end|>"  # optional, fallback
})

    print(tokenizer.chat_template)
    return tokenizer

def tokenize_function(example, tokenizer):
    prompt = example["question"]
    response = example["response"]

    assert "<think>" in response and "</think>" in response, "❌ Missing <think> tags"
    assert "<output>" in response and "</output>" in response, "❌ Missing <output> tags"

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]

    try:
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception as e:
        raise ValueError(f"Template rendering failed: {e}\nInput: {messages}")

    tokenized = tokenizer(chat_text, padding="max_length", max_length=2048, truncation=True)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def load_and_tokenize_dataset(tokenizer):
    assert os.path.exists(DATA_PATH), f"Data file not found at {DATA_PATH}"
    
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), remove_columns=["question", "response"])
    
    return dataset

def load_model_and_prepare_for_qora(tokenizer):
    start = time()

    log("Loading AutoConfig...")
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    log("Setting up BitsAndBytesConfig...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0,
    )

    log(f"Loading base model: {MODEL_NAME}")

    logging.set_verbosity_info()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
    )

    log("Resizing token embeddings...")
    model.resize_token_embeddings(len(tokenizer))

    log("Preparing model for QLoRA...")
    model = prepare_model_for_kbit_training(model)

    assert os.path.exists(LORA_CONFIG_PATH), "Missing LoRA config"
    
    log(f"Loading LoRA config from: {LORA_CONFIG_PATH}")
    with open(LORA_CONFIG_PATH) as f:
        lora_config = LoraConfig(**json.load(f))

    log("Wrapping model with LoRA adapter...")
    final_model = get_peft_model(model, lora_config)

    end = time()
    log(f"✅ Model loaded and prepared in {end - start:.2f} seconds")

    return final_model

def save_chat_jinja2(tokenizer, output_dir: Path):
    template_src = Path("templates/chat_template.jinja")
    assert template_src.exists(), f"Template missing at: {template_src}"

    # Load and assign template into tokenizer object
    with open(template_src, "r", encoding="utf-8") as f:
        template_text = f.read()

    tokenizer.chat_template = template_text

    # ✅ Ensure `chat_template` is saved inside tokenizer_config.json
    tokenizer.init_kwargs["chat_template"] = template_text

    # Save tokenizer (will now include chat_template in tokenizer_config.json)
    tokenizer.save_pretrained(output_dir)

    # Optional: Copy original jinja as reference
    template_dst = output_dir / "chat_template.jinja"
    template_dst.write_text(template_text, encoding="utf-8")

class EvalCallback(TrainerCallback):
    def __init__(self, tokenizer, interval):
        self.tokenizer = tokenizer
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval != 0:
            return

        prompt = [{"role": "user", "content": "What is the capital of France?"}]
        try:
            text = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"⚠️ Template failed: {e}")
            return

        inputs = self.tokenizer(text, return_tensors="pt").to(kwargs["model"].device)

        with torch.no_grad():
            output = kwargs["model"].generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            )

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=False)
        print(f"\n🧪 Eval @ step {state.global_step}:\n{decoded}\n")

def train_model(model, tokenizer, dataset, output_dir):
    log("Configuring training arguments...")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        max_steps=10,  # For quick test runs — change to num_train_epochs later
        # num_train_epochs=6,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=10,
        bf16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        disable_tqdm=False,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    log("Checking model parameters for meta device...")
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            raise RuntimeError(f"❌ Parameter {name} is still on meta device!")

    log("Disabling reentrant checkpointing...")
    torch.utils.checkpoint._use_reentrant = False

    log("Disabling use_cache for training...")
    model.config.use_cache = False

    log("Instantiating Trainer...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,  # Helps avoid tokenizer warnings
        callbacks=[EvalCallback(tokenizer, interval=20)]
    )

    save_chat_jinja2(tokenizer, output_dir)

    log("🔥 Starting training loop...")
    trainer.train()

    log("Saving final model and tokenizer...")
    model.save_pretrained(output_dir)

    log("Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)

    log("✅ Training complete.")

def test_training():
    from peft import PeftModel

    BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    ADAPTER_PATH = "output/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/training-5/checkpoint-10"

    log("Loading base model and tokenizer for testing...")
    
    # Load tokenizer from training dir — with special tokens and correct vocab size
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)

    # Make sure chat_template is explicitly loaded AND assigned
    chat_template_path = Path(ADAPTER_PATH).parent / "chat_template.jinja"
    assert chat_template_path.exists(), f"Template missing at: {chat_template_path}"

    with open(chat_template_path, "r", encoding="utf-8") as f:
        tokenizer.chat_template = f.read()

    # ✅ FORCE tokenizer config refresh to reflect it
    tokenizer.init_kwargs["chat_template"] = tokenizer.chat_template

    # Load base model (HuggingFace)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")

    # Resize embedding **before** applying adapter (to match trained dimensions)
    model.resize_token_embeddings(len(tokenizer))

    # Load adapter — now it works!
    model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=False)

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    print("🔍 Final input prompt:")
    print(prompt_text)
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=256, 
            do_sample=False,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
        )

    print(tokenizer.special_tokens_map)
    print(tokenizer.convert_tokens_to_ids(["<|im_start|>", "<think>", "<output>", "<|im_end|>"]))

    generated = tokenizer.decode(output[0], skip_special_tokens=False)
    # assert "<think>" in generated and "<output>" in generated, "Missing tags in generation"

    print("\nGenerated response:")
    print(generated)

def main():
    log("Preparing output directory")
    output_dir = prepare_output_dir()

    log("Loading tokenizer and adding special tags")
    tokenizer = load_and_prepare_tokenizer()

    log("Loading and tokenizing dataset")
    dataset = load_and_tokenize_dataset(tokenizer)

    log("Loading model and applying LoRA")
    model = load_model_and_prepare_for_qora(tokenizer)

    log("Training model")
    train_model(model, tokenizer, dataset, output_dir)

    log('Testing training with a small dataset')
    test_training()

if __name__ == "__main__":
    main()