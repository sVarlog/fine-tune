import os
import json
from time import time
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    logging, AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, TrainerCallback, BitsAndBytesConfig,
    DataCollatorForSeq2Seq
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

    assert "<think>" in response and "</think>" in response
    assert "<output>" in response and "</output>" in response

    messages = [
        {"role": "system", "content": "You are a structured assistant. Respond in exactly two parts using the format:\n<think>...</think>\n<output>...</output>"},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]

    _, tokenized = format_and_tokenize(messages, tokenizer)

    # Everything is list[int], safe to copy
    tokenized["labels"] = [
        tok_id if tok_id != tokenizer.pad_token_id else -100
        for tok_id in tokenized["input_ids"]
    ]

    return tokenized

def format_and_tokenize(messages, tokenizer, return_tensors=False, add_generation_prompt=False):
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt
    )
    if return_tensors:
        tokenized = tokenizer(formatted_text, return_tensors="pt", add_special_tokens=False)
    else:
        tokenized = tokenizer(
            formatted_text,
            padding="longest",         # or "longest" for dynamic batch-based
            truncation=True,
            max_length=2048,              # pick appropriate max length
            return_tensors=None,          # ✅ Don't return tensor now
            add_special_tokens=False,
        )
    return formatted_text, tokenized

def load_and_tokenize_dataset(tokenizer):
    assert os.path.exists(DATA_PATH), f"Data file not found at {DATA_PATH}"
    
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), remove_columns=["question", "response"])
    print(tokenizer.chat_template[:200])

    print("---------------")

    # Print some debug info
    print(tokenizer.decode(dataset[0]["input_ids"]))

    print("---------------")
    
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

    with open(template_src, "r", encoding="utf-8") as f:
        template_text = f.read()

    print("TEMPLATE TEXT:")
    print(template_text)

    tokenizer.chat_template = template_text

    # ✅ Directly update tokenizer config dict
    tokenizer_config_path = output_dir / "tokenizer_config.json"
    tokenizer.save_pretrained(output_dir)

    # Reopen and patch the file directly after saving
    if tokenizer_config_path.exists():
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        config["chat_template"] = template_text

        with open(tokenizer_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    else:
        raise FileNotFoundError(f"Expected {tokenizer_config_path} not found!")

def is_structured_output(text: str) -> bool:
    return all(tag in text for tag in ["<think>", "</think>", "<output>", "</output>"])

class EvalCallback(TrainerCallback):
    def __init__(self, tokenizer, interval):
        self.tokenizer = tokenizer
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval != 0:
            return

        messages = [
            {"role": "system", "content": "You are a structured assistant. Respond in exactly two parts using the format:\n<think>[Your reasoning]</think>\n<output>[Your answer]</output>"},
            {"role": "user", "content": "What is the capital of France?"}
        ]

        prompt_text, tokenized = format_and_tokenize(messages, self.tokenizer)

        if not prompt_text.strip().endswith("<|im_start|>assistant") and "<|im_start|>assistant\n" not in prompt_text:
            print("⚠️ Generation prompt is not correctly appended.")

        inputs = {k: v.to(kwargs["model"].device) for k, v in tokenized.items()}

        with torch.no_grad():
            output = kwargs["model"].generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                top_k=1,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            )

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=False)
        print(f"\n🧪 Eval @ step {state.global_step}:\n{decoded}\n")
        print("Is structured output:", is_structured_output(decoded))

def train_model(model, tokenizer, dataset, output_dir):
    for row in dataset:
	    print(tokenizer.decode(row["input_ids"]))

    
    print(len(dataset))
    # log("Configuring training arguments...")

    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     per_device_train_batch_size=2,
    #     gradient_accumulation_steps=16,
    #     # max_steps=20,  # For quick test runs — change to num_train_epochs later
    #     num_train_epochs=8,
    #     learning_rate=2e-4,
    #     warmup_ratio=0.05,
    #     logging_dir=f"{output_dir}/logs",
    #     logging_steps=10,
    #     save_strategy="epoch",
    #     save_total_limit=10,
    #     bf16=True,
    #     optim="paged_adamw_8bit",
    #     lr_scheduler_type="cosine",
    #     report_to="none",
    #     disable_tqdm=False,
    #     gradient_checkpointing=True,
    #     ddp_find_unused_parameters=False,
    # )

    # log("Checking model parameters for meta device...")
    # for name, param in model.named_parameters():
    #     if param.device.type == "meta":
    #         raise RuntimeError(f"❌ Parameter {name} is still on meta device!")

    # log("Disabling reentrant checkpointing...")
    # torch.utils.checkpoint._use_reentrant = False

    # log("Disabling use_cache for training...")
    # model.config.use_cache = False

    # log("Instantiating Trainer...")

    # data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, label_pad_token_id=-100)

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset,
    #     tokenizer=tokenizer,  # Helps avoid tokenizer warnings
    #     callbacks=[EvalCallback(tokenizer, interval=20)],
    #     data_collator=data_collator,
    # )

    # log("🔥 Starting training loop...")
    # trainer.train()

    # log("Saving final model and tokenizer...")
    # model.save_pretrained(output_dir)

    # log("Saving tokenizer...")
    # tokenizer.save_pretrained(output_dir)

    # log("✅ Training complete.")

def test_training():
    from peft import PeftModel

    BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    ADAPTER_PATH = "output/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/training-3/checkpoint-352"

    log("Loading base model and tokenizer for testing...")

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    chat_template_path = Path(ADAPTER_PATH).parent / "chat_template.jinja"
    assert chat_template_path.exists(), f"Template missing at: {chat_template_path}"
    tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
    tokenizer.init_kwargs["chat_template"] = tokenizer.chat_template

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=False)

    messages = [
        {"role": "system", "content": "You are a structured assistant. Respond in exactly two parts using the format:\n<think>...</think>\n<output>...</output>"},
        {"role": "user", "content": "What is the capital of France?"}
    ]

    prompt_text, tokenized = format_and_tokenize(messages, tokenizer, return_tensors=True, add_generation_prompt=True)

    print("🔍 Final input prompt:")
    print(prompt_text)

    assert prompt_text.strip().endswith("<|im_start|>assistant"), "Missing assistant prompt trigger"

    inputs = {k: v.to(model.device) for k, v in tokenized.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            top_k=1,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
        )

    print(tokenizer.special_tokens_map)
    print(tokenizer.convert_tokens_to_ids(["<|im_start|>", "<think>", "<output>", "<|im_end|>"]))

    generated = tokenizer.decode(output[0], skip_special_tokens=False)

    print("Is structured output:", is_structured_output(generated))

    print("\nGenerated response:")
    print(generated)
    
def main():
    log("Preparing output directory")
    output_dir = prepare_output_dir()

    log("Loading tokenizer and adding special tags")
    tokenizer = load_and_prepare_tokenizer()

    log("Saving chat template to tokenizer")
    save_chat_jinja2(tokenizer, output_dir)

    log("Loading and tokenizing dataset")
    dataset = load_and_tokenize_dataset(tokenizer)

    log("Loading model and applying LoRA")
    model = load_model_and_prepare_for_qora(tokenizer)

    print("=== Final Chat Template ===")
    print(tokenizer.chat_template)
    print("===========================")

    log("Training model")
    train_model(model, tokenizer, dataset, output_dir)

    # log('Testing training with a small dataset')
    # test_training()

if __name__ == "__main__":
    main()