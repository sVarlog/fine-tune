import re
import os
import json
import shutil
from time import time
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    logging, AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, TrainerCallback, BitsAndBytesConfig,
)
from tokenizers import Tokenizer

from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from config.config import MODEL_NAME

SYSTEM_PROMPT = "You are a structured assistant. Respond in exactly two parts using the format:\n<think>[Your reasoning]</think>\n<output>[Your answer]</output>"

DATA_PATH = "datasets/data.jsonl"
OUTPUT_BASE_DIR = Path(f"output/{MODEL_NAME}")
LORA_CONFIG_PATH = "config/lora_config.json"


class SFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return (loss, outputs) if return_outputs else loss

def run_generation_and_print(model, tokenizer, messages, label=None, return_response=False):
    from transformers import StoppingCriteriaList, StoppingCriteria

    # 1) Define our criteria
    class MaxNewTokensCriteria(StoppingCriteria):
        def __init__(self, start_length, max_new_tokens):
            super().__init__()
            self.start_length = start_length
            self.max_new_tokens = max_new_tokens
        def __call__(self, input_ids, scores, **kwargs):
            return input_ids.shape[1] - self.start_length >= self.max_new_tokens

    class StopSequenceCriteria(StoppingCriteria):
        def __init__(self, stop_sequences_ids):
            super().__init__()
            # ensure list of lists
            self.stop_sequences_ids = [
                seq if isinstance(seq, (list, tuple)) else [seq]
                for seq in stop_sequences_ids
            ]
        def __call__(self, input_ids, scores, **kwargs):
            last_tokens = input_ids[0].tolist()
            for seq_ids in self.stop_sequences_ids:
                if seq_ids and last_tokens[-len(seq_ids):] == seq_ids:
                    return True
            return False

    # 2) Build the prompt
    prompt_text, tokenized = format_and_tokenize(
        messages, tokenizer, return_tensors=True, add_generation_prompt=True
    )
    if not prompt_text.strip().endswith("<|im_start|><|assistant|>"):
        print("⚠️ Generation prompt is not correctly appended.")

    # 3) Move inputs to device & compute prompt_length
    inputs = {k: v.to(model.device) for k, v in tokenized.items()}


    # 4) Assemble stoppers
    stop_sequences = [tokenizer.encode("</output>", add_special_tokens=False)]

    # 5) Generate with explicit overrides
    stopping_criteria = StoppingCriteriaList([
        MaxNewTokensCriteria(inputs["input_ids"].shape[1], 200),
        StopSequenceCriteria(stop_sequences)
    ])
    # output = model.generate(
    #     **inputs,
    #     max_new_tokens=200,
    #     temperature=0.7,
    #     top_p=0.9,
    #     pad_token_id=tokenizer.pad_token_id,
    #     eos_token_id=tokenizer.eos_token_id,  # Pass a single int
    #     do_sample=True,
    #     stopping_criteria=stopping_criteria,
    # )
    output = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,          # <— turn sampling OFF
        temperature=0.0,          # <— doesn’t matter when do_sample=False
        num_beams=1,              # <— greedy
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    print("GREEDY:", tokenizer.decode(output[0], skip_special_tokens=False))

    # 6) Decode & print
    decoded = tokenizer.decode(output[0], skip_special_tokens=False)
    header = f"\n🧪 {label}:\n" if label else "\n🧪 Generation:\n"
    out_str = header + decoded + "\n"
    print(header + decoded + "\n")
    print("Is structured output:", is_structured_output(decoded))
    print("-" * 60)

    if return_response:
        return out_str

def check_lora_modules(model, lora_config_path: str):
    """
    Load the LoRA config from disk, then verify which target_modules
    names actually appear in model.named_modules(), printing a summary.
    """
    # 1) Load your LoRA config
    with open(lora_config_path, "r") as f:
        lora_cfg = LoraConfig(**json.load(f))

    # 2) Gather all module names
    all_module_names = [name for name, _ in model.named_modules()]

    # 3) Check each target
    found, missing = [], []
    print("\n🔍 Checking LoRA target modules against the model…")
    for target in lora_cfg.target_modules:
        matches = [mn for mn in all_module_names if target in mn]
        if matches:
            found.append(target)
            snippet = matches[:3] + (["…"] if len(matches) > 3 else [])
            print(f"  ✔ `{target}` matched in: {snippet}")
        else:
            missing.append(target)
            print(f"  ❌ `{target}` NOT found in model modules!")
    print(f"\n✅ Modules to be LoRA‐tuned : {found}")
    if missing:
        print(f"⚠️ Warning: these targets were missing and will be skipped: {missing}")
    print("==============================================\n")

    # Return the loaded config so you can immediately use it
    return lora_cfg

def log(msg):
    print(f"\n🔧 {msg}\n{'=' * 60}")

def prepare_output_dir() -> Path:
    existing_dirs = [d for d in os.listdir(OUTPUT_BASE_DIR) if d.startswith("training-")]
    next_training_num = len(existing_dirs) + 1
    output_dir = OUTPUT_BASE_DIR / f"training-{next_training_num}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_and_prepare_tokenizer(output_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        add_bos_token=True,
        add_eos_token=False  # We'll handle this manually
    )
    
    # Explicitly add special tokens with proper normalization
    special_tokens_dict = {
        "additional_special_tokens": [
            "<think>", "</think>", 
            "<output>", "</output>"
        ],
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|pad|>" if tokenizer.pad_token is None else tokenizer.pad_token,
    }
    
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added} special tokens")
    
    # Verify tokenization of structure tags
    test_str = "<think>Test</think><output>Test</output>"
    test_tokens = tokenizer.tokenize(test_str)
    print("Tokenization test:", test_tokens)
    
    return tokenizer

def find_token_sequence(token_ids, seq_ids):
    """Returns index where seq_ids starts in token_ids, or -1 if not found."""
    for i in range(len(token_ids) - len(seq_ids) + 1):
        if token_ids[i:i+len(seq_ids)] == seq_ids:
            return i
    return -1

def tokenize_function(ex, tokenizer):
    response = f"<think>{ex['think']}</think><output>{ex['output']}</output>"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ex["question"]},
        {"role": "assistant", "content": response}
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    print("CHAT TEMPLATE TEXT:\n", chat_text)

    token_ids = list(tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
        max_length=2048,
        truncation=True
    ))

    # Find assistant block as a *sequence* of tokens
    im_start_id_seq = tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_token_seq = tokenizer.convert_tokens_to_ids("<|assistant|>")
    im_end_id_seq = tokenizer.convert_tokens_to_ids("<|im_end|>")
    # If tokens are not single-token, encode as a sequence
    if isinstance(im_start_id_seq, int): im_start_id_seq = [im_start_id_seq]
    if isinstance(assistant_token_seq, int): assistant_token_seq = [assistant_token_seq]
    if isinstance(im_end_id_seq, int): im_end_id_seq = [im_end_id_seq]

    # Encode full sequence for <|im_start|><|assistant|>
    assistant_marker = tokenizer.encode("<|im_start|><|assistant|>\n", add_special_tokens=False)
    im_end_marker = tokenizer.encode("<|im_end|>", add_special_tokens=False)

    print('Assistant marker:', assistant_marker)

    # Find last occurrence of assistant_marker
    start_idx = -1
    for i in range(len(token_ids) - len(assistant_marker) + 1):
        if token_ids[i:i+len(assistant_marker)] == assistant_marker:
            start_idx = i + len(assistant_marker)
    assert start_idx != -1, "Did not find assistant block as expected"

    # Optionally skip whitespace/newline
    while start_idx < len(token_ids) and token_ids[start_idx] in [
        tokenizer.convert_tokens_to_ids("\n"), tokenizer.convert_tokens_to_ids(" ")
    ]:
        start_idx += 1

    # Find end (first <|im_end|> after answer start)
    # im_end_marker may also be multi-token!
    end_idx = -1
    for i in range(start_idx, len(token_ids) - len(im_end_marker) + 1):
        if token_ids[i:i+len(im_end_marker)] == im_end_marker:
            end_idx = i
            break
    if end_idx == -1: end_idx = len(token_ids)

    # Mask everything except answer tokens
    labels = [-100] * len(token_ids)
    labels[start_idx:end_idx] = token_ids[start_idx:end_idx]
    attention_mask = [1] * len(token_ids)

    return {
        "input_ids": token_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }

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
            padding="longest",         
            truncation=True,
            max_length=2048,              # pick appropriate max length
            return_tensors=None,          # ✅ Don't return tensor now
            add_special_tokens=False,
        )
    return formatted_text, tokenized

def load_and_tokenize_dataset(tokenizer):
    assert os.path.exists(DATA_PATH), f"Data file not found at {DATA_PATH}"

    # 1) Load the flattened JSONL into a HuggingFace dataset
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    # dataset example
    print("Sample dataset entry:", dataset[0])

    # 2) Tokenize & build labels
    dataset = dataset.map(
    lambda ex: tokenize_function(ex, tokenizer),
    remove_columns=["id", "topic", "question", "think", "output"],
    batched=False
)

    print(f"Dataset loaded with {len(dataset)} examples.")
    print("Sample tokenized example:", dataset[0])
    
    print('-----------------------------')
    labels = dataset[0]['labels']
    input_ids = dataset[0]['input_ids']
    print(tokenizer.decode([t for t in labels if t != -100], skip_special_tokens=False))
    print(tokenizer.decode(input_ids, skip_special_tokens=False))

    print('\n--- Token/Label Alignment Table ---')
    for i, (iid, lbl) in enumerate(zip(input_ids, labels)):
        tok = tokenizer.decode([iid])
        print(f"{i:3d} | token: {tok:16} | label: {lbl}")
    print('-----------------------------------')

    return dataset

def load_model_and_prepare_for_qora(tokenizer, output_dir: Path):
    start = time()
    log("Loading model config and weights…")

    # 1) Base config + 4-bit quant setup
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0,
    )

    # 2) Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb,
    )

    # 2.1) Snapshot everything we'll need downstream
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Saving base model config and tokenizer to {output_dir}")
    config.save_pretrained(output_dir)

    # 3) Resize & patch
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # 4) Prepare for QLoRA
    log("Preparing model for QLoRA adapters…")
    model = prepare_model_for_kbit_training(model)

    # 5) Load your LoRA config, wrap with PEFT…
    assert os.path.exists(LORA_CONFIG_PATH), "Missing LoRA config"
    log(f"Checking LoRA config at {LORA_CONFIG_PATH}…")
    lora_cfg = check_lora_modules(model, LORA_CONFIG_PATH)
    log("Applying LoRA adapters…")
    model = get_peft_model(model, lora_cfg)

    # 6) Monkey-patch generation defaults
    model.generation_config.do_sample   = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p       = 1.0
    model.generation_config.top_k       = 0
    model.config.use_cache              = False

    end = time()
    log(f"✅ Model & LoRA ready in {end - start:.2f}s")
    return model


def save_chat_jinja2(tokenizer, output_dir: Path):
    template_src = Path("templates/chat_template.jinja")
    assert template_src.exists(), f"Template missing at: {template_src}"

    # 1) add the chat template to your tokenizer
    with open(template_src, "r", encoding="utf-8") as f:
        template_text = f.read()
    tokenizer.chat_template = template_text
    tokenizer.init_kwargs["chat_template"] = template_text

    # 2) save the HF tokenizer (this writes tokenizer.json and tokenizer_config.json)
    tokenizer.save_pretrained(output_dir)

    # 3) now open the *fast* tokenizer to dump vocab + merges
    fast_tok = Tokenizer.from_file(str(output_dir / "tokenizer.json"))
    bpe = fast_tok.model  # this is a tokenizers.models.BPE

    # 4) save the BPE tokenizer files
    bpe_folder = output_dir / "bpe-tokenizer"
    bpe_folder.mkdir(exist_ok=True)
    bpe.save(str(bpe_folder))  # writes bpe-tokenizer/vocab.json & bpe-tokenizer/merges.txt

    # Move them up one level if you like:
    (bpe_folder / "vocab.json").rename(output_dir / "vocab.json")
    (bpe_folder / "merges.txt").rename(output_dir / "merges.txt")
    bpe_folder.rmdir()  # remove the now-empty subfolder

    print(f"✅ Chat template + vocab/merges dumped to {output_dir}")


def is_structured_output(text: str) -> bool:
    # Extract only the assistant block (between <|im_start|>assistant and <|im_end|>)
    match = re.search(r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>", text, re.DOTALL)
    if not match:
        return False

    assistant_text = match.group(1)
    return all(tag in assistant_text for tag in ["<think>", "</think>", "<output>", "</output>"])

class EvalCallback(TrainerCallback):
    def __init__(self, tokenizer, output_dir, interval):
        self.tokenizer = tokenizer
        self.interval = interval
        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True, parents=True)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval != 0:
            return

        log(f"🔬 Running evaluation at step {state.global_step}...")

        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            # {"role": "user",      "content": "What is the capital of France?"}
            {"role": "user",      "content": "2+2?"},
        ]

        output_str = run_generation_and_print(
            kwargs["model"],
            self.tokenizer,
            messages,
            label=f"Eval @ step {state.global_step}",
            return_response=True
        )

        # Get last metrics (if available)
        log_dict = state.log_history[-1] if state.log_history else {}

        # Format the metrics as a string
        metrics_str = f"Metrics: {json.dumps(log_dict, indent=2)}\n\n"

        # Combine metrics and LLM output
        log_file = self.logs_dir / f"callback-{state.global_step}.txt"

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(metrics_str)
            f.write(output_str)

def train_model(model, tokenizer, dataset, output_dir):
    log("Configuring training arguments...")

    log("Checking model parameters for meta device...")
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            raise RuntimeError(f"❌ Parameter {name} is still on meta device!")

    log("Disabling reentrant checkpointing...")
    torch.utils.checkpoint._use_reentrant = False

    log("Disabling use_cache for training...")
    model.config.use_cache = False

    log("Instantiating Trainer...")
    model.config.use_cache = False
    torch.utils.checkpoint._use_reentrant = False

    def pad_collator(features):
        # Ensure all sequences are 1D lists
        def flatten1d(x):
            if hasattr(x, "flatten"):
                x = x.flatten()
            if hasattr(x, "tolist"):
                x = x.tolist()
            # If still nested, flatten manually
            if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
                x = [item for sublist in x for item in sublist]
            return x

        # Find the max length in the batch
        max_len = max(len(flatten1d(f["input_ids"])) for f in features)
        batch = {k: [] for k in features[0]}

        for f in features:
            for k, v in f.items():
                pad_token = tokenizer.pad_token_id if k != "labels" else -100
                v = flatten1d(v)
                arr = v + [pad_token] * (max_len - len(v))
                batch[k].append(arr)

        # Convert to tensors
        return {k: torch.tensor(v, dtype=torch.long if k != "labels" else torch.int64) for k, v in batch.items()}

    log("Creating causal collator...")
    # 3) Configure Trainer
    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     per_device_train_batch_size=1,  # Reduce for stability
    #     gradient_accumulation_steps=8,
    #     # num_train_epochs=5,  # Increase epochs
    #     max_steps=50,  # Limit steps for quick testing
    #     learning_rate=1e-3,  # Lower learning rate
    #     weight_decay=0.01,
    #     warmup_ratio=0.1,
    #     logging_steps=10,
    #     save_strategy="epoch",
    #     eval_steps=100,
    #     fp16=True,
    #     optim="paged_adamw_8bit",
    #     report_to="none",
    #     remove_unused_columns=False,  # Important!
    #     group_by_length=True,
    # )

    # Overfit config:
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,    # ← No accumulation
        max_steps=100,                    # ← More optimizer steps
        learning_rate=2e-3,
        weight_decay=0.0,                 # ← Turn off decay
        warmup_steps=0,                   # ← No warmup
        logging_steps=10,
        fp16=False,                       # ← Easier debugging
        optim="adamw_torch",              # ← Simpler optimizer
        remove_unused_columns=False,
    )

    log("Creating Trainer instance...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=pad_collator,
        callbacks=[EvalCallback(tokenizer, output_dir, interval=25)],
    )

    log("Trainer instance created successfully.")
    # 4) Train & save
    trainer.train()
    log("Training completed successfully.")


    model.save_pretrained(output_dir)
    log("Model saved successfully.")


def test_training():
    from peft import PeftModel

    BASE_MODEL = MODEL_NAME
    training_dirs = [
        d for d in os.listdir(OUTPUT_BASE_DIR)
        if d.startswith("training-") and os.path.isdir(os.path.join(OUTPUT_BASE_DIR, d))
    ]
    if not training_dirs:
        TRAINING_NUM = 1
        OUTPUT_DIR = OUTPUT_BASE_DIR / f"training-{TRAINING_NUM}"
        print(f"No training-N directories found. Creating {OUTPUT_DIR} for testing.")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    else:
        nums = [int(d.split('-')[1]) for d in training_dirs if d.split('-')[1].isdigit()]
        TRAINING_NUM = max(nums)
        OUTPUT_DIR = OUTPUT_BASE_DIR / f"training-{TRAINING_NUM}"

    log("Loading base model and tokenizer for testing...")

    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
    chat_template_path = Path(OUTPUT_DIR) / "chat_template.jinja"
    assert chat_template_path.exists(), f"Template missing at: {chat_template_path}"
    tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
    tokenizer.init_kwargs["chat_template"] = tokenizer.chat_template

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")

    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, OUTPUT_DIR, is_trainable=False)

    examples = [
        "2+2?",
        # "What is the capital of France?",
        # "Who wrote 'To Kill a Mockingbird'?",
        # "Explain the theory of relativity in simple terms.",
        # "What is the boiling point of water?",
        # "How do airplanes fly?"
    ]

    for i, question in enumerate(examples, start=1):
        log(f"Processing example {i}: {question}")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question}
        ]
        run_generation_and_print(
            model,
            tokenizer,
            messages,
            label=f"Example {i}"
        )

def main():
    log("Preparing output directory")
    output_dir = prepare_output_dir()

    log("Loading tokenizer and adding special tags")
    tokenizer = load_and_prepare_tokenizer(output_dir)

    log("Debugging special tokens")
    # debugging special tokens
    for tag in ["<think>", "</think>", "<output>", "</output>"]:
        tok = tokenizer.tokenize(tag)
        tid = tokenizer.convert_tokens_to_ids(tag)
        print(f"{tag}: tokens={tok} id={tid}")

    log("Saving chat template to tokenizer")
    save_chat_jinja2(tokenizer, output_dir)

    log("Loading and tokenizing dataset")
    dataset = load_and_tokenize_dataset(tokenizer)

    log("Loading model and applying LoRA")
    model = load_model_and_prepare_for_qora(tokenizer, output_dir)

    print("=== Final Chat Template ===")
    print(tokenizer.chat_template)
    print("===========================")

    log("Training model")
    train_model(model, tokenizer, dataset, output_dir)

    log('Testing training with a small dataset')
    test_training()

if __name__ == "__main__":
    main()