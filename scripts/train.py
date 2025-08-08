import re
import os
import json
import sys
import warnings
import logging as pylog  # stdlib logging
from time import time
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    logging,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig,
    StoppingCriteriaList, 
    StoppingCriteria
)
from tokenizers import Tokenizer

from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from config.config import MODEL_NAME

SYSTEM_PROMPT = (
    "You are a structured assistant. Respond in exactly two parts using the format:\n"
    "<think>[Your reasoning]</think>\n<output>[Your answer]</output>"
)

DATA_PATH = "datasets/data.jsonl"
OUTPUT_BASE_DIR = Path(f"output/{MODEL_NAME}")
LORA_CONFIG_PATH = "config/lora_config.json"

# We'll compute a canonical assistant-open IDs at runtime (see main()).
# Default ASSISTANT_OPEN string used for template matching — your
# chat_template.jinja should produce one of the two common variants:
#   "<|im_start|><|assistant|>\n"  (with newline)
#   "<|im_start|><|assistant|>"   (no newline)
# We'll try to detect which the tokenizer/chat_template emits.
ASSISTANT_OPEN = "<|im_start|><|assistant|>\n"  # kept as default; canonical computed at runtime

# If True, anchor generation prompt inside <output> to force answer tokens
ANCHOR_INTO_OUTPUT = True

# If True, supervise ONLY the <output>...</output> span instead of entire assistant block.
SUPERVISE_OUTPUT_ONLY = False

# Debugging toggle
DEBUG = True
DEBUG_SAMPLE_LIMIT = 10
DEBUG_SAMPLE_RANDOM = False
DEBUG_SAMPLE_PROB = 0.05
_DEBUG_SEEN = 0
DEF_LOG_PREFIX = "🔧 "
DEF_DBG_PREFIX = "🐞 "
FINAL_LOG_FH = None
_ORIG_STDOUT = None
_ORIG_STDERR = None
TEE_ACTIVE = False  # set True after we install the tee streams

def _write_sink(s: str):
    try:
        if FINAL_LOG_FH:
            FINAL_LOG_FH.write(s + "\n")
            FINAL_LOG_FH.flush()
    except Exception:
        pass

def log(msg):
    s = f"\n{DEF_LOG_PREFIX}{msg}\n{'=' * 60}"
    print(s)
    # avoid double-writing: when tee is active, print already goes to file
    if not TEE_ACTIVE:
        _write_sink(s)


def debug(msg):
    if DEBUG:
        s = f"\n{DEF_DBG_PREFIX}{msg}\n{'-' * 60}"
        print(s)
        if not TEE_ACTIVE:
            _write_sink(s)

EVAL_QUESTIONS = [
  "2+2?",
  "Translate 'focus' to Polish.",
  "Is 7 > 5?",
  "Capital of France?",
]

class _TeeStream:
    """
    Tee writes to console AND finalLog.txt without breaking tqdm/Trainer formatting.
    """
    def __init__(self, stream, sink_fh_getter):
        self._stream = stream
        self._sink_getter = sink_fh_getter
    def write(self, data):
        try:
            self._stream.write(data)
        except Exception:
            pass
        try:
            fh = self._sink_getter()
            if fh:
                fh.write(data)
        except Exception:
            pass
    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            fh = self._sink_getter()
            if fh:
                fh.flush()
        except Exception:
            pass

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
            shift_labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss

class StopSequenceCriteria(StoppingCriteria):
     def __init__(self, stop_sequences_ids):
         self.stop_sequences_ids = stop_sequences_ids
         self.window = max(len(s) for s in stop_sequences_ids) if stop_sequences_ids else 0
         self.buf = []

     def __call__(self, input_ids, scores, **kwargs):
         if self.window == 0:
             return False
         self.buf = input_ids[0].tolist()[-self.window:]
         for s in self.stop_sequences_ids:
             if len(self.buf) >= len(s) and self.buf[-len(s):] == s:
                 return True
         return False

def prepare_output_dir() -> Path:
    existing_dirs = [
        d for d in os.listdir(OUTPUT_BASE_DIR) if d.startswith("training-")
    ]
    next_training_num = len(existing_dirs) + 1
    output_dir = OUTPUT_BASE_DIR / f"training-{next_training_num}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_and_prepare_tokenizer(output_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        add_bos_token=False,
        add_eos_token=False,
    )
    # Do NOT extend vocab with new tags under PEFT/QLoRA.
    # Only ensure we have a pad token.
    if tokenizer.pad_token is None:
        # Prefer eos as pad for many chat models to avoid OOV '<|pad|>'
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    # Quick visibility check
    test_str = "<think>Test</think><output>Test</output>"
    test_tokens = tokenizer.tokenize(test_str)
    print("Tokenization test (no new tokens added):", test_tokens)
    return tokenizer


def find_token_sequence(token_ids, seq_ids):
    """Returns index where seq_ids starts in token_ids, or -1 if not found."""
    if not seq_ids:
        return -1
    for i in range(len(token_ids) - len(seq_ids) + 1):
        if token_ids[i : i + len(seq_ids)] == seq_ids:
            return i
    return -1


def tokenize_function(ex, tokenizer, canonical_assistant_ids):
    """
    Tokenize a single example and produce input_ids, labels, attention_mask.
    This function is robust to assistant marker variant: canonical_assistant_ids
    (a list[int]) is used to locate the assistant-open marker inside the
    chat-template token list.
    """

    response = f"<think>{ex['think']}</think><output>{ex['output']}</output>"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ex["question"]},
        {"role": "assistant", "content": response},
    ]

    # Tokenize with chat template (fast tokenizer apply_chat_template)
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
        max_length=2048,
        truncation=True,
    )
    # token_ids is list[int]

    im_end_marker = tokenizer.encode("<|im_end|>", add_special_tokens=False)

    # Locate assistant-open using canonical_assistant_ids
    start_pos = find_token_sequence(token_ids, canonical_assistant_ids)
    if start_pos == -1:
        # fall back to trying both common variants
        cand_with_nl = tokenizer.encode("<|im_start|><|assistant|>\n", add_special_tokens=False)
        cand_no_nl = tokenizer.encode("<|im_start|><|assistant|>", add_special_tokens=False)
        start_pos = find_token_sequence(token_ids, cand_with_nl)
        used_marker = cand_with_nl
        if start_pos == -1:
            start_pos = find_token_sequence(token_ids, cand_no_nl)
            used_marker = cand_no_nl
        else:
            used_marker = cand_with_nl
        if start_pos == -1:
            # helpful debug before failing
            print("❌ Could not find assistant marker in tokens (tokenize_function)")
            tail = token_ids[-120:] if len(token_ids) > 120 else token_ids
            print("tail ids:", tail)
            print("tail toks:", tokenizer.convert_ids_to_tokens(tail))
            raise AssertionError("❌ Could not find assistant marker in tokens")
    else:
        used_marker = canonical_assistant_ids

    start_idx = start_pos + len(used_marker)

    # end marker detection
    end_idx = -1
    for i in range(start_idx, len(token_ids) - len(im_end_marker) + 1):
        if token_ids[i : i + len(im_end_marker)] == im_end_marker:
            end_idx = i                      # ← stop BEFORE <|im_end|>
            break
    if end_idx == -1:
        end_idx = len(token_ids)

    # Build labels.
    labels = [-100] * len(token_ids)

    if SUPERVISE_OUTPUT_ONLY:
        # find <output> ... </output> inside assistant span
        out_open = tokenizer.encode("<output>", add_special_tokens=False)
        out_close = tokenizer.encode("</output>", add_special_tokens=False)
        start_out = find_token_sequence(token_ids[start_idx:end_idx], out_open)
        end_out = find_token_sequence(token_ids[start_idx:end_idx], out_close)
        if start_out != -1 and end_out != -1:
            o_s = start_idx + start_out
            o_e = start_idx + end_out + len(out_close)
            labels[o_s:o_e] = token_ids[o_s:o_e]
        else:
            # fallback to supervising entire assistant span
            labels[start_idx:end_idx] = token_ids[start_idx:end_idx]
            debug("Could not find explicit <output> tags — supervising whole assistant span")
    else:
        labels[start_idx:end_idx] = token_ids[start_idx:end_idx]

    attention_mask = [1] * len(token_ids)

    # Print only a few sample debugs to avoid spamming logs on big datasets
    global _DEBUG_SEEN
    if DEBUG and _DEBUG_SEEN < DEBUG_SAMPLE_LIMIT:
        import random
        if (not DEBUG_SAMPLE_RANDOM) or (random.random() < DEBUG_SAMPLE_PROB):
            dec_labels = tokenizer.decode([t for t in labels if t != -100], skip_special_tokens=False)
            dec_input = tokenizer.decode(token_ids, skip_special_tokens=False)
            debug("=== Tokenize fn sample debug ===")
            debug(f"Decoded labels (trimmed): {dec_labels}")
            debug(f"Decoded input tail: {dec_input[-300:]}")
            debug(f"start_idx={start_idx} end_idx={end_idx} used_marker={tokenizer.convert_ids_to_tokens(used_marker)}")
            _DEBUG_SEEN += 1

    return {"input_ids": token_ids, "labels": labels, "attention_mask": attention_mask}


def format_and_tokenize(messages, tokenizer, return_tensors=False, add_generation_prompt=False, canonical_assistant_ids=None):
    """
    Produce formatted_text and tokenized object. If add_generation_prompt True
    we optionally anchor the prompt inside <output> to force generation there.
    canonical_assistant_ids is a list[int] used to verify template output.
    """
    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )

    if add_generation_prompt:
        # Verify template produced the assistant open; best to check token-level
        if canonical_assistant_ids is not None:
            tmp = tokenizer(formatted_text, return_tensors=None, add_special_tokens=False)
            ids = tmp["input_ids"]
            if isinstance(ids, list):
                ids_list = ids
            elif isinstance(ids, (list, tuple)):
                ids_list = ids
            else:
                # when using fast tokenizer it may return different shape; normalize
                try:
                    ids_list = ids[0] if hasattr(ids, "__len__") else ids
                except Exception:
                    ids_list = ids
            pos = find_token_sequence(ids_list, canonical_assistant_ids)
            if pos == -1:
                print("⚠️ formatted_text does NOT contain canonical assistant marker")
                debug("formatted_text repr: " + repr(formatted_text[-200:]))
                debug("canonical_assistant_ids tokens: " + str(tokenizer.convert_ids_to_tokens(canonical_assistant_ids)))
            else:
                debug("formatted_text contains canonical assistant marker at token pos " + str(pos))
        # Optionally anchor inside output so generation starts at answer location:
        if ANCHOR_INTO_OUTPUT:
            # append safe anchor that matches training layout
            # formatted_text = formatted_text + "<think>"
            pass

    if return_tensors:
        tokenized = tokenizer(formatted_text, return_tensors="pt", add_special_tokens=False)
    else:
        tokenized = tokenizer(
            formatted_text, padding="longest", truncation=True, max_length=2048, return_tensors=None, add_special_tokens=False
        )

    return formatted_text, tokenized


def run_generation_and_print(model, tokenizer, messages, canonical_assistant_ids=None, label="Eval", mode="auto"):
    """
    mode:
      - "auto": no anchor; model may output <think>... or just <output>...
      - "force_think": append "<think>" to encourage think->output
      - "output_only": append "<output>" to force no think section
    """
    formatted_text, inputs = format_and_tokenize(
        messages, tokenizer,
        return_tensors=True,
        add_generation_prompt=True,
        canonical_assistant_ids=canonical_assistant_ids
    )

    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    if mode == "force_think":
        formatted_text = formatted_text + "<think>"
        inputs = tokenizer([formatted_text], return_tensors="pt")
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
    elif mode == "output_only":
        formatted_text = formatted_text + "<output>"
        inputs = tokenizer([formatted_text], return_tensors="pt")
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

    stop_out   = tokenizer.encode("</output>", add_special_tokens=False)
    stop_imend = tokenizer.encode("<|im_end|>", add_special_tokens=False)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=192,
            stopping_criteria=StoppingCriteriaList([StopSequenceCriteria([stop_out, stop_imend])]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=4,      # ← stronger anti-repeat
            repetition_penalty=1.05,     # ← mild penalty
        )

    input_len = inputs["input_ids"].shape[1]
    decoded = tokenizer.decode(output[0][input_len:], skip_special_tokens=False)

    # Include the prompt too (tail to keep logs readable)
    def _tail(s, n=400):
        return s if len(s) <= n else s[-n:]
    prompt_tail = _tail(formatted_text, 800)
    header = f"\n🧪 {label}:\n" if label else "\n🧪 Generation:\n"
    out_str = (
        header +
        "📥 Prompt (tail):\n" + prompt_tail + "\n\n" +
        "📤 Output:\n" + decoded + "\n"
    )

    log(f"Is structured output: {is_structured_output(decoded)}")
    log(f"Is structured output: {is_structured_output(decoded)}")
    
    return out_str


def check_lora_modules(model, lora_config_path: str):
    with open(lora_config_path, "r") as f:
        lora_cfg = LoraConfig(**json.load(f))
    all_module_names = [name for name, _ in model.named_modules()]
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
    return lora_cfg


def load_model_and_prepare_for_qora(tokenizer, output_dir: Path):
    start = time()
    log("Loading model config and weights…")
    # reduce HF verbosity during heavy init to avoid config dumps
    logging.set_verbosity_warning()

    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
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
        quantization_config=bnb,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Saving base model config and tokenizer to {output_dir}")
    config.save_pretrained(output_dir)

    model.config.pad_token_id = tokenizer.pad_token_id

    log("Preparing model for QLoRA adapters…")
    model = prepare_model_for_kbit_training(model)

    assert os.path.exists(LORA_CONFIG_PATH), "Missing LoRA config"
    log(f"Checking LoRA config at {LORA_CONFIG_PATH}…")
    lora_cfg = check_lora_modules(model, LORA_CONFIG_PATH)
    log("Applying LoRA adapters…")
    model = get_peft_model(model, lora_cfg)

    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 0
    model.config.use_cache = False

    end = time()

    log(f"✅ Model & LoRA ready in {end - start:.2f}s")
    logging.set_verbosity_warning()

    return model


def is_structured_output(text: str) -> bool:
	# Try to isolate the assistant chunk if present, else just scan tail
	m = re.search(r"<\|im_start\|><\|assistant\|>\s*(.*)", text, re.DOTALL)
	segment = m.group(1) if m else text
	return all(tag in segment for tag in ("<think>", "</think>", "<output>", "</output>"))


class EvalCallback(TrainerCallback):
    def __init__(self, tokenizer, canonical_assistant_ids, output_dir, interval):
        self.tokenizer = tokenizer
        self.canonical_assistant_ids = canonical_assistant_ids
        self.interval = interval
        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True, parents=True)
    
    def _pick_eval_question(self, state):
        """
        Pick an evaluation question based on the current global step.
        This is a simple round-robin selection from EVAL_QUESTIONS.
        """
        idx = (state.global_step // self.interval) % len(EVAL_QUESTIONS)

        return EVAL_QUESTIONS[idx]

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval != 0:
            return
        
        log(f"🔬 Running evaluation at step {state.global_step}...")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._pick_eval_question(state)},
        ]

        mode = "force_think" if state.global_step < 100 else "auto"

        output_str = run_generation_and_print(
            kwargs["model"],
            self.tokenizer,
            messages,
            canonical_assistant_ids=self.canonical_assistant_ids,
            label=f"Eval @ step {state.global_step}",
            mode=mode
        )

        log_dict = state.log_history[-1] if state.log_history else {}
        metrics_str = f"Metrics: {json.dumps(log_dict, indent=2)}\n\n"
        log_file = self.logs_dir / f"callback-{state.global_step}.txt"
        
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(metrics_str)
            f.write(output_str)

        try:
            if FINAL_LOG_FH:
                FINAL_LOG_FH.write(metrics_str)
                FINAL_LOG_FH.write(output_str)
                FINAL_LOG_FH.flush()
        except Exception:
            pass


def train_model(model, tokenizer, dataset, output_dir, canonical_assistant_ids):
    log("Configuring training arguments...")
    pylog.getLogger("accelerate").setLevel(pylog.INFO)
    pylog.getLogger("peft").setLevel(pylog.INFO)

    for name, param in model.named_parameters():
        if param.device.type == "meta":
            raise RuntimeError(f"❌ Parameter {name} is still on meta device!")

    torch.utils.checkpoint._use_reentrant = False
    model.config.use_cache = False

    def pad_collator(features):
        # flatten & pad manually
        def flatten1d(x):
            if hasattr(x, "flatten"):
                x = x.flatten()
            if hasattr(x, "tolist"):
                x = x.tolist()
            if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
                x = [item for sublist in x for item in sublist]
            return x

        max_len = max(len(flatten1d(f["input_ids"])) for f in features)
        batch = {k: [] for k in features[0]}
        for f in features:
            for k, v in f.items():
                pad_token = tokenizer.pad_token_id if k != "labels" else -100
                v = flatten1d(v)
                arr = v + [pad_token] * (max_len - len(v))
                batch[k].append(arr)
        return {k: torch.tensor(v, dtype=torch.long if k != "labels" else torch.int64) for k, v in batch.items()}

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=14,
        # max_steps=260,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.3,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_steps=100,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        group_by_length=True,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=pad_collator,
        callbacks=[EvalCallback(tokenizer, canonical_assistant_ids, output_dir, interval=20)],
    )

    trainer.train()
    model.save_pretrained(output_dir)


def test_training():
    from peft import PeftModel

    BASE_MODEL = MODEL_NAME
    training_dirs = [
        d
        for d in os.listdir(OUTPUT_BASE_DIR)
        if d.startswith("training-") and os.path.isdir(os.path.join(OUTPUT_BASE_DIR, d))
    ]
    if not training_dirs:
        TRAINING_NUM = 1
        OUTPUT_DIR = OUTPUT_BASE_DIR / f"training-{TRAINING_NUM}"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    else:
        nums = [int(d.split("-")[1]) for d in training_dirs if d.split("-")[1].isdigit()]
        TRAINING_NUM = max(nums)
        OUTPUT_DIR = OUTPUT_BASE_DIR / f"training-{TRAINING_NUM}"

    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
    chat_template_path = Path(OUTPUT_DIR) / "chat_template.jinja"
    assert chat_template_path.exists(), f"Template missing at: {chat_template_path}"
    tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
    tokenizer.init_kwargs["chat_template"] = tokenizer.chat_template

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, OUTPUT_DIR, is_trainable=False)

    examples = ["2+2?"]
    for i, question in enumerate(examples, start=1):
        log(f"Processing example {i}: {question}")
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
        run_generation_and_print(model, tokenizer, messages, canonical_assistant_ids=None, label=f"Example {i}", mode="force_think")


def main():
    log("Preparing output directory")
    output_dir = prepare_output_dir()
    # Global log sink
    global FINAL_LOG_FH
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True, parents=True)
    FINAL_LOG_FH = open(logs_dir / "finalLog.txt", "a", encoding="utf-8")

    # === Tee stdout/stderr so tqdm + Trainer bars and prints land in finalLog.txt ===
    global _ORIG_STDOUT, _ORIG_STDERR
    _ORIG_STDOUT, _ORIG_STDERR = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(_ORIG_STDOUT, lambda: FINAL_LOG_FH)
    sys.stderr = _TeeStream(_ORIG_STDERR, lambda: FINAL_LOG_FH)
    global TEE_ACTIVE
    TEE_ACTIVE = True

    # === Capture Python warnings into the sink ===
    warnings.simplefilter("default")  # show deprecations by default
    def _showwarning(message, category, filename, lineno, file=None, line=None):
        s = warnings.formatwarning(message, category, filename, lineno, line)
        try:
            if FINAL_LOG_FH:
                FINAL_LOG_FH.write(s)
                FINAL_LOG_FH.flush()
        except Exception:
            pass
        # also print to console (already teed)
        try:
            _ORIG_STDERR.write(s)
            _ORIG_STDERR.flush()
        except Exception:
            pass
    warnings.showwarning = _showwarning

    # === Keep Transformers logger quieter to avoid config spam ===
    pylog.basicConfig(level=pylog.WARNING)
    tf_logger = pylog.getLogger("transformers")
    tf_logger.setLevel(pylog.WARNING)

    # avoid duplicate handlers on reruns
    if not any(isinstance(h, pylog.StreamHandler) and getattr(h.stream, "name", "") == FINAL_LOG_FH.name
               for h in tf_logger.handlers if hasattr(h, "stream")):
        tf_logger.addHandler(pylog.StreamHandler(FINAL_LOG_FH))

    # Optional: leave HF internal verbosity at WARNING for clean logs
    logging.set_verbosity_warning()

    log("Loading tokenizer and adding special tags")
    tokenizer = load_and_prepare_tokenizer(output_dir)

    # Determine canonical assistant-open token ids by rendering a sample formatted prompt
    # and checking which variant exists in the tokenization.
    s_nl = tokenizer.encode("<|im_start|><|assistant|>\n", add_special_tokens=False)
    s_no = tokenizer.encode("<|im_start|><|assistant|>", add_special_tokens=False)
    debug(f"candidate s_nl ids: {s_nl} tokens: {tokenizer.convert_ids_to_tokens(s_nl)}")
    debug(f"candidate s_no ids: {s_no} tokens: {tokenizer.convert_ids_to_tokens(s_no)}")

    # Save chat template & tokenizer files (so canonical detection uses same template)
    log("Saving chat template to tokenizer")
    save_dir = output_dir
    with open("templates/chat_template.jinja", "r", encoding="utf-8") as f:
        chat_template_text = f.read()
    tokenizer.chat_template = chat_template_text
    tokenizer.init_kwargs["chat_template"] = chat_template_text
    tokenizer.save_pretrained(save_dir)
    # also dump BPE files for inspection
    fast_tok = Tokenizer.from_file(str(save_dir / "tokenizer.json"))
    bpe = fast_tok.model
    bpe_folder = save_dir / "bpe-tokenizer"
    bpe_folder.mkdir(exist_ok=True)
    bpe.save(str(bpe_folder))
    (bpe_folder / "vocab.json").rename(save_dir / "vocab.json")
    (bpe_folder / "merges.txt").rename(save_dir / "merges.txt")
    bpe_folder.rmdir()
    print(f"✅ Chat template + vocab/merges dumped to {save_dir}")

    # Build a small formatted prompt and detect which variant appears
    fmt, tok = format_and_tokenize(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "__DETECT__"}],
        tokenizer,
        return_tensors=True,
        add_generation_prompt=True,
        canonical_assistant_ids=None,
    )
    fmt_ids = tok["input_ids"][0].tolist()
    pos_no = find_token_sequence(fmt_ids, s_no)
    pos_nl = find_token_sequence(fmt_ids, s_nl)

    if pos_no != -1 and pos_nl == -1:
        canonical_assistant_ids = s_no
        ASSISTANT_OPEN_STR = "<|im_start|><|assistant|>"
        debug("Canonical assistant marker: no-newline variant")
    elif pos_nl != -1 and pos_no == -1:
        canonical_assistant_ids = s_nl
        ASSISTANT_OPEN_STR = "<|im_start|><|assistant|>\n"
        debug("Canonical assistant marker: newline variant")
    elif pos_nl != -1 and pos_no != -1:
        # prefer exact match with no newline if both present (rare)
        canonical_assistant_ids = s_no
        ASSISTANT_OPEN_STR = "<|im_start|><|assistant|>"
        debug("Both variants in template; picking no-newline as canonical")
    else:
        # fallback: prefer s_nl if tokenizer tends to add newline (observed in your logs)
        canonical_assistant_ids = s_nl
        ASSISTANT_OPEN_STR = "<|im_start|><|assistant|>\n"
        debug("No variant found in detection; defaulting to newline variant (best-effort)")

    print("Canonical assistant marker ids:", canonical_assistant_ids, tokenizer.convert_ids_to_tokens(canonical_assistant_ids))
    print("Formatted prompt tail repr:", repr(fmt[-200:]))

    log("Loading and tokenizing dataset")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    print("Sample dataset entry:", dataset[0])

    # Map dataset with our tokenize_function wrapper that uses canonical ids
    def map_fn(ex):
        return tokenize_function(ex, tokenizer, canonical_assistant_ids)

    dataset = dataset.map(map_fn, remove_columns=["id", "topic", "question", "think", "output"], batched=False)
    print(f"Dataset loaded with {len(dataset)} examples.")
    print("Sample tokenized example:", dataset[0])

    stop_ids = tokenizer.encode("</output>", add_special_tokens=False)
    print("stop ids:", stop_ids, tokenizer.convert_ids_to_tokens(stop_ids))

    log("Loading model and applying LoRA")
    model = load_model_and_prepare_for_qora(tokenizer, output_dir)

    print("=== Final Chat Template ===")
    print(tokenizer.chat_template)
    print("===========================")

    log("Training model")
    train_model(model, tokenizer, dataset, output_dir, canonical_assistant_ids)

    log("Testing training with a small dataset")
    test_training()

    try:
        # restore std streams first
        if _ORIG_STDOUT: sys.stdout = _ORIG_STDOUT
        if _ORIG_STDERR: sys.stderr = _ORIG_STDERR
    except Exception:
        pass
    try:
        if FINAL_LOG_FH:
            FINAL_LOG_FH.flush()
            FINAL_LOG_FH.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()