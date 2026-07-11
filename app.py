#!/usr/bin/env python3
"""
AI Fine-Tuning Studio — Gradio Web Application
===============================================
A general-purpose 3-stage LLM fine-tuning studio: train a domain-specific
assistant from scratch on ANY uploaded data, or chat/compare against your
already-trained Healthcare models hosted on HuggingFace Hub.

Layout:
  Sidebar (left)  — Model source, model selection, LoRA/training config
  Main area (tabs)— Chat/Compare, Train from Scratch, Data Upload,
                    Evaluation, Export

HuggingFace repos (ekblaise) — pre-trained Healthcare pipeline:
  - ekblaise/healthcare-qwen2.5-stage1-merged
  - ekblaise/healthcare-qwen2.5-stage2-merged
  - ekblaise/healthcare-qwen2.5-dpo-adapter
  - ekblaise/healthcare-qwen2.5-final

Scanned-PDF extraction uses `marker-pdf` (matches Notebook 01), not pytesseract.

Usage (Colab):
    !python app.py              # share=True gives public URL
    OR
    exec(open("app.py").read())
    demo.launch(share=True)
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os, gc, re, json, time, zipfile, unicodedata, statistics
import warnings
warnings.filterwarnings("ignore")

import torch
import gradio as gr
import pandas as pd

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

HF_USERNAME = "ekblaise"
HF_REPOS = {
    "Base  (Stage 1 merged)": f"{HF_USERNAME}/healthcare-qwen2.5-stage1-merged",
    "SFT   (Stage 2 merged)": f"{HF_USERNAME}/healthcare-qwen2.5-stage2-merged",
    "DPO   (Final model)":    f"{HF_USERNAME}/healthcare-qwen2.5-final",
}

SUPPORTED_BASE_MODELS = {
    "Qwen2.5-1.5B-Instruct (recommended)": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    "Qwen2.5-0.5B-Instruct (fastest)":     "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
    "TinyLlama-1.1B":                       "unsloth/tinyllama-bnb-4bit",
    "Llama-3.2-1B-Instruct":                "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
}

OUTPUT_ROOT = "/content/finetuning_studio"
REPORTS_DIR = f"{OUTPUT_ROOT}/reports"
SRC_DIR     = f"{OUTPUT_ROOT}/src"
for d in (OUTPUT_ROOT, REPORTS_DIR, SRC_DIR):
    os.makedirs(d, exist_ok=True)

DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable and empathetic assistant. Provide accurate, "
    "evidence-based answers. Where relevant, recommend consulting a "
    "qualified professional before acting on the information given."
)

DEFAULT_EVAL_QUESTIONS = [
    "What is the first-line treatment for type 2 diabetes and why?",
    "What are the warning signs of a heart attack and what should someone do immediately?",
    "How do SGLT-2 inhibitors work and what are their benefits beyond blood sugar control?",
    "What are the symptoms of clinical depression and how is it different from normal sadness?",
    "What lifestyle changes are most effective for lowering high blood pressure?",
    "When should a patient with a fever seek emergency medical attention?",
    "What is the difference between a viral and a bacterial infection?",
    "Why is completing the full course of antibiotics important?",
    "What dietary changes help manage type 2 diabetes?",
    "What is diabetic ketoacidosis and what are the emergency steps to manage it?",
]

SAFETY_PATTERNS = [
    r"\b(take|prescribe|dose|dosage|mg|milligram|inject|administer)\b",
    r"\b(diagnosis|diagnose|you have|you are suffering)\b",
    r"\b(stop|discontinue|increase|decrease)\s+your\s+\w+",
]
SAFETY_DISCLAIMER = (
    "\n\n---\n*Please consult a qualified professional before making "
    "any decisions based on this information. Educational purposes only.*"
)

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Global state ───────────────────────────────────────────────────────────────
_loaded_models = {}     # cache: {label: (model, tokenizer)}
_active_model   = None
_active_tok     = None
_active_label   = None

# Registry of every model the user has trained this session.
# Each entry: label -> {"path": <merged dir>, "stage": "PT"|"SFT"|"DPO", "domain": str}
_trained_registry = {}

def _register_trained(label, path, stage, domain):
    _trained_registry[label] = {"path": path, "stage": stage, "domain": domain}

def _models_available_as_input():
    """Base foundation models + every trained merged model, for use as a training input."""
    choices = [f"Foundation · {k}" for k in SUPPORTED_BASE_MODELS.keys()]
    choices += [f"Trained · {label}" for label in _trained_registry.keys()]
    return choices

def _load_jsonl(path):
    """
    Robustly load a dataset file that may be:
      - JSONL (one JSON object per line), or
      - a JSON array of objects, or
      - JSONL with blank lines / trailing commas.
    Returns (records, message). Raises ValueError with a clear message on bad data.
    """
    with open(path, encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        raise ValueError("The uploaded dataset file is empty.")

    # Case 1: a JSON array  [ {...}, {...} ]
    if raw[0] == "[":
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data, f"Loaded JSON array ({len(data)} records)"
        except json.JSONDecodeError as e:
            raise ValueError(f"File starts with '[' but is not valid JSON: {e}")

    # Case 2: JSONL — parse line by line, skipping blanks, tolerating trailing commas
    records = []
    for i, line in enumerate(raw.split("\n"), start=1):
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Line {i} is not valid JSON: {e}\n   → {line[:120]}")
    if not records:
        raise ValueError("No valid JSON records found in the file.")
    return records, f"Loaded JSONL ({len(records)} records)"


def _resolve_input_model(choice):
    """Map an input-model dropdown choice to a concrete model id/path."""
    if choice.startswith("Foundation · "):
        key = choice.replace("Foundation · ", "")
        return SUPPORTED_BASE_MODELS.get(key, list(SUPPORTED_BASE_MODELS.values())[0])
    if choice.startswith("Trained · "):
        label = choice.replace("Trained · ", "")
        entry = _trained_registry.get(label)
        return entry["path"] if entry else None
    return None


# ═════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════

def _load_model(model_id: str, label: str):
    """Load a model by HuggingFace ID or local path. Caches to avoid reloads."""
    global _active_model, _active_tok, _active_label

    if label in _loaded_models:
        _active_model, _active_tok = _loaded_models[label]
        _active_label = label
        return True, f"✅ {label} already loaded (cached)."

    try:
        import unsloth  # noqa
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id, max_seq_length=512, dtype=None, load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    except Exception as e:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16, device_map="auto"
            )
            model.eval()
        except Exception as e2:
            return False, f"❌ Load failed:\n  Unsloth: {e}\n  HF: {e2}"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    _loaded_models[label] = (model, tokenizer)
    _active_model, _active_tok = model, tokenizer
    _active_label = label
    return True, f"✅ {label} loaded successfully."



# ═════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def _generate(model, tokenizer, question: str, system_prompt: str,
              max_tokens: int = 200, temp: float = 0.3, add_safety: bool = True) -> str:
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question.strip()},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    in_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=True,
            temperature=temp, top_p=0.9, repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    ans = tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()
    if add_safety and any(re.search(p, ans.lower()) for p in SAFETY_PATTERNS):
        ans += SAFETY_DISCLAIMER
    return ans


def chat_answer(question: str, system_prompt: str, max_tokens: int, temperature: float):
    global _active_model, _active_tok
    if not question.strip():
        return "Please enter a question.", "—"
    if _active_model is None:
        return "⚠️ No model loaded. Use the sidebar to load a model first.", "❌ No model loaded"
    t0 = time.time()
    answer = _generate(_active_model, _active_tok, question, system_prompt,
                       int(max_tokens), temperature)
    latency = round(time.time() - t0, 1)
    return answer, f"✅ {_active_label} | {len(answer.split())} words | {latency}s"


def compare_answer(question: str, system_prompt: str, max_tokens: int, progress=gr.Progress()):
    """Run the same question through Base, SFT, and DPO — Hub or custom-trained."""
    if not question.strip():
        return "—", "—", "—", "⚠️ Please enter a question first."

    # Prefer the most recent trained model of each stage, else fall back to Hub
    def _latest(stage_code):
        hits = [(lbl, e) for lbl, e in _trained_registry.items() if e["stage"] == stage_code]
        return hits[-1] if hits else None

    stages = []
    for stage_code, hub_repo in [("PT",  HF_REPOS["Base  (Stage 1 merged)"]),
                                 ("SFT", HF_REPOS["SFT   (Stage 2 merged)"]),
                                 ("DPO", HF_REPOS["DPO   (Final model)"])]:
        found = _latest(stage_code)
        if found:
            stages.append((f"Trained · {found[0]}", found[1]["path"]))
        else:
            stages.append((stage_code, hub_repo))

    answers = []
    for i, (label, repo_or_path) in enumerate(stages):
        progress(i / 3, desc=f"Loading {label}...")
        ok, msg = _load_model(repo_or_path, label)
        if ok:
            m, t = _loaded_models[label]
            progress((i + 0.5) / 3, desc=f"Generating with {label}...")
            ans = _generate(m, t, question, system_prompt, int(max_tokens))
        else:
            ans = f"❌ Could not load {label}: {msg}"
        answers.append(ans)

    status = f"✅ Comparison complete — {question[:60]}..."
    return answers[0], answers[1], answers[2], status


# ═════════════════════════════════════════════════════════════════════════════
# DATA INGESTION  (matches Notebook 01: PyMuPDF + marker-pdf fallback for scans)
# ═════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\u00ad", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text.strip()


def extract_from_pdf(pdf_path: str) -> tuple[list, str]:
    """
    Extract text using PyMuPDF when possible.
    If any page appears scanned (very little extractable text),
    fall back to Marker OCR for the entire document — same approach as Notebook 01.
    """
    import fitz  # PyMuPDF

    paragraphs = []
    scanned_detected = False

    with fitz.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if len(text) < 50:
                scanned_detected = True
                break
            paragraphs.extend(
                p.strip() for p in text.split("\n\n") if len(p.strip()) > 40
            )

    if not scanned_detected:
        return paragraphs, f"✅ PyMuPDF extraction ({len(paragraphs)} paragraphs)"

    # ── Scanned PDF → Marker OCR (matches Notebook 01 exactly) ──────────────
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered  = converter(pdf_path)
        markdown, _, _ = text_from_rendered(rendered)

        paras = [p.strip() for p in markdown.split("\n\n") if len(p.strip()) > 40]
        return paras, f"✅ Scanned PDF detected → Marker OCR extraction ({len(paras)} paragraphs)"
    except ImportError:
        return [], "❌ Scanned PDF detected but marker-pdf is not installed. Run: !pip install -q marker-pdf"
    except Exception as e:
        return [], f"❌ Marker OCR failed: {e}"


def extract_text(file_path: str) -> tuple[str, str]:
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".txt":
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return text, f"✅ Plain text loaded ({len(text):,} chars)"

        elif ext == ".pdf":
            paras, status = extract_from_pdf(file_path)
            return "\n\n".join(paras), status

        elif ext in (".docx", ".doc"):
            from docx import Document
            doc = Document(file_path)
            paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paras), f"✅ DOCX extracted ({len(paras)} paragraphs)"

        elif ext == ".csv":
            df = pd.read_csv(file_path)
            text_cols = [c for c in df.columns if df[c].dtype == object]
            if not text_cols:
                return "", "❌ No text columns found in CSV."
            combined = []
            for col in text_cols[:3]:
                for val in df[col].dropna().astype(str):
                    if len(val.strip()) > 20:
                        combined.append(val.strip())
            return "\n\n".join(combined), f"✅ CSV: {len(combined)} records from {text_cols[:3]}"

        else:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return text, f"✅ Loaded as plain text ({len(text):,} chars)"

    except Exception as e:
        return "", f"❌ Extraction failed: {e}"


def process_upload(file_obj):
    if file_obj is None:
        return "", "No file uploaded.", gr.update(visible=False)

    text, status = extract_text(file_obj.name)
    if not text:
        return "", status, gr.update(visible=False)

    cleaned    = clean_text(text)
    paragraphs = [p.strip() for p in cleaned.split("\n\n") if len(p.strip()) >= 80]

    seen, unique = set(), []
    for p in paragraphs:
        key = p[:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)

    save_path = f"{OUTPUT_ROOT}/uploaded_non_instruction_data.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(unique))

    char_total = sum(len(p) for p in unique)
    stats_text = (
        f"{status}\n\n"
        f"📊 Paragraphs extracted: {len(unique)}\n"
        f"📝 Total characters: {char_total:,}\n"
        f"💾 Saved to: {save_path}\n\n"
        f"Recommended minimum (≥50): {'✅ Met' if len(unique) >= 50 else '⚠️ Consider adding more data'}"
    )
    preview = "\n\n".join(unique[:3])
    return preview, stats_text, gr.update(visible=True, value=f"✅ {len(unique)} paragraphs ready for training")


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING PIPELINE — train from scratch on ANY domain, ANY uploaded data
# ═════════════════════════════════════════════════════════════════════════════

def _push_to_hub(model, tokenizer, repo_name, emit_fn):
    """Optionally push a merged model to the ekblaise HuggingFace account."""
    try:
        emit_fn(f"   ⬆️  Pushing to HuggingFace: {repo_name} ...")
        model.push_to_hub_merged(
            repo_name, tokenizer, save_method="merged_16bit",
        )
        emit_fn(f"   ✅ Pushed → https://huggingface.co/{repo_name}")
        return True
    except Exception as e:
        emit_fn(f"   ⚠️  Hub push failed ({e}). Model is still saved locally.")
        return False


def train_one_stage(
    stage, domain_name, input_model_choice, run_name,
    lora_r, lora_alpha, learning_rate, max_steps,
    corpus_used, instruction_file, preference_file,
    push_to_hub, hub_repo_name,
    progress=gr.Progress(),
):
    """
    Train exactly ONE stage, using an explicitly chosen input model.
      stage = "Pre-training (domain adaptation)" | "SFT (instruction tuning)" | "DPO (alignment)"
    The resulting merged model is registered so it can feed the next stage
    or be selected in the Chat tab.
    """
    log = []
    def emit(msg):
        log.append(msg); return "\n".join(log)

    domain = (domain_name or "custom").strip()
    input_model = _resolve_input_model(input_model_choice)
    if input_model is None:
        yield emit(f"❌ Could not resolve input model: {input_model_choice}")
        yield emit("   If it is a 'Trained · …' model, train that stage first.")
        return
    if input_model_choice.startswith("Trained ·") and not os.path.exists(str(input_model)):
        yield emit(f"❌ The selected trained model no longer exists on disk: {input_model}")
        return

    # Auto run-name if not supplied
    stage_short = {"Pre-training (domain adaptation)": "PT",
                   "SFT (instruction tuning)": "SFT",
                   "DPO (alignment)": "DPO"}[stage]
    run_name = (run_name or f"{domain}-{stage_short}-{time.strftime('%H%M%S')}").strip().replace(" ", "-")

    yield emit(f"🚀 Training stage: {stage}")
    yield emit(f"   Run name:    {run_name}")
    yield emit(f"   Domain:      {domain}")
    yield emit(f"   Input model: {input_model_choice}")
    yield emit(f"                → {input_model}")
    yield emit(f"   LoRA r={int(lora_r)}, alpha={int(lora_alpha)} | LR={learning_rate:.0e} | steps={int(max_steps)}")
    yield emit("")

    try:
        import unsloth  # noqa
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
        from datasets import Dataset
    except ImportError as e:
        yield emit(f"❌ Import error: {e}")
        yield emit("   Run: !pip install unsloth transformers==4.56.2 --no-deps trl==0.22.2")
        return

    system_prompt = (
        f"You are a knowledgeable and empathetic {domain} assistant. "
        "Provide accurate, evidence-based answers."
    )
    out_dir  = f"{OUTPUT_ROOT}/{run_name}"
    os.makedirs(out_dir, exist_ok=True)
    merged_dir = f"{out_dir}/merged"

    gc.collect(); torch.cuda.empty_cache()

    # ══════════════════════════ PRE-TRAINING ══════════════════════════════════
    if stage == "Pre-training (domain adaptation)":
        data_path = f"{OUTPUT_ROOT}/uploaded_non_instruction_data.txt"
        if not os.path.exists(data_path):
            yield emit("⚠️  No corpus extracted yet (Step 3). Using a small placeholder.")
            paras = ["This is placeholder domain text used only when no corpus was uploaded."] * 20
        else:
            with open(data_path) as f:
                raw = f.read()
            paras = [p.strip() for p in raw.split("\n\n") if len(p.strip()) >= 80]
            yield emit(f"   Corpus: {len(paras)} paragraphs")

        try:
            progress(0.15, desc="Loading model...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=input_model, max_seq_length=512, dtype=None, load_in_4bit=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"; model.config.use_cache = False
            model = FastLanguageModel.get_peft_model(
                model, r=int(lora_r), lora_alpha=int(lora_alpha),
                target_modules=LORA_TARGET_MODULES, lora_dropout=0,
                bias="none", use_gradient_checkpointing="unsloth", random_state=42)
            yield emit("   ✅ Model + LoRA ready.")

            dataset = Dataset.from_list([{"text": p} for p in paras])
            cfg = SFTConfig(
                output_dir=f"{out_dir}/logs", max_steps=int(max_steps),
                per_device_train_batch_size=1, gradient_accumulation_steps=8,
                learning_rate=learning_rate, warmup_steps=3,
                fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
                optim="adamw_8bit", dataset_text_field="text",
                max_length=512, packing=True, logging_steps=5,
                save_strategy="no", report_to="none", seed=42, remove_unused_columns=False)
            trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=dataset, args=cfg)

            progress(0.3, desc="Training...")
            yield emit("   Training…")
            t0 = time.time(); result = trainer.train()
            yield emit(f"   ✅ Loss={result.training_loss:.4f} | {round(time.time()-t0,1)}s | "
                       f"{round(torch.cuda.max_memory_allocated()/1024**3,2)}GB")

            model.save_pretrained(f"{out_dir}/adapter"); tokenizer.save_pretrained(f"{out_dir}/adapter")
            model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
            _register_trained(run_name, merged_dir, "PT", domain)
            yield emit(f"   💾 Merged model → {merged_dir}")

            if push_to_hub and hub_repo_name.strip():
                _push_to_hub(model, tokenizer, f"{HF_USERNAME}/{hub_repo_name.strip()}", emit)

            del trainer, model; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            yield emit(f"   ❌ Pre-training error: {e}")
            return

    # ══════════════════════════════ SFT ═══════════════════════════════════════
    elif stage == "SFT (instruction tuning)":
        instr_path = instruction_file.name if instruction_file else None
        if instr_path and os.path.exists(instr_path):
            try:
                records, load_msg = _load_jsonl(instr_path)
                yield emit(f"   {load_msg}")
            except ValueError as ve:
                yield emit(f"   ❌ Could not read instruction file:\n   {ve}")
                return
        else:
            yield emit("   ⚠️  No instruction file. Using placeholder data.")
            records = [{"instruction": "What is this domain about?", "input": "",
                        "output": "Placeholder training data used only when no dataset was uploaded."}] * 30
        try:
            progress(0.15, desc="Loading model...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=input_model, max_seq_length=512, dtype=None, load_in_4bit=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"; model.config.use_cache = False
            model = FastLanguageModel.get_peft_model(
                model, r=int(lora_r), lora_alpha=int(lora_alpha),
                target_modules=LORA_TARGET_MODULES, lora_dropout=0.05,
                bias="none", use_gradient_checkpointing="unsloth", random_state=42)
            yield emit("   ✅ Model + LoRA ready.")

            # Split into prompt / completion so completion_only_loss can mask the
            # prompt tokens — matches the reference notebook (no DataCollator needed).
            def fmt(rec):
                prompt_msgs = [{"role": "system", "content": system_prompt},
                               {"role": "user", "content": rec.get("instruction", "").strip()}]
                prompt = tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True)
                completion = rec.get("output", "").strip() + tokenizer.eos_token
                return {"prompt": prompt, "completion": completion}
            dataset = Dataset.from_list([fmt(r) for r in records])
            yield emit("   Response-only loss: ✅ (completion_only_loss=True)")

            cfg = SFTConfig(
                output_dir=f"{out_dir}/logs", max_steps=int(max_steps),
                per_device_train_batch_size=1, gradient_accumulation_steps=8,
                learning_rate=learning_rate, warmup_steps=3,
                fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
                optim="paged_adamw_8bit",
                completion_only_loss=True,
                max_length=512, packing=False, logging_steps=5,
                save_strategy="no", report_to="none", seed=42, remove_unused_columns=False)
            trainer = SFTTrainer(model=model, processing_class=tokenizer,
                                 train_dataset=dataset, args=cfg)

            progress(0.3, desc="Training...")
            yield emit("   Training…")
            t0 = time.time(); result = trainer.train()
            yield emit(f"   ✅ Loss={result.training_loss:.4f} | {round(time.time()-t0,1)}s | "
                       f"{round(torch.cuda.max_memory_allocated()/1024**3,2)}GB")

            model.save_pretrained(f"{out_dir}/adapter"); tokenizer.save_pretrained(f"{out_dir}/adapter")
            model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
            _register_trained(run_name, merged_dir, "SFT", domain)
            yield emit(f"   💾 Merged model → {merged_dir}")

            if push_to_hub and hub_repo_name.strip():
                _push_to_hub(model, tokenizer, f"{HF_USERNAME}/{hub_repo_name.strip()}", emit)

            del trainer, model; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            yield emit(f"   ❌ SFT error: {e}")
            return

    # ══════════════════════════════ DPO ═══════════════════════════════════════
    elif stage == "DPO (alignment)":
        pref_path = preference_file.name if preference_file else None
        if pref_path and os.path.exists(pref_path):
            try:
                pref_records, load_msg = _load_jsonl(pref_path)
                yield emit(f"   {load_msg}")
            except ValueError as ve:
                yield emit(f"   ❌ Could not read preference file:\n   {ve}")
                return
        else:
            yield emit("   ⚠️  No preference file. Using placeholder data.")
            pref_records = [{"prompt": "What is this domain about?",
                             "chosen": "A detailed, accurate, well-structured answer with caveats.",
                             "rejected": "idk, look it up"}] * 25
        try:
            progress(0.15, desc="Loading model...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=input_model, max_seq_length=512, dtype=None, load_in_4bit=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"; model.config.use_cache = False
            model = FastLanguageModel.get_peft_model(
                model, r=int(lora_r), lora_alpha=int(lora_alpha),
                target_modules=LORA_TARGET_MODULES, lora_dropout=0.05,
                bias="none", use_gradient_checkpointing="unsloth", random_state=42)
            try:
                from unsloth import PatchDPOTrainer
                PatchDPOTrainer(); yield emit("   ✅ PatchDPOTrainer applied.")
            except Exception:
                pass

            # Validate the preference schema DPO requires
            missing = [k for k in ("prompt", "chosen", "rejected")
                       if k not in pref_records[0]]
            if missing:
                yield emit(f"   ❌ Preference data missing required field(s): {missing}")
                yield emit("      Each record needs: prompt, chosen, rejected")
                return
            dataset = Dataset.from_list(pref_records)
            cfg = DPOConfig(
                output_dir=f"{out_dir}/logs", max_steps=int(max_steps),
                per_device_train_batch_size=1, gradient_accumulation_steps=8,
                learning_rate=learning_rate, warmup_steps=3,
                fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
                optim="paged_adamw_8bit", beta=0.1,
                max_length=512, max_prompt_length=256,
                logging_steps=5, save_strategy="no",
                report_to="none", seed=42, remove_unused_columns=False)
            trainer = DPOTrainer(model=model, ref_model=None, processing_class=tokenizer,
                                 train_dataset=dataset, args=cfg)

            progress(0.3, desc="Training...")
            yield emit("   Training…")
            t0 = time.time(); result = trainer.train()
            yield emit(f"   ✅ Loss={result.training_loss:.4f} | {round(time.time()-t0,1)}s | "
                       f"{round(torch.cuda.max_memory_allocated()/1024**3,2)}GB")

            tokenizer.padding_side = "right"
            model.save_pretrained(f"{out_dir}/adapter"); tokenizer.save_pretrained(f"{out_dir}/adapter")
            model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
            _register_trained(run_name, merged_dir, "DPO", domain)
            yield emit(f"   ✨ Final merged model → {merged_dir}")

            if push_to_hub and hub_repo_name.strip():
                _push_to_hub(model, tokenizer, f"{HF_USERNAME}/{hub_repo_name.strip()}", emit)

            # Make immediately usable in Chat
            from unsloth import FastLanguageModel as FLM
            FLM.for_inference(model)
            global _active_model, _active_tok, _active_label
            _loaded_models[f"Trained · {run_name}"] = (model, tokenizer)
            _active_model, _active_tok, _active_label = model, tokenizer, f"Trained · {run_name}"

            del trainer; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            yield emit(f"   ❌ DPO error: {e}")
            return

    yield emit("")
    yield emit("=" * 55)
    yield emit(f"🏁 DONE — '{run_name}' registered.")
    yield emit(f"   Use it as input for the next stage, or pick it in the Chat tab.")


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def run_evaluation(questions_text: str, system_prompt: str, progress=gr.Progress()):
    global _active_model, _active_tok
    if _active_model is None:
        return "⚠️ No model loaded. Load one from the sidebar first.", pd.DataFrame(), ""

    questions = [q.strip() for q in questions_text.split("\n") if q.strip()] or DEFAULT_EVAL_QUESTIONS

    try:
        from rouge_score import rouge_scorer as rs_mod
        scorer_obj = rs_mod.RougeScorer(["rougeL"], use_stemmer=True)
        has_rouge = True
    except ImportError:
        has_rouge = False

    rows, rouge_scores = [], []
    for i, q in enumerate(questions):
        progress((i + 1) / len(questions), desc=f"[{i+1}/{len(questions)}] {q[:45]}...")
        ans   = _generate(_active_model, _active_tok, q, system_prompt, max_tokens=200)
        rouge = 0.0
        if has_rouge:
            rouge = scorer_obj.score(q, ans)["rougeL"].fmeasure
            rouge_scores.append(rouge)
        has_safety = SAFETY_DISCLAIMER.strip()[:20] in ans
        rows.append({
            "#": i + 1,
            "Question": q,
            "Answer (excerpt)": ans[:250] + ("..." if len(ans) > 250 else ""),
            "Words": len(ans.split()),
            "Safety ✓": "✅" if has_safety else "—",
            "ROUGE-L": f"{rouge:.3f}" if has_rouge else "N/A",
        })

    df  = pd.DataFrame(rows)
    avg = f"{sum(rouge_scores)/len(rouge_scores):.3f}" if rouge_scores else "N/A"
    log = "\n".join(f"  [{i+1}/{len(questions)}] ✅ {q[:65]}" for i, q in enumerate(questions))
    summary = f"✅ Evaluation complete\n   Model: {_active_label}\n   Avg ROUGE-L: {avg}\n   Questions answered: {len(rows)}/{len(questions)}"
    return log, df, summary


# ═════════════════════════════════════════════════════════════════════════════
# EXPORT
# ═════════════════════════════════════════════════════════════════════════════

def make_export_zip(include_weights: bool):
    zip_path = f"{OUTPUT_ROOT}/finetuning_studio_export.zip"
    included = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(OUTPUT_ROOT):
            for fname in files:
                full = os.path.join(root, fname)
                rel  = os.path.relpath(full, OUTPUT_ROOT)
                if not include_weights and fname.endswith((".safetensors", ".bin")):
                    continue
                if "export" in fname:
                    continue
                zf.write(full, rel)
                included.append(rel)
    size_mb = os.path.getsize(zip_path) / 1024**2
    log = "\n".join(f"  ✅ {f}" for f in sorted(included)[:30])
    if len(included) > 30:
        log += f"\n  ... and {len(included) - 30} more files"
    return zip_path, f"✅ ZIP created: {size_mb:.1f} MB\n{log}"



# ═════════════════════════════════════════════════════════════════════════════
# MODEL CHOICE RESOLUTION (single dropdown → concrete model)
# ═════════════════════════════════════════════════════════════════════════════

HUB_CHOICES = {
    "Healthcare · Base — Stage 1 (Hub)": HF_REPOS["Base  (Stage 1 merged)"],
    "Healthcare · SFT — Stage 2 (Hub)":  HF_REPOS["SFT   (Stage 2 merged)"],
    "Healthcare · DPO — Final (Hub)":    HF_REPOS["DPO   (Final model)"],
}

def all_model_choices():
    """Hub Healthcare models + every model trained this session."""
    trained = [f"Trained · {label}  ({e['stage']})" for label, e in _trained_registry.items()]
    return list(HUB_CHOICES.keys()) + trained

def resolve_and_load(choice: str, progress=gr.Progress()):
    progress(0.2, desc="Resolving model...")
    if not choice:
        return "⚠️ Pick a model first."
    if choice in HUB_CHOICES:
        ok, msg = _load_model(HUB_CHOICES[choice], choice)
    elif choice.startswith("Trained · "):
        label = choice.replace("Trained · ", "").split("  (")[0]
        entry = _trained_registry.get(label)
        if not entry or not os.path.exists(entry["path"]):
            return f"⚠️ '{label}' is no longer available. Re-train it."
        ok, msg = _load_model(entry["path"], choice)
    else:
        return "⚠️ Unknown model choice."
    progress(1.0)
    return msg

def refresh_model_dropdown():
    """Return an updated dropdown listing all currently available models."""
    return gr.update(choices=all_model_choices())

def refresh_input_dropdown():
    """Return an updated dropdown of models usable as a training input."""
    return gr.update(choices=_models_available_as_input())


# Per-stage default hyperparameters — applied automatically, user can still override.
STAGE_DEFAULTS = {
    "Pre-training (domain adaptation)": {"lora_r": 16, "lora_alpha": 32, "lr": 2e-4, "steps": 60},
    "SFT (instruction tuning)":         {"lora_r": 16, "lora_alpha": 32, "lr": 1e-4, "steps": 60},
    "DPO (alignment)":                  {"lora_r": 16, "lora_alpha": 32, "lr": 5e-5, "steps": 60},
}

def on_stage_change(stage):
    """When the stage changes, swap in that stage's default hyperparameters,
    show only the relevant dataset upload, and suggest an input-model default."""
    d = STAGE_DEFAULTS.get(stage, STAGE_DEFAULTS["Pre-training (domain adaptation)"])

    show_corpus = stage == "Pre-training (domain adaptation)"
    show_instr  = stage == "SFT (instruction tuning)"
    show_pref   = stage == "DPO (alignment)"

    # Input-model hint: foundation for PT, a trained model for SFT/DPO
    if show_corpus:
        input_hint = f"Foundation · {list(SUPPORTED_BASE_MODELS.keys())[0]}"
    else:
        trained = list(_trained_registry.keys())
        input_hint = f"Trained · {trained[-1]}" if trained else f"Foundation · {list(SUPPORTED_BASE_MODELS.keys())[0]}"

    return (
        gr.update(value=d["lora_r"]),          # lora_r_sl
        gr.update(value=d["lora_alpha"]),      # lora_alpha_sl
        gr.update(value=d["lr"]),              # lr_sl
        gr.update(value=d["steps"]),           # steps_sl
        gr.update(visible=show_corpus),        # corpus column
        gr.update(visible=show_instr),         # instruction column
        gr.update(visible=show_pref),          # preference column
        gr.update(value=input_hint),           # input_model_dd
    )

# ═════════════════════════════════════════════════════════════════════════════
# GRADIO UI — clean, editorial, no persistent sidebar
# ═════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,500;0,6..72,600&display=swap');

/* ══ Force a single light, calm palette regardless of OS/browser dark mode ══ */
:root, .dark {
    --acc:       #0f766e;   /* deep teal, used sparingly            */
    --acc-soft:  #f0f7f6;   /* barely-there teal tint               */
    --ink:       #17211e;   /* near-black text                      */
    --muted:     #64726e;   /* secondary text                       */
    --line:      #e6eae8;   /* hairline borders                     */
    --paper:     #ffffff;   /* main background                      */
    --paper-2:   #fafbfb;   /* subtle panel background              */

    --body-background-fill:#ffffff !important;
    --background-fill-primary:#ffffff !important;
    --background-fill-secondary:#fafbfb !important;
    --block-background-fill:#ffffff !important;
    --block-label-background-fill:transparent !important;
    --block-label-text-color:#64726e !important;
    --block-title-text-color:#17211e !important;
    --body-text-color:#17211e !important;
    --body-text-color-subdued:#64726e !important;
    --border-color-primary:#e6eae8 !important;
    --input-background-fill:#ffffff !important;
    --checkbox-background-color-selected:#0f766e !important;
    --color-accent:#0f766e !important;
    --color-accent-soft:#f0f7f6 !important;
    --link-text-color:#0f766e !important;
}

body, .gradio-container, .dark body, .dark .gradio-container {
    background:#ffffff !important; color:#17211e !important;
}
.gradio-container {
    max-width:1200px !important; margin:auto !important;
    font-family:'Inter',-apple-system,sans-serif !important;
}

/* text never washes out on white */
p, span, label, li, td, th, .prose, .prose *, textarea, input,
.markdown, .markdown *, h1, h2, h3, h4 { color:#17211e; }
.markdown a, a { color:#0f766e !important; }

/* ── Header ─────────────────────────────────────────────── */
#hdr { padding:30px 2px 22px; border-bottom:1px solid var(--line); margin-bottom:26px; }
#hdr-title {
    font-family:'Newsreader',Georgia,serif !important;
    font-size:2.15rem !important; font-weight:600 !important;
    letter-spacing:-0.02em; color:#17211e !important; margin:0 0 7px !important; line-height:1.1;
}
#hdr-sub { font-size:0.95rem !important; color:#64726e !important; margin:0 !important; max-width:640px; }

/* ── Kill the ugly filled label pills — labels are plain text ── */
.block > label > span,
span[data-testid="block-info"],
.gr-form > div > label span,
label.svelte-1gfkn6j,
.wrap.svelte-1gfkn6j > label {
    background:transparent !important;
    color:#64726e !important;
    font-size:0.8rem !important;
    font-weight:500 !important;
    padding-left:0 !important;
}

/* ── Radio & checkbox: clean cards, subtle selected state (no big teal fill) ── */
fieldset label, .gr-radio label, .wrap label {
    background:#ffffff !important;
    border:1px solid var(--line) !important;
    border-radius:9px !important;
    padding:9px 13px !important;
    color:#17211e !important;
    font-weight:450 !important;
    transition:border-color .15s, background .15s;
}
fieldset label:hover, .gr-radio label:hover { border-color:#c9d3d0 !important; }
fieldset label.selected, .gr-radio label.selected,
input[type="radio"]:checked + span {
    background:var(--acc-soft) !important;
    border-color:var(--acc) !important;
    color:#0f766e !important;
    font-weight:500 !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
button.primary, .gr-button-primary {
    background:#17211e !important; color:#fff !important; border:none !important;
    border-radius:9px !important; font-weight:500 !important; box-shadow:none !important;
    transition:opacity .15s, transform .08s;
}
button.primary:hover { opacity:.88; }
button.primary:active { transform:scale(.985); }
button.secondary {
    background:#fff !important; color:#17211e !important;
    border:1px solid var(--line) !important; border-radius:9px !important;
    font-weight:500 !important; box-shadow:none !important;
}
button.secondary:hover { border-color:#c9d3d0 !important; }

/* ── Inputs ──────────────────────────────────────────────── */
.block, .gr-box { border-radius:11px !important; box-shadow:none !important; border-color:var(--line) !important; }
textarea, input[type="text"], input[type="number"], .gr-dropdown {
    border-radius:9px !important; border:1px solid var(--line) !important;
    background:#fff !important; font-family:'Inter',sans-serif !important;
}
textarea:focus, input:focus {
    border-color:var(--acc) !important; box-shadow:0 0 0 3px var(--acc-soft) !important; outline:none !important;
}

/* ── Tabs — underline style ──────────────────────────────── */
.tabs { border:none !important; }
.tab-nav { border-bottom:1px solid var(--line) !important; gap:2px; }
.tab-nav button {
    font-weight:500 !important; font-size:0.92rem !important; color:#64726e !important;
    border:none !important; border-bottom:2px solid transparent !important;
    border-radius:0 !important; padding:11px 18px !important; background:transparent !important;
}
.tab-nav button.selected { color:#17211e !important; border-bottom:2px solid var(--acc) !important; }

/* ── Step number badges in the Train flow ────────────────── */
.step-label { font-weight:600 !important; font-size:0.95rem !important; color:#17211e !important; margin:6px 0 2px !important; }
.step-hint  { color:#64726e !important; font-size:0.85rem !important; margin:0 0 6px !important; }

/* ── Status pill ─────────────────────────────────────────── */
.status-pill {
    font-size:0.85rem !important; padding:10px 14px; border-radius:9px;
    background:var(--acc-soft) !important; border:1px solid rgba(15,118,110,.15); margin-top:6px;
}
.status-pill p { color:#0f766e !important; margin:0 !important; font-weight:500; }

/* ── Dataframe + examples ────────────────────────────────── */
/* ── Dataframe — force readable light cells (fixes dark-on-dark) ──────── */
table { border-radius:9px !important; overflow:hidden; border:1px solid var(--line) !important; }
.gr-dataframe, .gr-dataframe *, [class*="dataframe"] table, [class*="dataframe"] * {
    background:#ffffff !important;
    color:#17211e !important;
}
.gr-dataframe thead th, [class*="dataframe"] thead th, thead th {
    background:#f2f5f4 !important;
    color:#17211e !important;
    font-weight:600 !important;
    border-bottom:1px solid var(--line) !important;
}
.gr-dataframe tbody td, [class*="dataframe"] tbody td, tbody td {
    background:#ffffff !important;
    color:#17211e !important;
    border-bottom:1px solid #eef1f0 !important;
}
.gr-dataframe tbody tr:nth-child(even) td,
[class*="dataframe"] tbody tr:nth-child(even) td {
    background:#fafbfb !important;
}
.gr-dataframe tbody tr:hover td, [class*="dataframe"] tbody tr:hover td {
    background:#f0f7f6 !important;
}
.gr-samples-table button {
    border-radius:20px !important; border:1px solid var(--line) !important;
    background:#fff !important; color:#17211e !important; font-size:0.8rem !important;
}
.gr-samples-table button:hover { border-color:var(--acc) !important; background:var(--acc-soft) !important; }

/* ── Slim divider + footer ───────────────────────────────── */
.slim-divider { border:none; border-top:1px solid var(--line); margin:22px 0 18px; }
#footer { color:#64726e !important; font-size:0.8rem !important; text-align:center; padding:26px 0 6px; }
"""

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    radius_size=gr.themes.sizes.radius_md,
)
_theme_overrides = dict(
    body_background_fill="#ffffff",
    block_background_fill="#ffffff",
    block_label_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#e6eae8",
    block_shadow="none",
    input_background_fill="#ffffff",
)
try:
    THEME = THEME.set(**_theme_overrides)
except TypeError:
    import inspect
    valid = set(inspect.signature(THEME.set).parameters.keys())
    THEME = THEME.set(**{k: v for k, v in _theme_overrides.items() if k in valid})


with gr.Blocks(title="Fine-Tuning Studio", theme=THEME, css=CUSTOM_CSS) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    with gr.Column(elem_id="hdr"):
        gr.Markdown("Fine-Tuning Studio", elem_id="hdr-title")
        gr.Markdown(
            "Converse with the pre-trained Healthcare pipeline on HuggingFace Hub, "
            "or fine-tune a brand-new assistant on your own domain — one workspace, "
            "a three-stage pipeline (domain adaptation → instruction tuning → DPO).",
            elem_id="hdr-sub",
        )

    with gr.Tabs():

        # ══════════════════════════════ CHAT ══════════════════════════════════
        with gr.Tab("Chat"):
            gr.Markdown("#### Choose a model")
            with gr.Row():
                model_dd = gr.Dropdown(
                    choices=all_model_choices(),
                    value="Healthcare · DPO — Final (Hub)",
                    label="Model", scale=5,
                )
                refresh_btn = gr.Button("↻", variant="secondary", scale=1, min_width=48)
                load_btn = gr.Button("Load", variant="primary", scale=1, min_width=110)
            load_status = gr.Markdown("_No model loaded yet._", elem_classes="status-pill")
            load_btn.click(resolve_and_load, inputs=[model_dd], outputs=[load_status])
            refresh_btn.click(refresh_model_dropdown, inputs=[], outputs=[model_dd])

            with gr.Accordion("Generation settings", open=False):
                system_prompt_tb = gr.Textbox(label="System prompt", value=DEFAULT_SYSTEM_PROMPT, lines=3)
                with gr.Row():
                    max_tokens_sl  = gr.Slider(50, 400, value=200, step=10, label="Max response tokens")
                    temperature_sl = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Temperature")

            gr.Markdown('<hr class="slim-divider">')

            with gr.Tabs():
                with gr.Tab("Ask"):
                    chat_q = gr.Textbox(label="Your question", lines=2,
                                        placeholder="What is the first-line treatment for type 2 diabetes?")
                    ask_btn = gr.Button("Ask", variant="primary")
                    chat_out = gr.Textbox(label="Answer", lines=10)
                    chat_status = gr.Textbox(label="Status", lines=1)
                    ask_btn.click(chat_answer,
                                  inputs=[chat_q, system_prompt_tb, max_tokens_sl, temperature_sl],
                                  outputs=[chat_out, chat_status])
                    gr.Examples(examples=[[q] for q in DEFAULT_EVAL_QUESTIONS[:6]], inputs=[chat_q])

                with gr.Tab("Compare stages"):
                    gr.Markdown(
                        "Runs the same question through **Base → SFT → DPO**. Uses your freshly "
                        "trained model when available, otherwise the Hub Healthcare pipeline."
                    )
                    cmp_q = gr.Textbox(label="Question to compare", lines=2)
                    cmp_btn = gr.Button("Run comparison", variant="primary")
                    cmp_stat = gr.Textbox(label="Status", lines=1)
                    with gr.Row():
                        cmp_base = gr.Textbox(label="Base", lines=12)
                        cmp_sft  = gr.Textbox(label="SFT",  lines=12)
                        cmp_dpo  = gr.Textbox(label="DPO",  lines=12)
                    cmp_btn.click(compare_answer, inputs=[cmp_q, system_prompt_tb, max_tokens_sl],
                                  outputs=[cmp_base, cmp_sft, cmp_dpo, cmp_stat])
                    gr.Examples(examples=[[q] for q in DEFAULT_EVAL_QUESTIONS[:3]], inputs=[cmp_q])

        # ══════════════════════════════ TRAIN ═════════════════════════════════
        with gr.Tab("Train a new model"):
            gr.Markdown(
                "Train **one stage at a time**. Each finished model is saved and added to the "
                "model list — so you can feed a pre-trained model into SFT, an SFT model into DPO, "
                "and train as many models as you like."
            )

            gr.Markdown("Step 1 · Name & domain", elem_classes="step-label")
            gr.Markdown("The run name identifies this model in the lists. Domain shapes the system prompt.",
                        elem_classes="step-hint")
            with gr.Row():
                domain_name_tb = gr.Textbox(label="Domain", placeholder="e.g. Legal, Finance, Nutrition…")
                run_name_tb    = gr.Textbox(label="Run name (optional)", placeholder="auto-generated if blank")

            gr.Markdown("Step 2 · Choose the stage to train", elem_classes="step-label")
            gr.Markdown(
                "Pre-training adapts a foundation model to your domain. SFT teaches instruction "
                "following (input: a pre-trained model). DPO aligns preferences (input: an SFT model).",
                elem_classes="step-hint",
            )
            stage_dd = gr.Radio(
                choices=["Pre-training (domain adaptation)",
                         "SFT (instruction tuning)",
                         "DPO (alignment)"],
                value="Pre-training (domain adaptation)",
                label="Stage",
            )

            gr.Markdown("Step 3 · Choose the input model", elem_classes="step-label")
            gr.Markdown(
                "For pre-training, pick a foundation model. For SFT or DPO, pick a model you've "
                "already trained (the merged output of the previous stage). Press ↻ to refresh.",
                elem_classes="step-hint",
            )
            with gr.Row():
                input_model_dd = gr.Dropdown(
                    choices=_models_available_as_input(),
                    value=f"Foundation · {list(SUPPORTED_BASE_MODELS.keys())[0]}",
                    label="Input model", scale=5,
                )
                input_refresh_btn = gr.Button("↻", variant="secondary", scale=1, min_width=48)
            input_refresh_btn.click(refresh_input_dropdown, inputs=[], outputs=[input_model_dd])

            gr.Markdown("Step 4 · Provide the dataset for this stage", elem_classes="step-label")
            gr.Markdown("Only the dataset for the selected stage is shown.", elem_classes="step-hint")

            # Pre-training corpus (visible for PT only)
            with gr.Column(visible=True) as corpus_col:
                gr.Markdown("**Raw corpus** — PDF, DOCX, CSV, TXT, or scanned PDF")
                corpus_file  = gr.File(label="Raw corpus",
                                       file_types=[".pdf", ".docx", ".doc", ".csv", ".txt"], show_label=False)
                extract_btn  = gr.Button("Extract & preview", variant="secondary", size="sm")
                corpus_badge = gr.Markdown("", visible=False, elem_classes="status-pill")
                with gr.Accordion("Corpus extraction details", open=False):
                    corpus_status = gr.Textbox(label="Extraction status", lines=5)
                    corpus_prev   = gr.Textbox(label="Preview (first 3 paragraphs)", lines=8)
                extract_btn.click(process_upload, inputs=[corpus_file],
                                  outputs=[corpus_prev, corpus_status, corpus_badge])

            # SFT instruction data (hidden until SFT selected)
            with gr.Column(visible=False) as instr_col:
                gr.Markdown("**Instruction pairs** — `.jsonl` with `instruction` / `output` fields")
                instr_file = gr.File(label="Instruction data", file_types=[".jsonl", ".json"], show_label=False)

            # DPO preference data (hidden until DPO selected)
            with gr.Column(visible=False) as pref_col:
                gr.Markdown("**Preference pairs** — `.jsonl` with `prompt` / `chosen` / `rejected` fields")
                pref_file  = gr.File(label="Preference data", file_types=[".jsonl", ".json"], show_label=False)

            with gr.Accordion("Advanced hyperparameters (stage defaults applied automatically)", open=False):
                gr.Markdown(
                    "Defaults follow the reference notebooks: **PT** r=16 α=32 LR=2e-4 · "
                    "**SFT** LR=1e-4 · **DPO** LR=5e-5. Change only if you know why.",
                    elem_classes="step-hint",
                )
                with gr.Row():
                    lora_r_sl     = gr.Slider(4, 64, value=16, step=4, label="LoRA rank (r)")
                    lora_alpha_sl = gr.Slider(4, 128, value=32, step=4, label="LoRA alpha")
                with gr.Row():
                    lr_sl    = gr.Slider(1e-5, 5e-4, value=2e-4, step=1e-5, label="Learning rate")
                    steps_sl = gr.Slider(10, 300, value=60, step=10, label="Max steps")

            # Wire the stage radio to swap defaults + toggle dataset visibility
            stage_dd.change(
                on_stage_change,
                inputs=[stage_dd],
                outputs=[lora_r_sl, lora_alpha_sl, lr_sl, steps_sl,
                         corpus_col, instr_col, pref_col, input_model_dd],
            )

            gr.Markdown("Step 5 · (Optional) push final model to HuggingFace", elem_classes="step-label")
            gr.Markdown(f"Pushes the merged model to your account (**{HF_USERNAME}**). "
                        "Requires `notebook_login()` beforehand.", elem_classes="step-hint")
            with gr.Row():
                push_cb       = gr.Checkbox(label="Push to HuggingFace Hub after training", value=False)
                hub_repo_tb   = gr.Textbox(label="Repo name",
                                           placeholder="e.g. legal-qwen2.5-dpo-final", scale=2)

            train_btn = gr.Button("Start training this stage", variant="primary", size="lg")
            train_log = gr.Textbox(label="Training log", lines=22, max_lines=60, autoscroll=True)
            train_btn.click(
                train_one_stage,
                inputs=[stage_dd, domain_name_tb, input_model_dd, run_name_tb,
                        lora_r_sl, lora_alpha_sl, lr_sl, steps_sl,
                        corpus_file, instr_file, pref_file, push_cb, hub_repo_tb],
                outputs=[train_log],
            )
            gr.Markdown(
                '<hr class="slim-divider">After a stage finishes, press ↻ on the input-model list '
                'to feed it into the next stage, or press ↻ in the **Chat** tab to talk to it.'
            )

        # ══════════════════════════════ EVALUATE ══════════════════════════════
        with gr.Tab("Evaluate"):
            gr.Markdown(
                "Benchmark the **currently loaded model** (set in the Chat tab) and compute "
                "ROUGE-L scores.\n\n**Healthcare run** — Base 0.125 · SFT 0.242 (+0.118) · "
                "DPO aligned (rewards/accuracy = 1.0)."
            )
            eval_questions_tb = gr.Textbox(label="Evaluation questions (one per line)",
                                           value="\n".join(DEFAULT_EVAL_QUESTIONS), lines=6)
            eval_btn = gr.Button("Run evaluation", variant="primary")
            eval_log = gr.Textbox(label="Progress", lines=10)
            eval_table = gr.Dataframe(label="Results", wrap=True)
            eval_summary = gr.Textbox(label="Summary", lines=4)
            eval_btn.click(run_evaluation, inputs=[eval_questions_tb, system_prompt_tb],
                           outputs=[eval_log, eval_table, eval_summary])

        # ══════════════════════════════ EXPORT ════════════════════════════════
        with gr.Tab("Export"):
            gr.Markdown("Download your training artifacts as a single archive.")
            inc_weights_cb = gr.Checkbox(label="Include merged model weights (~3GB per stage)", value=False)
            export_btn = gr.Button("Create export ZIP", variant="primary")
            export_file = gr.File(label="Download")
            export_status = gr.Textbox(label="Contents", lines=12)
            export_btn.click(make_export_zip, inputs=[inc_weights_cb], outputs=[export_file, export_status])
            gr.Markdown(f"""
            <hr class="slim-divider">

            **HuggingFace Hub — pre-trained Healthcare pipeline ({HF_USERNAME})**

            | Stage | Repo |
            |---|---|
            | Base (Stage 1) | [`{HF_REPOS['Base  (Stage 1 merged)']}`](https://huggingface.co/{HF_REPOS['Base  (Stage 1 merged)']}) |
            | SFT (Stage 2)  | [`{HF_REPOS['SFT   (Stage 2 merged)']}`](https://huggingface.co/{HF_REPOS['SFT   (Stage 2 merged)']}) |
            | DPO (Final)    | [`{HF_REPOS['DPO   (Final model)']}`](https://huggingface.co/{HF_REPOS['DPO   (Final model)']}) |
            """)

    gr.Markdown("Fine-Tuning Studio · Unsloth + TRL + QLoRA + DPO", elem_id="footer")


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(share=True, server_port=7860, show_error=True, quiet=False)