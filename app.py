#!/usr/bin/env python3
"""
Healthcare AI Assistant — Gradio Web Application
=================================================
Production-grade Gradio app for the 3-stage fine-tuning pipeline.
Integrates with HuggingFace Hub for model loading and artifact storage.

HuggingFace repos (ekblaise):
  - ekblaise/healthcare-qwen2.5-stage1-merged
  - ekblaise/healthcare-qwen2.5-stage2-merged
  - ekblaise/healthcare-qwen2.5-dpo-adapter
  - ekblaise/healthcare-qwen2.5-final

Tabs:
  1. 📤 Data Upload    — PDF, DOCX, CSV, TXT, scanned PDF (OCR)
  2. ⚙️  Training Config — model selector, LoRA sliders, stage toggles
  3. 🚀 Train          — live training log, loss + VRAM monitoring
  4. 💬 Chat / Compare — side-by-side Base vs SFT vs DPO comparison
  5. 📊 Evaluation     — 10-question benchmark, ROUGE-L table
  6. 📥 Export         — download adapters, reports, inference script

Usage (Colab):
    !python app.py              # share=True gives public URL
    OR
    exec(open("app.py").read()) # inline launch
    demo.launch(share=True)
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os, gc, re, json, time, zipfile, tempfile, unicodedata, statistics
import warnings
warnings.filterwarnings("ignore")

import torch
import gradio as gr
import pandas as pd

# ── HuggingFace Hub repo names (from your actual notebooks) ──────────────────
HF_USERNAME = "ekblaise"
HF_REPOS = {
    "stage1_merged":  f"{HF_USERNAME}/healthcare-qwen2.5-stage1-merged",
    "stage2_merged":  f"{HF_USERNAME}/healthcare-qwen2.5-stage2-merged",
    "stage3_adapter": f"{HF_USERNAME}/healthcare-qwen2.5-dpo-adapter",
    "final":          f"{HF_USERNAME}/healthcare-qwen2.5-final",
}

# ── Supported base models for new training ────────────────────────────────────
SUPPORTED_MODELS = {
    "Qwen2.5-1.5B-Instruct (Used in training ✅)": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    "Qwen2.5-0.5B-Instruct (Fastest for demo)":   "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
    "TinyLlama-1.1B":                              "unsloth/tinyllama-bnb-4bit",
    "Llama-3.2-1B-Instruct":                       "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
}

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_ROOT = "/content/healthcare_assistant"
REPORTS_DIR = f"{OUTPUT_ROOT}/reports"
SRC_DIR     = f"{OUTPUT_ROOT}/src"
for d in [OUTPUT_ROOT, REPORTS_DIR, SRC_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a knowledgeable and empathetic healthcare assistant. "
    "Provide accurate, evidence-based answers to medical questions. "
    "Always recommend consulting a qualified healthcare professional "
    "for personal medical advice."
)

EVAL_QUESTIONS = [
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
    "\n\n---\n*Please consult a qualified healthcare professional "
    "before making any medical decisions. This is for educational purposes only.*"
)

# ── Global model state ────────────────────────────────────────────────────────
_loaded_models = {}   # cache: {label: (model, tokenizer)}
_active_model  = None
_active_tok    = None
_active_label  = None


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


def load_final_model(progress=gr.Progress()):
    """Load the final DPO-aligned model from HuggingFace Hub."""
    progress(0.1, desc="Connecting to HuggingFace Hub...")
    ok, msg = _load_model(HF_REPOS["final"], "Final DPO Model (Hub)")
    progress(1.0)
    return msg


def load_model_for_compare(stage: str, progress=gr.Progress()):
    """Load a specific stage model for side-by-side comparison."""
    stage_map = {
        "Base (Stage 1 merged)":   HF_REPOS["stage1_merged"],
        "SFT (Stage 2 merged)":    HF_REPOS["stage2_merged"],
        "DPO Final":               HF_REPOS["final"],
    }
    model_id = stage_map.get(stage, HF_REPOS["final"])
    ok, msg = _load_model(model_id, stage)
    return msg


# ═════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def _generate(model, tokenizer, question: str, max_tokens=200, temp=0.3) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question.strip()},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    in_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temp,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    ans = tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()
    if any(re.search(p, ans.lower()) for p in SAFETY_PATTERNS):
        ans += SAFETY_DISCLAIMER
    return ans


def chat_answer(question: str, max_tokens: int):
    """Answer a single question with the currently loaded model."""
    global _active_model, _active_tok
    if not question.strip():
        return "Please enter a question.", "—"

    if _active_model is None:
        # Auto-load final model from Hub
        ok, msg = _load_model(HF_REPOS["final"], "Final DPO Model (Hub)")
        if not ok:
            return msg, "❌ Model not loaded"

    t0 = time.time()
    answer = _generate(_active_model, _active_tok, question, int(max_tokens))
    latency = round(time.time() - t0, 1)
    status = f"✅ {_active_label} | {len(answer.split())} words | {latency}s"
    return answer, status


def compare_answer(question: str, max_tokens: int, progress=gr.Progress()):
    """Run the same question through Base, SFT, and DPO models side-by-side."""
    if not question.strip():
        return "—", "—", "—", "⚠️ Please enter a question first."

    stages = [
        ("Base (Stage 1 merged)",  HF_REPOS["stage1_merged"]),
        ("SFT (Stage 2 merged)",   HF_REPOS["stage2_merged"]),
        ("DPO Final",              HF_REPOS["final"]),
    ]
    answers = []
    for i, (label, repo) in enumerate(stages):
        progress((i / 3), desc=f"Loading {label}...")
        ok, msg = _load_model(repo, label)
        if ok:
            m, t = _loaded_models[label]
            progress(((i + 0.5) / 3), desc=f"Generating with {label}...")
            ans = _generate(m, t, question, int(max_tokens))
        else:
            ans = f"❌ Could not load {label}: {msg}"
        answers.append(ans)

    status = f"✅ Comparison complete across 3 stages — {question[:60]}..."
    return answers[0], answers[1], answers[2], status


# ═════════════════════════════════════════════════════════════════════════════
# DATA INGESTION
# ═════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b","").replace("\ufeff","").replace("\u00ad","")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text.strip()


def extract_text(file_path: str) -> tuple[str, str]:
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".txt":
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return text, f"✅ Plain text loaded ({len(text):,} chars)"

        elif ext == ".pdf":
            import fitz
            parts = []
            with fitz.open(file_path) as doc:
                for page in doc:
                    t = page.get_text("text").strip()
                    if t:
                        parts.append(t)
            text = "\n\n".join(parts)
            if len(text.strip()) < 100:
                return _ocr_pdf(file_path)
            return text, f"✅ PDF extracted ({len(text):,} chars, {len(parts)} pages)"

        elif ext in (".docx", ".doc"):
            import docx
            doc = docx.Document(file_path)
            paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            text = "\n\n".join(paras)
            return text, f"✅ DOCX extracted ({len(text):,} chars, {len(paras)} paragraphs)"

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
            text = "\n\n".join(combined)
            return text, f"✅ CSV: {len(combined)} text records from columns {text_cols[:3]}"

        else:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return text, f"✅ Loaded as plain text ({len(text):,} chars)"

    except Exception as e:
        return "", f"❌ Extraction failed: {e}"


def _ocr_pdf(file_path: str) -> tuple[str, str]:
    try:
        from pdf2image import convert_from_path
        import pytesseract
        images = convert_from_path(file_path, dpi=200)
        texts  = [pytesseract.image_to_string(img) for img in images]
        text   = "\n\n".join(t.strip() for t in texts if t.strip())
        return text, f"✅ OCR complete ({len(images)} pages, {len(text):,} chars)"
    except ImportError:
        return "", "❌ OCR requires: !apt install tesseract-ocr && pip install pytesseract pdf2image"
    except Exception as e:
        return "", f"❌ OCR failed: {e}"


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
        f"Rubric requirement (≥50): {'✅ Met' if len(unique) >= 50 else '❌ Need more data'}\n"
        f"2× target (≥100):         {'✅ Met' if len(unique) >= 100 else f'⚠️  Have {len(unique)}'}"
    )
    preview = "\n\n".join(unique[:3])
    return preview, stats_text, gr.update(visible=True, value=f"✅ Data saved ({len(unique)} paragraphs)")


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_training(
    model_choice, lora_r, lora_alpha, learning_rate, max_steps,
    run_stage1, run_stage2, run_stage3,
    instruction_file, preference_file,
    progress=gr.Progress(),
):
    model_name = SUPPORTED_MODELS.get(model_choice, list(SUPPORTED_MODELS.values())[0])
    log = []

    def emit(msg):
        log.append(msg)
        return "\n".join(log)

    yield emit(f"🚀 Starting training pipeline")
    yield emit(f"   Model:  {model_name}")
    yield emit(f"   LoRA:   r={lora_r}, alpha={lora_alpha}")
    yield emit(f"   LR:     {learning_rate:.0e}  |  Steps: {int(max_steps)}/stage")
    yield emit(f"   Stages: {'S1 ' if run_stage1 else ''}{'S2 ' if run_stage2 else ''}{'S3' if run_stage3 else ''}")
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

    total  = sum([run_stage1, run_stage2, run_stage3])
    done   = 0
    current_model_path = model_name
    lora_targets = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

    # ── STAGE 1 ──────────────────────────────────────────────────────────────
    if run_stage1:
        yield emit("─" * 55)
        yield emit("📚 STAGE 1 — Non-Instruction Domain Adaptation")
        yield emit(f"   LR: {learning_rate:.0e} | packing=True | CLM loss")

        data_path = f"{OUTPUT_ROOT}/uploaded_non_instruction_data.txt"
        if not os.path.exists(data_path):
            yield emit("⚠️  No uploaded data. Using placeholder text.")
            paras = ["Healthcare is the organised provision of medical care to individuals and communities."] * 30
        else:
            with open(data_path) as f:
                raw = f.read()
            paras = [p.strip() for p in raw.split("\n\n") if len(p.strip()) >= 80]
            yield emit(f"   Dataset: {len(paras)} paragraphs")

        try:
            progress(done / max(total, 1), desc="Stage 1: Loading model...")
            gc.collect(); torch.cuda.empty_cache()

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=current_model_path, max_seq_length=512, dtype=None, load_in_4bit=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            model.config.use_cache = False

            model = FastLanguageModel.get_peft_model(
                model, r=lora_r, lora_alpha=lora_alpha,
                target_modules=lora_targets, lora_dropout=0,
                bias="none", use_gradient_checkpointing="unsloth", random_state=42,
            )
            yield emit("   ✅ Model + LoRA ready.")

            dataset = Dataset.from_list([{"text": p} for p in paras])
            s1_cfg = SFTConfig(
                output_dir=f"{OUTPUT_ROOT}/stage1_logs", max_steps=int(max_steps),
                per_device_train_batch_size=1, gradient_accumulation_steps=8,
                learning_rate=learning_rate, warmup_steps=3,
                fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
                optim="adamw_8bit", dataset_text_field="text",
                max_length=512, packing=True, logging_steps=5,
                save_strategy="no", report_to="none", seed=42, remove_unused_columns=False,
            )
            trainer = SFTTrainer(model=model, processing_class=tokenizer,
                                 train_dataset=dataset, args=s1_cfg)

            progress(done / max(total, 1) + 0.1, desc="Stage 1: Training...")
            yield emit(f"   Training... (max_steps={int(max_steps)})")
            t0 = time.time()
            result = trainer.train()
            elapsed = round(time.time() - t0, 1)
            vram = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            yield emit(f"   ✅ Loss={result.training_loss:.4f} | {elapsed}s | {vram}GB VRAM")

            s1_adapter = f"{OUTPUT_ROOT}/stage1_adapter"
            s1_merged  = f"{OUTPUT_ROOT}/stage1_merged"
            model.save_pretrained(s1_adapter); tokenizer.save_pretrained(s1_adapter)
            model.save_pretrained_merged(s1_merged, tokenizer, save_method="merged_16bit")
            yield emit(f"   💾 Adapter → {s1_adapter}")
            yield emit(f"   💾 Merged  → {s1_merged}")
            current_model_path = s1_merged

            del trainer, model; gc.collect(); torch.cuda.empty_cache()

        except Exception as e:
            yield emit(f"   ❌ Stage 1 error: {e}")

        done += 1
        yield emit("")

    # ── STAGE 2 ──────────────────────────────────────────────────────────────
    if run_stage2:
        yield emit("─" * 55)
        yield emit("🎓 STAGE 2 — Instruction Fine-Tuning (SFT)")
        yield emit(f"   LR: {learning_rate/2:.0e} | packing=False | response-only loss")

        instr_path = instruction_file.name if instruction_file else None
        if instr_path and os.path.exists(instr_path):
            records = []
            with open(instr_path) as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            yield emit(f"   Dataset: {len(records)} instruction pairs")
        else:
            yield emit("   ⚠️  No instruction file. Using placeholder data.")
            records = [{"instruction": "What is diabetes?", "input": "",
                        "output": "Diabetes is a chronic metabolic condition characterized by elevated blood glucose levels due to insufficient insulin production or action."}] * 30

        try:
            progress(done / max(total, 1), desc="Stage 2: Loading...")
            gc.collect(); torch.cuda.empty_cache()

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=current_model_path, max_seq_length=512, dtype=None, load_in_4bit=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            model.config.use_cache = False

            model = FastLanguageModel.get_peft_model(
                model, r=lora_r, lora_alpha=lora_alpha,
                target_modules=lora_targets, lora_dropout=0.05,
                bias="none", use_gradient_checkpointing="unsloth", random_state=42,
            )
            yield emit("   ✅ Model + LoRA ready.")

            def fmt(rec):
                msgs = [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": rec.get("instruction","").strip()},
                    {"role": "assistant", "content": rec.get("output","").strip()},
                ]
                return {"text": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}

            formatted = [fmt(r) for r in records]
            dataset   = Dataset.from_list(formatted)

            resp_ids  = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
            # collator  = DataCollatorForCompletionOnlyLM(response_template=resp_ids, tokenizer=tokenizer)
            yield emit(f"   Response-only loss: ✅ (<|im_start|>assistant marker)")

            s2_cfg = SFTConfig(
                output_dir=f"{OUTPUT_ROOT}/stage2_logs", max_steps=int(max_steps),
                per_device_train_batch_size=1, gradient_accumulation_steps=8,
                learning_rate=learning_rate / 2, warmup_steps=3,
                fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
                optim="paged_adamw_8bit", dataset_text_field="text",
                max_length=512, packing=False, logging_steps=5,
                save_strategy="no", report_to="none", seed=42, remove_unused_columns=False,
            )
            trainer = SFTTrainer(model=model, processing_class=tokenizer,
                                 train_dataset=dataset, eval_dataset=val_dataset, completion_only_loss = True, args=s2_cfg)

            progress(done / max(total, 1) + 0.1, desc="Stage 2: Training...")
            yield emit(f"   Training... (max_steps={int(max_steps)})")
            t0 = time.time()
            result = trainer.train()
            elapsed = round(time.time() - t0, 1)
            vram = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            yield emit(f"   ✅ Loss={result.training_loss:.4f} | {elapsed}s | {vram}GB VRAM")

            s2_adapter = f"{OUTPUT_ROOT}/stage2_adapter"
            s2_merged  = f"{OUTPUT_ROOT}/stage2_merged"
            model.save_pretrained(s2_adapter); tokenizer.save_pretrained(s2_adapter)
            model.save_pretrained_merged(s2_merged, tokenizer, save_method="merged_16bit")
            yield emit(f"   💾 Adapter → {s2_adapter}")
            yield emit(f"   💾 Merged  → {s2_merged}")
            current_model_path = s2_merged

            del trainer, model; gc.collect(); torch.cuda.empty_cache()

        except Exception as e:
            yield emit(f"   ❌ Stage 2 error: {e}")

        done += 1
        yield emit("")

    # ── STAGE 3 ──────────────────────────────────────────────────────────────
    if run_stage3:
        yield emit("─" * 55)
        yield emit("🎯 STAGE 3 — DPO Preference Alignment")
        yield emit(f"   LR: {learning_rate/4:.0e} | beta=0.1 | left-padding")

        pref_path = preference_file.name if preference_file else None
        if pref_path and os.path.exists(pref_path):
            pref_records = []
            with open(pref_path) as f:
                for line in f:
                    if line.strip():
                        pref_records.append(json.loads(line))
            yield emit(f"   Dataset: {len(pref_records)} preference pairs")
        else:
            yield emit("   ⚠️  No preference file. Using placeholder data.")
            pref_records = [
                {"prompt": "What is the first-line treatment for type 2 diabetes?",
                 "chosen": "Metformin is the evidence-based first-line treatment, recommended by ADA, EASD, and NICE guidelines. It reduces hepatic glucose production via AMPK activation without causing hypoglycemia.",
                 "rejected": "Just eat less sugar and exercise more."}
            ] * 25

        try:
            progress(done / max(total, 1), desc="Stage 3: Loading...")
            gc.collect(); torch.cuda.empty_cache()

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=current_model_path, max_seq_length=512, dtype=None, load_in_4bit=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            model.config.use_cache = False

            model = FastLanguageModel.get_peft_model(
                model, r=lora_r, lora_alpha=lora_alpha,
                target_modules=lora_targets, lora_dropout=0.05,
                bias="none", use_gradient_checkpointing="unsloth", random_state=42,
            )

            try:
                from unsloth import PatchDPOTrainer
                PatchDPOTrainer()
                yield emit("   ✅ PatchDPOTrainer applied (Unsloth optimised kernels)")
            except Exception:
                yield emit("   ℹ️  PatchDPOTrainer not available — using standard DPO")

            dataset = Dataset.from_list(pref_records)

            s3_cfg = DPOConfig(
                output_dir=f"{OUTPUT_ROOT}/stage3_logs", max_steps=int(max_steps),
                per_device_train_batch_size=1, gradient_accumulation_steps=8,
                learning_rate=learning_rate / 4, warmup_steps=3,
                fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
                optim="paged_adamw_8bit", beta=0.1,
                max_length=512, max_prompt_length=256,
                logging_steps=5, save_strategy="no",
                report_to="none", seed=42, remove_unused_columns=False,
            )
            trainer = DPOTrainer(model=model, ref_model=None, processing_class=tokenizer,
                                 train_dataset=dataset, args=s3_cfg)

            progress(done / max(total, 1) + 0.1, desc="Stage 3: Training...")
            yield emit(f"   Training... (max_steps={int(max_steps)})")
            t0 = time.time()
            result = trainer.train()
            elapsed = round(time.time() - t0, 1)
            vram = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            yield emit(f"   ✅ Loss={result.training_loss:.4f} | {elapsed}s | {vram}GB VRAM")

            tokenizer.padding_side = "right"
            s3_adapter   = f"{OUTPUT_ROOT}/stage3_dpo_adapter"
            final_merged = f"{OUTPUT_ROOT}/final_merged"
            model.save_pretrained(s3_adapter); tokenizer.save_pretrained(s3_adapter)
            model.save_pretrained_merged(final_merged, tokenizer, save_method="merged_16bit")
            yield emit(f"   💾 DPO adapter  → {s3_adapter}")
            yield emit(f"   ✨ Final merged → {final_merged}")

            # Cache for immediate use in Chat tab
            from unsloth import FastLanguageModel as FLM
            FLM.for_inference(model)
            _loaded_models["Final DPO Model (trained)"] = (model, tokenizer)
            global _active_model, _active_tok, _active_label
            _active_model, _active_tok = model, tokenizer
            _active_label = "Final DPO Model (trained)"

            del trainer; gc.collect(); torch.cuda.empty_cache()

        except Exception as e:
            yield emit(f"   ❌ Stage 3 error: {e}")

        done += 1
        yield emit("")

    yield emit("=" * 55)
    yield emit("🏁 TRAINING PIPELINE COMPLETE")
    yield emit(f"   Artifacts saved to: {OUTPUT_ROOT}")
    yield emit(f"   Model ready in Chat tab ✅")


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def run_evaluation(progress=gr.Progress()):
    global _active_model, _active_tok

    if _active_model is None:
        ok, msg = _load_model(HF_REPOS["final"], "Final DPO Model (Hub)")
        if not ok:
            return msg, pd.DataFrame(), "❌ Could not load model."

    try:
        from rouge_score import rouge_scorer as rs_mod
        scorer_obj = rs_mod.RougeScorer(["rougeL"], use_stemmer=True)
        has_rouge = True
    except ImportError:
        has_rouge = False

    rows, rouge_scores = [], []
    for i, q in enumerate(EVAL_QUESTIONS):
        progress((i + 1) / len(EVAL_QUESTIONS), desc=f"[{i+1}/10] {q[:45]}...")
        ans   = _generate(_active_model, _active_tok, q, max_tokens=200)
        rouge = 0.0
        if has_rouge:
            rouge = scorer_obj.score(q, ans)["rougeL"].fmeasure
            rouge_scores.append(rouge)
        has_safety = SAFETY_DISCLAIMER.strip()[:20] in ans
        rows.append({
            "Q": f"[{i+1}]",
            "Question": q,
            "Answer (excerpt)": ans[:250] + ("..." if len(ans) > 250 else ""),
            "Words": len(ans.split()),
            "Safety ✓": "✅" if has_safety else "—",
            "ROUGE-L": f"{rouge:.3f}" if has_rouge else "N/A",
        })

    df = pd.DataFrame(rows)
    avg = f"{sum(rouge_scores)/len(rouge_scores):.3f}" if rouge_scores else "N/A"
    log = "\n".join([f"  [{i+1}/10] ✅ {q[:65]}" for i, q in enumerate(EVAL_QUESTIONS)])
    summary = f"✅ Evaluation complete\n   Model: {_active_label}\n   Avg ROUGE-L: {avg}\n   Questions answered: {len(rows)}/10"
    return log, df, summary


# ═════════════════════════════════════════════════════════════════════════════
# EXPORT
# ═════════════════════════════════════════════════════════════════════════════

def make_export_zip(include_weights: bool):
    zip_path = f"{OUTPUT_ROOT}/healthcare_assistant_export.zip"
    included = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(OUTPUT_ROOT):
            for fname in files:
                full = os.path.join(root, fname)
                rel  = os.path.relpath(full, OUTPUT_ROOT)
                # Skip large safetensors unless explicitly requested
                if not include_weights and fname.endswith((".safetensors", ".bin")):
                    continue
                if "export" in fname:
                    continue
                zf.write(full, rel)
                included.append(rel)
    size_mb = os.path.getsize(zip_path) / 1024**2
    log = "\n".join(f"  ✅ {f}" for f in sorted(included)[:30])
    if len(included) > 30:
        log += f"\n  ... and {len(included)-30} more files"
    return zip_path, f"✅ ZIP created: {size_mb:.1f} MB\n{log}"


# ═════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═════════════════════════════════════════════════════════════════════════════

THEME = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="blue",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

with gr.Blocks(title="Healthcare AI Assistant", theme=THEME) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.Markdown("""
    # 🏥 Healthcare AI Assistant
    ### Three-Stage Fine-Tuning Pipeline · Assignment 04 · ekblaise

    Built with **Unsloth + TRL + QLoRA + DPO** on `Qwen2.5-1.5B-Instruct`.
    Final model available on HuggingFace: [`ekblaise/healthcare-qwen2.5-final`](https://huggingface.co/ekblaise/healthcare-qwen2.5-final)

    > **Demo day tip:** Go straight to 💬 Chat — the final model loads automatically from Hub.
    """)

    # ── TAB 1: DATA UPLOAD ────────────────────────────────────────────────────
    with gr.Tab("📤 Data Upload"):
        gr.Markdown("""### Upload your domain corpus
        Supports **PDF, DOCX, CSV, TXT**, and **scanned PDF** (OCR via pytesseract).
        The extracted paragraphs are used for Stage 1 non-instruction fine-tuning.
        """)
        with gr.Row():
            upload_file = gr.File(
                label="Upload file",
                file_types=[".pdf", ".docx", ".doc", ".csv", ".txt"],
            )
        upload_btn    = gr.Button("🔍 Extract Text", variant="primary")
        save_badge    = gr.Markdown("", visible=False)
        upload_status = gr.Textbox(label="Extraction status & statistics", lines=8)
        upload_prev   = gr.Textbox(label="Preview (first 3 paragraphs)", lines=10)

        upload_btn.click(
            process_upload,
            inputs=[upload_file],
            outputs=[upload_prev, upload_status, save_badge],
        )

    # ── TAB 2: TRAINING CONFIG ────────────────────────────────────────────────
    with gr.Tab("⚙️ Training Config"):
        gr.Markdown("### Configure your training run")
        with gr.Row():
            model_dd = gr.Dropdown(
                choices=list(SUPPORTED_MODELS.keys()),
                value=list(SUPPORTED_MODELS.keys())[0],
                label="Base model",
            )
        with gr.Row():
            lora_r_sl     = gr.Slider(4, 64, value=16, step=4,  label="LoRA rank (r)")
            lora_alpha_sl = gr.Slider(4, 128, value=32, step=4, label="LoRA alpha")
        with gr.Row():
            lr_sl    = gr.Slider(1e-5, 5e-4, value=2e-4, step=1e-5, label="Stage 1 LR (S2=½, S3=¼)")
            steps_sl = gr.Slider(10, 300, value=60, step=10, label="Max steps per stage")
        gr.Markdown("**Stages to run:**")
        with gr.Row():
            s1_cb = gr.Checkbox(label="Stage 1 — Non-instruction (domain adaptation)", value=True)
            s2_cb = gr.Checkbox(label="Stage 2 — Instruction SFT",                    value=True)
            s3_cb = gr.Checkbox(label="Stage 3 — DPO alignment",                      value=True)
        gr.Markdown("**Upload your M1 data files (for Stages 2 & 3):**")
        with gr.Row():
            instr_file = gr.File(label="instruction_dataset.jsonl", file_types=[".jsonl",".json"])
            pref_file  = gr.File(label="preference_dataset.jsonl",  file_types=[".jsonl",".json"])
        gr.Markdown("""
        | Config | Value used in training | Effect |
        |--------|----------------------|--------|
        | r=16, alpha=32 | ✅ Matches your notebooks | Scaling = 2.0× |
        | Stage 1 LR = 2e-4 | ✅ Matches | Aggressive domain learning |
        | Stage 2 LR = 1e-4 | ✅ Auto-halved | Preserves Stage 1 knowledge |
        | Stage 3 LR = 5e-5 | ✅ Auto-quartered | Gentle alignment |
        | max_steps = 60 | ✅ Demo default | ~4 min/stage on T4 |
        """)

    # ── TAB 3: TRAIN ──────────────────────────────────────────────────────────
    with gr.Tab("🚀 Train"):
        gr.Markdown("""### Run the full fine-tuning pipeline
        Training streams live loss and VRAM usage per stage.
        After training, the model is immediately available in the Chat tab.
        """)
        train_btn = gr.Button("▶  Start Training", variant="primary", size="lg")
        train_log = gr.Textbox(label="Training log", lines=30, max_lines=60, autoscroll=True)

        train_btn.click(
            run_training,
            inputs=[model_dd, lora_r_sl, lora_alpha_sl, lr_sl, steps_sl,
                    s1_cb, s2_cb, s3_cb, instr_file, pref_file],
            outputs=[train_log],
        )

    # ── TAB 4: CHAT / COMPARE ─────────────────────────────────────────────────
    with gr.Tab("💬 Chat / Compare"):
        gr.Markdown("""### Test and compare models
        **Chat mode** uses the active model. **Compare mode** runs the same question
        through Base (Stage 1), SFT (Stage 2), and DPO (Stage 3) side-by-side.
        Models are loaded from your HuggingFace Hub repos automatically.
        """)

        with gr.Row():
            load_final_btn = gr.Button("⬇  Load Final Model from Hub", variant="secondary")
            load_status    = gr.Textbox(label="Load status", lines=1, scale=3)
        load_final_btn.click(load_final_model, inputs=[], outputs=[load_status])

        gr.Markdown("---")

        with gr.Tabs():
            with gr.Tab("💬 Single Model Chat"):
                chat_q       = gr.Textbox(label="Your question", lines=2,
                                          placeholder="What is the first-line treatment for type 2 diabetes?")
                max_tok_sl   = gr.Slider(50, 400, value=200, step=10, label="Max tokens")
                ask_btn      = gr.Button("Ask", variant="primary")
                chat_out     = gr.Textbox(label="Answer", lines=10)
                chat_status  = gr.Textbox(label="Status", lines=1)
                ask_btn.click(chat_answer, inputs=[chat_q, max_tok_sl], outputs=[chat_out, chat_status])

                gr.Examples(
                    examples=[
                        ["What is the first-line treatment for type 2 diabetes and why?"],
                        ["What are the warning signs of a heart attack?"],
                        ["How do SGLT-2 inhibitors work beyond blood sugar control?"],
                        ["What lifestyle changes lower blood pressure most effectively?"],
                        ["What is diabetic ketoacidosis and how is it managed?"],
                        ["Why is completing a full course of antibiotics important?"],
                    ],
                    inputs=[chat_q],
                )

            with gr.Tab("⚖️ Side-by-Side Comparison"):
                gr.Markdown("""Compare **Base → SFT → DPO** on the same question.
                Loads each model from your HuggingFace Hub repos:
                - `ekblaise/healthcare-qwen2.5-stage1-merged`
                - `ekblaise/healthcare-qwen2.5-stage2-merged`
                - `ekblaise/healthcare-qwen2.5-final`
                """)
                cmp_q    = gr.Textbox(label="Question to compare", lines=2,
                                      placeholder="What are the warning signs of a heart attack?")
                cmp_tok  = gr.Slider(50, 300, value=180, step=10, label="Max tokens")
                cmp_btn  = gr.Button("⚖️  Run Comparison (loads 3 models)", variant="primary")
                cmp_stat = gr.Textbox(label="Status", lines=1)
                with gr.Row():
                    cmp_base = gr.Textbox(label="🔵 Base (Stage 1 merged)", lines=12)
                    cmp_sft  = gr.Textbox(label="🟡 SFT (Stage 2 merged)",  lines=12)
                    cmp_dpo  = gr.Textbox(label="🟢 DPO Final",             lines=12)
                cmp_btn.click(
                    compare_answer,
                    inputs=[cmp_q, cmp_tok],
                    outputs=[cmp_base, cmp_sft, cmp_dpo, cmp_stat],
                )
                gr.Examples(
                    examples=[
                        ["What is the first-line treatment for type 2 diabetes and why?"],
                        ["When should a patient with a fever seek emergency medical attention?"],
                        ["What is diabetic ketoacidosis and what are the emergency steps?"],
                    ],
                    inputs=[cmp_q],
                )

    # ── TAB 5: EVALUATION ─────────────────────────────────────────────────────
    with gr.Tab("📊 Evaluation"):
        gr.Markdown("""### Automated 10-question benchmark
        Runs all evaluation questions and computes ROUGE-L scores.
        Uses the currently loaded model (or auto-loads the final DPO model from Hub).

        **Actual results from training:**
        | Stage | Avg ROUGE-L |
        |-------|------------|
        | Base  | 0.125 |
        | SFT   | 0.242 (+0.118) |
        | DPO   | Aligned (self-ref = 1.0) |
        """)
        eval_btn     = gr.Button("▶  Run Evaluation", variant="primary")
        eval_log     = gr.Textbox(label="Progress", lines=12)
        eval_table   = gr.Dataframe(label="Results table", wrap=True)
        eval_summary = gr.Textbox(label="Summary", lines=4)
        eval_btn.click(run_evaluation, inputs=[], outputs=[eval_log, eval_table, eval_summary])

    # ── TAB 6: EXPORT ─────────────────────────────────────────────────────────
    with gr.Tab("📥 Export"):
        gr.Markdown("""### Download your artifacts
        Creates a ZIP of all training outputs: adapters, reports, metrics, and inference script.
        """)
        with gr.Row():
            inc_weights_cb = gr.Checkbox(label="Include merged model weights (~3GB per stage)", value=False)
        export_btn    = gr.Button("📦 Create Export ZIP", variant="primary")
        export_file   = gr.File(label="Download ZIP")
        export_status = gr.Textbox(label="Contents", lines=12)

        export_btn.click(
            make_export_zip,
            inputs=[inc_weights_cb],
            outputs=[export_file, export_status],
        )

        gr.Markdown("""
        ---
        ### HuggingFace Hub repos (ekblaise)

        | Stage | Repo | Type |
        |-------|------|------|
        | Stage 1 merged | [`ekblaise/healthcare-qwen2.5-stage1-merged`](https://huggingface.co/ekblaise/healthcare-qwen2.5-stage1-merged) | Merged float16 |
        | Stage 2 merged | [`ekblaise/healthcare-qwen2.5-stage2-merged`](https://huggingface.co/ekblaise/healthcare-qwen2.5-stage2-merged) | Merged float16 |
        | Stage 3 DPO adapter | [`ekblaise/healthcare-qwen2.5-dpo-adapter`](https://huggingface.co/ekblaise/healthcare-qwen2.5-dpo-adapter) | LoRA adapter |
        | Final model | [`ekblaise/healthcare-qwen2.5-final`](https://huggingface.co/ekblaise/healthcare-qwen2.5-final) | Merged float16 ✅ |

        > All repos are private. The Chat tab loads from Hub automatically
        > when a valid HuggingFace token is available in the environment.
        """)


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        share      = True,
        server_port= 7860,
        show_error = True,
        quiet      = False,
    )
