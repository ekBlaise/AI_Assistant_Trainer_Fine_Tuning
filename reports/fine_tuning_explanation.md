# Fine-Tuning Explanation Report

## Healthcare FAQ Assistant — Three-Stage Pipeline

---

## 1. What is fine-tuning and why was it used?

Fine-tuning adapts a pre-trained language model to a specific domain or task by
continuing training on domain-relevant data. Rather than training from scratch
(which requires billions of tokens and weeks of compute), fine-tuning leverages
the general knowledge already encoded in the base model and redirects its
capabilities toward a specific use case — in this case, healthcare FAQ answering.

Fine-tuning was chosen over prompt engineering alone because the domain requires
consistent clinical precision, specific vocabulary, and a reliable response format
that few-shot prompting cannot reliably achieve across diverse questions.

---

## 2. What is QLoRA and why was it used?

QLoRA (Quantized Low-Rank Adaptation) combines two techniques:

**4-bit quantization (bitsandbytes):** Model weights are stored in 4-bit NF4 format
instead of 16-bit floats, reducing memory from ~3GB to ~700MB for a 1.5B model.
During computation, weights are dequantized to float16 on-the-fly.

**LoRA (Low-Rank Adaptation):** Instead of updating all model parameters, small
low-rank matrices A and B are injected into target layers. The update is
W_new = W_original + B×A×(alpha/r). Only A and B are trained (~0.38% of parameters).

In this project: r=16, alpha=32, targeting 7 layer types.
QLoRA made training feasible on a free Colab T4 GPU (15GB VRAM) that would
otherwise be unable to fine-tune even a 1.5B model.

---

## 3. What is the three-stage pipeline and why this order?

**Stage 1 — Non-instruction pretraining (Domain Adaptation):**
Trains on 110 raw healthcare paragraphs using causal language modelling.
The model learns healthcare vocabulary, clinical writing patterns, and domain facts.
Like immersing someone in medical literature before teaching them to answer questions.
LR: 2e-4 | packing=True | 110 paragraphs

**Stage 2 — Instruction fine-tuning (SFT):**
Trains on 296 instruction-response pairs. Teaches the model the question-answer
format using apply_chat_template (ChatML special tokens). Response-only loss
(DataCollatorForCompletionOnlyLM) ensures gradients only flow from response tokens.
LR: 1e-4 (lower to preserve Stage 1 knowledge) | 296 pairs

**Stage 3 — DPO preference alignment:**
Trains on 100 chosen/rejected pairs. Teaches the model to prefer safe, accurate,
well-structured answers over vague, dangerous, or incorrect ones using Direct
Preference Optimization (DPO) without a separate reward model.
LR: 5e-5 (very gentle) | β=0.1 | 100 preference pairs

The order is critical: you must learn the domain before you can follow instructions
about it, and you must follow instructions before you can be aligned on quality.

---

## 4. What is DPO and how does it differ from RLHF?

RLHF (Reinforcement Learning from Human Feedback) trains a separate reward model
on human preferences, then uses PPO (a complex RL algorithm) to optimize the
policy against that reward model. This requires three models simultaneously
(policy, reference, reward) and complex training dynamics.

DPO (Direct Preference Optimization) simplifies this by deriving the optimal
policy directly from preference pairs without a separate reward model. The loss
function directly increases the log-probability of chosen responses relative to
rejected ones, controlled by the β parameter (0.1 in this project).

DPO was chosen because it is simpler, more stable, uses less memory, and
achieves comparable or better alignment than PPO for most tasks.

---

## 5. What is the beta parameter in DPO?

Beta (β=0.1) controls the KL divergence penalty in DPO — how far the
aligned model is allowed to deviate from the reference model (Stage 2 model).

Low β (e.g. 0.1): Gentle alignment, model stays close to SFT behavior.
High β (e.g. 1.0): Strong alignment, model changes more aggressively.

β=0.1 was chosen as the standard starting value from the original DPO paper,
appropriate for a medical domain where preserving the SFT model's clinical
knowledge is important — aggressive alignment risks overwriting it.

---

## 6. What is LoRA rank and how was it chosen?

LoRA rank (r=16) is the inner dimension of the A and B adapter matrices.
Higher rank = more expressive adapter = more trainable parameters = more VRAM.

With r=16 and d_model=2048: each LoRA layer adds 2×(2048×16)=65,536 parameters.
Across 7 target module types × model depth: ~0.38% of total parameters trainable.

r=16 was chosen as the standard starting configuration validated across
many fine-tuning papers. r=8 would be faster; r=32 more expressive but slower.

---

## 7. What is sequence packing and why was it used in Stage 1 only?

Sequence packing concatenates multiple training examples into one fixed-length
block (512 tokens here), eliminating the padding tokens that waste GPU compute.
Without packing: a 100-token paragraph is padded to 512 tokens — 80% waste.
With packing: multiple paragraphs fill one 512-token block — ~0% waste.

Packing was used in Stage 1 (raw text paragraphs) where boundary blurring
between paragraphs is harmless — the model just predicts the next token.

Packing was disabled in Stage 2 (instruction pairs) because packing can blur
the boundary between one answer and the next instruction, confusing the model
about where one conversation ends and another begins.

---

## 8. Why is a decreasing learning rate used across stages?

| Stage | LR | Rationale |
|-------|-----|-----------|
| Stage 1 | 2e-4 | LoRA adapters start at zero, need aggressive updates to learn domain |
| Stage 2 | 1e-4 | Lower LR preserves Stage 1 domain knowledge while teaching instruction format |
| Stage 3 | 5e-5 | Very gentle: DPO only nudges response quality, must not erase SFT |

This cascade pattern (2e-4 → 1e-4 → 5e-5) is the standard for multi-stage
cascaded fine-tuning, validated across many production fine-tuning projects.

---

## 9. What production improvements were made beyond the rubric?

| Improvement | Standard Approach | This Project |
|-------------|------------------|--------------|
| Prompt format | Alpaca ### headers | apply_chat_template (ChatML) |
| Loss masking | Full sequence | Response-only (DataCollatorForCompletionOnlyLM) |
| Validation | None | 85/15 split, eval every 10 steps |
| Evaluation | Qualitative only | ROUGE-L automated scoring |
| DPO kernels | Standard TRL | PatchDPOTrainer (Unsloth optimised) |
| Safety | None | Medical disclaimer layer in inference |
| Deployment | Notebook only | inference.py CLI + Gradio web app |
| Tracking | Print statements | Weights & Biases (optional) |