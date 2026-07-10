#!/usr/bin/env python3
"""
inference.py — Healthcare FAQ Assistant Inference Script
=========================================================
Loads the final DPO-aligned model and answers healthcare questions.

Usage:
    python inference.py
    python inference.py --question "What is metformin?"
    python inference.py --question "What is metformin?" --max_tokens 300
    python inference.py --model_path /path/to/final_merged

Requirements:
    pip install unsloth transformers==4.56.2 torch
"""

import argparse
import re
import sys
import warnings
warnings.filterwarnings("ignore")

import torch

SYSTEM_PROMPT = (
    "You are a knowledgeable and empathetic healthcare assistant. "
    "Provide accurate, evidence-based answers to medical questions. "
    "Always recommend consulting a qualified healthcare professional "
    "for personal medical advice."
)

SAFETY_PATTERNS = [
    r"\b(take|prescribe|dose|dosage|mg|milligram|inject|administer)\b",
    r"\b(diagnosis|diagnose|you have|you are suffering)\b",
    r"\b(stop|discontinue|increase|decrease)\s+your\s+\w+",
]

SAFETY_DISCLAIMER = (
    "\n\n---\n*Please consult a qualified healthcare professional before "
    "making any medical decisions. This information is for educational purposes only.*"
)

def needs_disclaimer(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in SAFETY_PATTERNS)


def load_model(model_path: str, max_seq_length: int = 512):
    """Load the fine-tuned model."""
    try:
        import unsloth  # noqa
        from unsloth import FastLanguageModel
        print(f"Loading model from: {model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    except Exception:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Falling back to standard HuggingFace loading...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def answer(
    question: str,
    model,
    tokenizer,
    max_new_tokens: int = 200,
    temperature: float = 0.3,
    add_safety: bool = True,
) -> str:
    """Generate an answer for a question."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question.strip()},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    in_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(output[0][in_len:], skip_special_tokens=True).strip()
    if add_safety and needs_disclaimer(response):
        response += SAFETY_DISCLAIMER
    return response


def main():
    parser = argparse.ArgumentParser(description="Healthcare FAQ Assistant")
    parser.add_argument(
        "--model_path", type=str,
        default="/content/healthcare_assistant/final_merged",
        help="Path to the final merged model directory",
    )
    parser.add_argument("--question",   type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--no_safety",  action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)
    print("\n✅ Healthcare FAQ Assistant ready.")

    print("Interactive mode — type your question and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    # If --question was passed, answer it first, then stay in the loop
    if args.question:
        response = answer(args.question, model, tokenizer,
                          max_new_tokens=args.max_tokens,
                          add_safety=not args.no_safety)
        print(f"Question: {args.question}")
        print(f"\nAnswer: {response}\n")
        print("-" * 60)

        while True:
            try:
                question = input("Question: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
            if question.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break
            if not question:
                continue
            response = answer(question, model, tokenizer,
                              max_new_tokens=args.max_tokens,
                              add_safety=not args.no_safety)
            print(f"\nAnswer: {response}\n")
            print("-" * 60)


if __name__ == "__main__":
    main()