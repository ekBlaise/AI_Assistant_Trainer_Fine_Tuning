"""
GPT-4 Data Generation Script
==============================
Uses the OpenAI API to generate additional instruction and preference pairs
from non-instruction corpus text.

SETUP:
    pip install openai
    export OPENAI_API_KEY="sk-your-key-here"

USAGE:
    python generate_with_gpt4.py --mode instruction --input non_instruction_data.txt --output new_pairs.jsonl --n 50
    python generate_with_gpt4.py --mode preference  --input instruction_dataset.jsonl  --output new_pref.jsonl  --n 30
"""

import os
import json
import time
import argparse
import random
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ─── Prompts ──────────────────────────────────────────────────────────────────

INSTRUCTION_SYSTEM = """You are a medical education expert creating training data for a healthcare AI assistant.
Given a medical text passage, generate high-quality instruction-output pairs.
Each output must be:
- Clinically accurate and evidence-based
- Written in clear, accessible language for patients and clinicians
- Between 80-200 words
- Safe — include appropriate caveats for medical advice

Return ONLY valid JSON — a list of objects with keys: instruction, input, output.
No preamble, no markdown fences, no explanation. Pure JSON array only."""

INSTRUCTION_USER = """From this medical text, generate {n} diverse instruction-output pairs covering
different aspects (symptoms, treatment, prevention, mechanism, patient education):

TEXT:
{passage}

Return exactly {n} JSON objects as a JSON array."""

PREFERENCE_SYSTEM = """You are a medical AI safety expert creating preference training data.
Given a medical question, generate one high-quality (chosen) and one low-quality (rejected) answer.

The CHOSEN answer must be: accurate, evidence-based, appropriately cautious, and recommend 
professional consultation where appropriate.

The REJECTED answer must be: plausible-sounding but contain at least one of: 
dangerous advice, oversimplification, missing important safety caveat, or factual error.

Return ONLY valid JSON with keys: prompt, chosen, rejected. No markdown, no explanation."""

PREFERENCE_USER = """Medical question: {question}

Generate one chosen (high quality, safe, accurate) response and one rejected 
(unsafe, oversimplified, or factually wrong) response.
Return as a single JSON object."""

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_passages(path: str, max_passages: int = 100) -> list[str]:
    """Load text passages from a plain text file."""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    passages = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 100]
    random.shuffle(passages)
    return passages[:max_passages]

def load_instructions(path: str, max_items: int = 100) -> list[str]:
    """Load instruction questions from a JSONL file."""
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                questions.append(obj.get("instruction", ""))
    random.shuffle(questions)
    return questions[:max_items]

def call_gpt4(system: str, user: str, retries: int = 3) -> str:
    """Call GPT-4 with retry logic."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",           # or "gpt-4-turbo" or "gpt-4"
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.7,
                max_tokens=1500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return ""

def safe_parse_json(text: str) -> list | dict | None:
    """Parse JSON, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON within the text
        import re
        match = re.search(r'[\[{].*[\]}]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None

# ─── Generation functions ─────────────────────────────────────────────────────

def generate_instruction_pairs(
    input_path: str,
    output_path: str,
    n_total: int = 50,
    pairs_per_call: int = 3,
) -> None:
    """Generate instruction-output pairs from raw text passages."""
    passages = load_passages(input_path)
    print(f"Loaded {len(passages)} passages. Generating ~{n_total} instruction pairs...")

    records = []
    calls_needed = (n_total + pairs_per_call - 1) // pairs_per_call

    for i, passage in enumerate(passages[:calls_needed]):
        print(f"  Call {i+1}/{calls_needed}...", end=" ", flush=True)
        user_msg = INSTRUCTION_USER.format(passage=passage[:1000], n=pairs_per_call)
        raw = call_gpt4(INSTRUCTION_SYSTEM, user_msg)
        parsed = safe_parse_json(raw)

        if parsed and isinstance(parsed, list):
            for item in parsed:
                if all(k in item for k in ["instruction", "output"]):
                    records.append({
                        "instruction": item["instruction"].strip(),
                        "input":       item.get("input", "").strip(),
                        "output":      item["output"].strip(),
                        "source":      "gpt4_generated",
                    })
            print(f"✓ ({len(parsed)} pairs)")
        else:
            print("✗ (parse failed)")

        time.sleep(0.5)   # Rate limit courtesy pause

        if len(records) >= n_total:
            break

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records[:n_total]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {min(len(records), n_total)} instruction pairs to {output_path}")

def generate_preference_pairs(
    input_path: str,
    output_path: str,
    n_total: int = 30,
) -> None:
    """Generate chosen/rejected preference pairs from existing questions."""
    questions = load_instructions(input_path, max_items=n_total * 2)
    print(f"Loaded {len(questions)} questions. Generating {n_total} preference pairs...")

    records = []
    for i, question in enumerate(questions):
        if len(records) >= n_total:
            break
        print(f"  Call {i+1}...", end=" ", flush=True)
        user_msg = PREFERENCE_USER.format(question=question)
        raw = call_gpt4(PREFERENCE_SYSTEM, user_msg)
        parsed = safe_parse_json(raw)

        if parsed and isinstance(parsed, dict):
            if all(k in parsed for k in ["prompt", "chosen", "rejected"]):
                records.append({
                    "prompt":   parsed["prompt"].strip(),
                    "chosen":   parsed["chosen"].strip(),
                    "rejected": parsed["rejected"].strip(),
                })
                print("✓")
            else:
                # Try using original question as prompt
                if "chosen" in parsed and "rejected" in parsed:
                    records.append({
                        "prompt":   question,
                        "chosen":   parsed["chosen"].strip(),
                        "rejected": parsed["rejected"].strip(),
                    })
                    print("✓ (prompt from original)")
                else:
                    print("✗ (missing keys)")
        else:
            print("✗ (parse failed)")

        time.sleep(0.5)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records[:n_total]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {min(len(records), n_total)} preference pairs to {output_path}")

# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate training data with GPT-4")
    parser.add_argument("--mode",   required=True,  choices=["instruction", "preference"],
                        help="What type of data to generate")
    parser.add_argument("--input",  required=True,  help="Input file path")
    parser.add_argument("--output", required=True,  help="Output JSONL file path")
    parser.add_argument("--n",      type=int, default=50,
                        help="Number of pairs to generate (default: 50)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        print("  export OPENAI_API_KEY='sk-your-key-here'")
        return

    if args.mode == "instruction":
        generate_instruction_pairs(args.input, args.output, n_total=args.n)
    else:
        generate_preference_pairs(args.input, args.output, n_total=args.n)

    print("\nDone. Merge new data with existing datasets:")
    print(f"  cat instruction_dataset.jsonl {args.output} > merged_instruction.jsonl")

if __name__ == "__main__":
    main()
