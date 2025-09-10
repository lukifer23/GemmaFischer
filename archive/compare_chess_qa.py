#!/usr/bin/env python3
"""Compare initial chess Q&A with current model outputs and write a comparison markdown.

Reads `initial_chess_q_and_a.md`, extracts questions, re-queries the local model snapshot, and writes
`comparison_after_training.md` containing the original answer, current answer, and a simple unified diff.
"""
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from datetime import datetime
import difflib
import sys

DEFAULT_MODEL_PATH = "models/unsloth-gemma-3-270m-it/models--unsloth--gemma-3-270m-it/snapshots/23cf460f6bb16954176b3ddcc8d4f250501458a9"
IN_MD = "initial_chess_q_and_a.md"
OUT_MD = "comparison_after_training.md"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                    help='Path to model snapshot or fine-tuned checkpoint (optional)')
parser.add_argument('--out_md', type=str, default=OUT_MD, help='Output markdown file')
args = parser.parse_args()

MODEL_PATH = args.model_path
OUT_MD = args.out_md

# Parse the initial md file format produced by generate_chess_qa.py
# Each entry looks like:
# ## Q1: <question>
#
# ### Model answer:
#
# <answer paragraphs>
#
# ---

SECTION_RE = re.compile(r"^##\s+Q\d+:\s*(.+)$", re.MULTILINE)
ANSWER_HEADER = "### Model answer:"

try:
    with open(IN_MD, "r", encoding="utf-8") as f:
        src = f.read()
except FileNotFoundError:
    print(f"Input file {IN_MD} not found. Run generate_chess_qa.py first.")
    sys.exit(1)

# Split sections by the '---' divider
parts = [p.strip() for p in src.split('---') if p.strip()]
entries = []
for p in parts:
    m = SECTION_RE.search(p)
    if not m:
        continue
    question = m.group(1).strip()
    # find answer header
    if ANSWER_HEADER in p:
        ans = p.split(ANSWER_HEADER, 1)[1].strip()
    else:
        ans = ""
    # Clean up answer text
    ans = ans.strip()
    entries.append((question, ans))

if not entries:
    print("No Q&A entries parsed from", IN_MD)
    sys.exit(1)

print(f"Parsed {len(entries)} questions from {IN_MD}")

# Load tokenizer and model
print("Loading tokenizer and model from:", MODEL_PATH)
try:
    # If MODEL_PATH points to a PEFT checkpoint root (like checkpoints/lora_poc),
    # look for checkpoint-* subdirs and pick the latest one that contains adapter files.
    import os
    adapter_dir = None
    if os.path.isdir(MODEL_PATH):
        # direct adapter/PEFT dir
        if os.path.exists(os.path.join(MODEL_PATH, 'adapter_config.json')) or os.path.exists(os.path.join(MODEL_PATH, 'adapter_model.safetensors')):
            adapter_dir = MODEL_PATH
        else:
            # look for checkpoint-* subdirs
            subs = [d for d in os.listdir(MODEL_PATH) if d.startswith('checkpoint-')]
            if subs:
                # pick highest numeric suffix
                def idx(s):
                    try:
                        return int(s.split('-')[-1])
                    except Exception:
                        return 0
                subs.sort(key=idx)
                for s in reversed(subs):
                    cand = os.path.join(MODEL_PATH, s)
                    if os.path.exists(os.path.join(cand, 'adapter_config.json')) or os.path.exists(os.path.join(cand, 'adapter_model.safetensors')):
                        adapter_dir = cand
                        break

    if adapter_dir:
        print("Detected PEFT adapter at:", adapter_dir)
        # Try tokenizer from adapter_dir, else fallback to base snapshot
        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH, local_files_only=True)

        # load base model then apply adapter
        base_model_path = DEFAULT_MODEL_PATH
        print("Loading base model from:", base_model_path)
        model = AutoModelForCausalLM.from_pretrained(base_model_path, local_files_only=True, device_map='auto')
        print("Applying PEFT adapter from:", adapter_dir)
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    else:
        # fallback: try loading as a full model snapshot
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True, device_map="auto")

    device = next(model.parameters()).device
    print("Model loaded on device:", device)
except Exception as e:
    print("Error loading model:", repr(e))
    sys.exit(1)


def generate_answer(prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        return text[len(prompt):].strip()
    return text.strip()

now = datetime.utcnow().isoformat() + "Z"
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(f"# Comparison of initial and current model answers generated on {now}\n\n")
    for i, (q, orig_ans) in enumerate(entries, start=1):
        print(f"Generating current answer for Q{i}")
        try:
            cur_ans = generate_answer(q)
        except Exception as e:
            cur_ans = f"ERROR generating answer: {repr(e)}"

        f.write(f"## Q{i}: {q}\n\n")
        f.write("### Original model answer:\n\n")
        for para in orig_ans.split('\n\n'):
            f.write(para.strip() + "\n\n")
        f.write("### Current model answer:\n\n")
        for para in cur_ans.split('\n\n'):
            f.write(para.strip() + "\n\n")

        # produce a small unified diff
        orig_lines = orig_ans.splitlines()
        cur_lines = cur_ans.splitlines()
        diff = list(difflib.unified_diff(orig_lines, cur_lines, fromfile='original', tofile='current', lineterm=''))
        if diff:
            f.write("### Unified diff:\n\n```")
            f.write("\n")
            for L in diff:
                f.write(L + "\n")
            f.write("```\n\n")
        else:
            f.write("### Unified diff: (no differences detected)\n\n")

        f.write("---\n\n")

print("Wrote", OUT_MD)
