#!/usr/bin/env python3
"""Generate several chess Q&A pairs using the local model snapshot and save to a markdown file.

Writes `initial_chess_q_and_a.md` with numbered questions and model answers.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
from datetime import datetime

MODEL_PATH = "models/unsloth-gemma-3-270m-it/models--unsloth--gemma-3-270m-it/snapshots/23cf460f6bb16954176b3ddcc8d4f250501458a9"
OUT_MD = "initial_chess_q_and_a.md"

questions = [
    "What is the best opening move for White and why?",
    "Explain the concept of zugzwang and give a short example.",
    "How should a player evaluate material versus initiative in the middlegame?",
    "What are common mating patterns to watch for when the opponent's king is in the center?",
    "Give three practical endgame tips for rook and pawn endgames."
]

print("Loading tokenizer and model from:", MODEL_PATH)
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True, device_map="auto")
    device = next(model.parameters()).device
    print("Loaded model on device:", device)
except Exception as e:
    print("Error loading model:", repr(e))
    sys.exit(1)

def generate_answer(prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # If the model echoes the prompt, strip it
    if text.startswith(prompt):
        return text[len(prompt):].strip()
    return text.strip()

now = datetime.utcnow().isoformat() + "Z"
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(f"# Initial chess Q&A generated on {now}\n\n")
    for i, q in enumerate(questions, start=1):
        print(f"Generating answer for question {i}/{len(questions)}")
        try:
            ans = generate_answer(q)
        except Exception as e:
            ans = f"ERROR generating answer: {repr(e)}"
        f.write(f"## Q{i}: {q}\n\n")
        f.write("### Model answer:\n\n")
        # Ensure markdown-friendly paragraphing
        for para in ans.split("\n\n"):
            f.write(para.strip() + "\n\n")
        f.write("---\n\n")

print("Wrote", OUT_MD)
