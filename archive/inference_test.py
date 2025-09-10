#!/usr/bin/env python3
"""Tiny inference test: load local snapshot and generate from prompt "Hello".

This script uses the local snapshot path downloaded earlier. It prints progress and a short generated answer.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys

MODEL_PATH = "models/unsloth-gemma-3-270m-it/models--unsloth--gemma-3-270m-it/snapshots/23cf460f6bb16954176b3ddcc8d4f250501458a9"

print("Using Python:", sys.executable)
print("Model path:", MODEL_PATH)

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    print("Tokenizer loaded. Loading model (this may take a few seconds)...")

    # Try to use device_map to place model on MPS if available; fall back to cpu on error
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True, device_map="auto")
    except Exception as e:
        print("device_map loading failed, falling back to CPU. Error:", repr(e))
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True, device_map=None)

    device = next(model.parameters()).device
    print("Model loaded on device:", device)

    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt")
    # move inputs to model device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n=== GENERATED TEXT ===")
    print(text)
    print("=== END ===\n")

except Exception as e:
    print("ERROR during inference test:", repr(e))
    raise
