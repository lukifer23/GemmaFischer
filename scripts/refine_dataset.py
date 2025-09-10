#!/usr/bin/env python3
"""Refine dataset templates for better Q&A alignment."""
import json
from pathlib import Path

input_file = Path("data/finetune/chess_finetune_full.jsonl")
output_file = Path("data/finetune/chess_finetune_refined.jsonl")

with open(input_file, 'r') as f:
    lines = f.readlines()

refined = []
for line in lines:
    data = json.loads(line.strip())
    text = data['text']
    # Extract question and answer
    q_start = text.find("Question: ") + len("Question: ")
    a_start = text.find("Answer: ")
    question = text[q_start:a_start].strip()
    answer = text[a_start + len("Answer: "):].strip()
    
    # New template
    new_text = f"Question: Given the chess moves {question}, what is the missing move and why? Answer: The missing move is {answer}, because it completes the sequence logically."
    refined.append({"text": new_text})

with open(output_file, 'w') as f:
    for item in refined:
        f.write(json.dumps(item) + '\n')

print(f"Refined dataset saved to {output_file}")
