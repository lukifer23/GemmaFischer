#!/usr/bin/env python3
"""Create a small fine-tune JSONL dataset from `initial_chess_q_and_a.md`.
Each line will be a JSON object with a `text` field suitable for causal LM training:
"Question: <q>\nAnswer: <a>".
"""
import re
import json
from pathlib import Path

IN_MD = Path('initial_chess_q_and_a.md')
OUT_DIR = Path('data/finetune')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / 'chess_finetune.jsonl'

SECTION_RE = re.compile(r"^##\s+Q\d+:\s*(.+)$", re.MULTILINE)
ANSWER_HEADER = "### Model answer:"

if not IN_MD.exists():
    print(f"{IN_MD} not found. Run generate_chess_qa.py first.")
    raise SystemExit(1)

text = IN_MD.read_text(encoding='utf-8')
parts = [p.strip() for p in text.split('---') if p.strip()]
entries = []
for p in parts:
    m = SECTION_RE.search(p)
    if not m:
        continue
    q = m.group(1).strip()
    if ANSWER_HEADER in p:
        a = p.split(ANSWER_HEADER, 1)[1].strip()
    else:
        a = ""
    entries.append((q, a))

# Limit size for a quick POC
limit = 20
entries = entries[:limit]

with OUT_FILE.open('w', encoding='utf-8') as f:
    for q, a in entries:
        obj = {"text": f"Question: {q}\nAnswer: {a}"}
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Wrote {len(entries)} examples to {OUT_FILE}")
