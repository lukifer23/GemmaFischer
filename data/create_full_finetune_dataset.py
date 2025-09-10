#!/usr/bin/env python3
"""Create a full fine-tune JSONL dataset from `data/processed/chess_conversations.json`.
Handles several common schemas and writes lines of JSON with a `text` field suitable for causal LM
training: "Question: <q>\nAnswer: <a>".

Usage: python create_full_finetune_dataset.py [--max_examples N]
"""
import json
import ast
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--max_examples', type=int, default=0, help='Limit number of examples (0 = all)')
args = parser.parse_args()

IN_FILE = Path('data/processed/chess_conversations.json')
OUT_DIR = Path('data/finetune')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / 'chess_finetune_full.jsonl'

if not IN_FILE.exists():
    print(f"Input file {IN_FILE} not found. Run data/prepare_dataset.py to produce it.")
    raise SystemExit(1)

file_text = IN_FILE.read_text(encoding='utf-8')

# If file is a fenced code block or contains python-style dicts (single quotes),
# try to extract top-level Python literals safely. Otherwise fall back to json.loads.
text = file_text.strip()
objs = []
if text.startswith('```'):
    # remove code fence lines
    lines = file_text.splitlines()
    if lines and lines[0].strip().startswith('```'):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith('```'):
        lines = lines[:-1]
    text = "\n".join(lines)

# scan for top-level dict/list literals and parse them with ast.literal_eval
i = 0
n = len(text)
while i < n:
    # skip whitespace
    while i < n and text[i].isspace():
        i += 1
    if i >= n:
        break
    if text[i] not in '{[':
        # skip to next line
        nl = text.find('\n', i)
        if nl == -1:
            break
        i = nl + 1
        continue
    start = i
    stack = []
    in_str = False
    str_char = ''
    esc = False
    j = i
    while j < n:
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == str_char:
                in_str = False
        else:
            if ch == '"' or ch == "'":
                in_str = True
                str_char = ch
            elif ch in '{[':
                stack.append(ch)
            elif ch in '}]':
                if not stack:
                    break
                stack.pop()
        j += 1
        if not stack:
            break
    obj_str = text[start:j]
    try:
        py_obj = ast.literal_eval(obj_str)
        objs.append(py_obj)
    except Exception:
        # try JSON fallback by normalizing quotes
        try:
            js = obj_str.replace("'", '"')
            py_obj = json.loads(js)
            objs.append(py_obj)
        except Exception:
            # give up on this chunk
            pass
    i = j + 1

# decide what raw should be
if not objs:
    try:
        raw = json.loads(file_text)
    except Exception as e:
        print(f"Failed to parse input file as JSON: {e}")
        raw = []
elif len(objs) == 1:
    raw = objs[0]
else:
    raw = objs

# normalize to a list of conversation objects
items = raw if isinstance(raw, list) else raw.get('conversations') if isinstance(raw, dict) and 'conversations' in raw else []
if not items and isinstance(raw, dict):
    # maybe dict of id->entry
    try:
        items = list(raw.values())
    except Exception:
        items = []

examples = []

for entry in items:
    # attempt several common shapes
    q = None
    a = None
    if isinstance(entry, dict):
        # direct keys
        if 'question' in entry and 'answer' in entry:
            q = entry['question']
            a = entry['answer']
        elif 'instruction' in entry and ('output' in entry or 'response' in entry or 'answer' in entry):
            q = entry.get('instruction')
            a = entry.get('output') or entry.get('response') or entry.get('answer')
        elif 'conversations' in entry and isinstance(entry['conversations'], list):
            # extract user/assistant pairs
            conv = entry['conversations']
            # pair consecutive user->assistant
            for i in range(len(conv)-1):
                if conv[i].get('from') in ('user', 'User', 'human') or conv[i].get('role')=='user':
                    if conv[i+1].get('from') in ('assistant', 'Assistant') or conv[i+1].get('role')=='assistant':
                        # support several keys for the content field
                        q = conv[i].get('value') or conv[i].get('text') or conv[i].get('content')
                        a = conv[i+1].get('value') or conv[i+1].get('text') or conv[i+1].get('content')
                        if q and a:
                            examples.append((q.strip(), a.strip()))
            continue
        # some datasets store the raw input/expected_output at top-level
        elif 'input' in entry and 'expected_output' in entry:
            q = entry.get('input')
            a = entry.get('expected_output')
        elif 'text' in entry:
            # best-effort split: assume "Question:...\nAnswer:..."
            txt = entry['text']
            if '\nAnswer:' in txt:
                parts = txt.split('\nAnswer:', 1)
                q = parts[0].replace('Question:', '').strip()
                a = parts[1].strip()
        else:
            # try any fields that look promising
            for k in ('prompt','query','input'):
                if k in entry:
                    q = entry[k]
            for k in ('completion','target','output'):
                if k in entry:
                    a = entry[k]
    # append if we collected q and a
    if q and a:
        examples.append((str(q).strip(), str(a).strip()))

# fallback: if no structured items, try to parse raw lines if it's a list of strings
if not examples and isinstance(raw, list):
    for s in raw:
        if isinstance(s, str) and '\nAnswer:' in s:
            parts = s.split('\nAnswer:', 1)
            q = parts[0].replace('Question:', '').strip()
            a = parts[1].strip()
            examples.append((q, a))

if args.max_examples>0:
    examples = examples[:args.max_examples]

with OUT_FILE.open('w', encoding='utf-8') as f:
    for q,a in examples:
        obj = {"text": f"Question: {q}\nAnswer: {a}"}
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')

print(f"Wrote {len(examples)} examples to {OUT_FILE}")
