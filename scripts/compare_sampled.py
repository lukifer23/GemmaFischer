#!/usr/bin/env python3
"""Generate sampled answers for the initial Q&A with base model and LoRA adapter, compute simple similarity, and write a report.

Outputs: comparison_sampling_report.md
"""
import re
import os
import sys
from datetime import datetime
import difflib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_MODEL_PATH = os.path.join(ROOT, "models/unsloth-gemma-3-270m-it/models--unsloth--gemma-3-270m-it/snapshots/23cf460f6bb16954176b3ddcc8d4f250501458a9")
ADAPTER_ROOT = os.path.join(ROOT, 'checkpoints', 'lora_full')
IN_MD = os.path.join(ROOT, 'initial_chess_q_and_a.md')
OUT_MD = os.path.join(ROOT, 'comparison_sampling_report.md')

SECTION_RE = re.compile(r"^##\s+Q\d+:\s*(.+)$", re.MULTILINE)

def parse_questions(md_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        src = f.read()
    parts = [p.strip() for p in src.split('---') if p.strip()]
    entries = []
    for p in parts:
        m = SECTION_RE.search(p)
        if not m:
            continue
        q = m.group(1).strip()
        entries.append(q)
    return entries

def latest_adapter_dir(root):
    if not os.path.isdir(root):
        return None
    subs = [d for d in os.listdir(root) if d.startswith('checkpoint-')]
    if not subs:
        if os.path.exists(os.path.join(root, 'adapter_model.safetensors')):
            return root
        return None
    def idx(s):
        try:
            return int(s.split('-')[-1])
        except:
            return 0
    subs.sort(key=idx)
    for s in reversed(subs):
        cand = os.path.join(root, s)
        if os.path.exists(os.path.join(cand, 'adapter_model.safetensors')) or os.path.exists(os.path.join(cand, 'adapter_config.json')):
            return cand
    return None

def generate_with_model(model, tokenizer, prompts, device, sampling_cfg):
    out = []
    model.eval()
    for p in prompts:
        inputs = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=sampling_cfg['max_new_tokens'], do_sample=True, top_p=sampling_cfg['top_p'], temperature=sampling_cfg['temperature'])
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        if text.startswith(p):
            text = text[len(p):].strip()
        out.append(text.strip())
    return out

def main():
    qs = parse_questions(IN_MD)
    if not qs:
        print('No questions found in', IN_MD)
        sys.exit(1)

    adapter_dir = latest_adapter_dir(ADAPTER_ROOT)
    print('Using adapter dir:', adapter_dir)

    print('Loading tokenizer and base model...')
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_PATH, local_files_only=True, device_map='auto', attn_implementation='eager')
    device = next(base.parameters()).device

    sampling_cfg = {'max_new_tokens': 200, 'top_p': 0.9, 'temperature': 0.8}

    print('Generating base answers...')
    base_answers = generate_with_model(base, tokenizer, qs, device, sampling_cfg)

    tuned_answers = []
    if adapter_dir:
        print('Applying adapter from', adapter_dir)
        model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_PATH, local_files_only=True, device_map='auto', attn_implementation='eager')
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
        tuned_answers = generate_with_model(model, tokenizer, qs, device, sampling_cfg)
    else:
        print('No adapter found; tuned answers will be empty')

    now = datetime.utcnow().isoformat() + 'Z'
    lines = []
    lines.append(f'# Sampling comparison report {now}\n')
    total_similarity = 0.0
    for i, q in enumerate(qs, start=1):
        base_ans = base_answers[i-1] if i-1 < len(base_answers) else ''
        tuned_ans = tuned_answers[i-1] if i-1 < len(tuned_answers) else ''
        ratio = difflib.SequenceMatcher(None, base_ans, tuned_ans).ratio() if tuned_ans else 0.0
        total_similarity += ratio
        lines.append(f'## Q{i}: {q}\n')
        lines.append('### Base (sampled):\n')
        lines.append(base_ans + '\n')
        lines.append('### Tuned (sampled):\n')
        lines.append(tuned_ans + '\n')
        lines.append(f'**Similarity ratio:** {ratio:.3f}\n')
        diff = list(difflib.unified_diff(base_ans.splitlines(), tuned_ans.splitlines(), fromfile='base', tofile='tuned', lineterm=''))
        if diff:
            lines.append('### Diff:\n')
            lines.append('```\n')
            lines.extend([L + '\n' for L in diff])
            lines.append('```\n')
        lines.append('---\n')

    avg_sim = total_similarity / len(qs) if qs else 0.0
    lines.insert(1, f'**Average similarity:** {avg_sim:.3f}\n')

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.writelines([l if l.endswith('\n') else l + '\n' for l in lines])

    print('Wrote', OUT_MD)

if __name__ == '__main__':
    main()
