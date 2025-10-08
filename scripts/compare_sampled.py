#!/usr/bin/env python3
"""Generate sampled answers for the initial Q&A with base model and LoRA adapter, compute simple similarity, and write a report.

Outputs: comparison_sampling_report.md
"""
import re
import os
import sys
from datetime import datetime
import difflib
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_MODEL_REF = "google/gemma-3-270m"
ADAPTER_ROOT = os.path.join(ROOT, 'checkpoints', 'lora_uci')
IN_MD = os.path.join(ROOT, 'initial_chess_q_and_a.md')
REPORT_DIR = os.path.join(ROOT, 'reports')
OUT_MD = os.path.join(REPORT_DIR, 'compare_sampling.md')

SECTION_RE = re.compile(r"^##\s+Q\d+:\s*(.+)$", re.MULTILINE)
DEFAULT_QUESTIONS = [
    "FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\nWhat is the best move?",
    "FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\nWhat is the best move?",
    "FEN: rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2\nWhat is the best move?",
    "FEN: rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 1 2\nWhat is the best move?",
    "FEN: rnbqkbnr/ppp2ppp/8/3pp3/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 0 3\nWhat is the best move?"
]

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
    qs = []
    questions_path = Path(IN_MD)
    if not questions_path.exists():
        archive_path = Path(ROOT) / 'archive' / 'initial_chess_q_and_a.md'
        if archive_path.exists():
            questions_path = archive_path
        else:
            print(f'Warning: {IN_MD} not found. Using built-in evaluation prompts.')
            qs = DEFAULT_QUESTIONS
    if questions_path.exists():
        qs = parse_questions(str(questions_path))
        if not qs:
            print(f'Warning: no questions parsed from {questions_path}. Falling back to defaults.')
            qs = DEFAULT_QUESTIONS

    adapter_dir = latest_adapter_dir(ADAPTER_ROOT)
    print('Using adapter dir:', adapter_dir)

    env_path = os.environ.get("CHESSGEMMA_MODEL_PATH")
    env_model_id = os.environ.get("CHESSGEMMA_MODEL_ID")
    if env_path:
        model_ref = env_path
    elif env_model_id:
        model_ref = env_model_id
    else:
        local_dir = os.path.join(ROOT, "models", "google-gemma-3-270m")
        model_ref = local_dir if os.path.exists(local_dir) else DEFAULT_MODEL_REF

    path_obj = Path(model_ref)
    using_local = path_obj.exists()
    load_target = str(path_obj) if using_local else model_ref

    print(f'Loading tokenizer and base model from {load_target} (local={using_local}) ...')
    tokenizer = AutoTokenizer.from_pretrained(load_target, local_files_only=using_local, trust_remote_code=True)

    import torch
    torch_dtype = torch.float16
    device_map = 'auto'
    if torch.backends.mps.is_available():
        torch_dtype = torch.float32
        device_map = None

    base = AutoModelForCausalLM.from_pretrained(
        load_target,
        local_files_only=using_local,
        device_map=device_map,
        attn_implementation='eager',
        trust_remote_code=True,
        torch_dtype=torch_dtype
    )
    device = next(base.parameters()).device

    sampling_cfg = {'max_new_tokens': 200, 'top_p': 0.9, 'temperature': 0.8}

    print('Generating base answers...')
    base_answers = generate_with_model(base, tokenizer, qs, device, sampling_cfg)

    tuned_answers = []
    if adapter_dir:
        print('Applying adapter from', adapter_dir)
        model = AutoModelForCausalLM.from_pretrained(
            load_target,
            local_files_only=using_local,
            device_map=device_map,
            attn_implementation='eager',
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )
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

    out_path = Path(OUT_MD)
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        f.writelines([l if l.endswith('\n') else l + '\n' for l in lines])

    print('Wrote', out_path)

if __name__ == '__main__':
    main()
