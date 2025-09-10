#!/usr/bin/env python3
"""Quick integrity test: load base model and optional adapter, produce deterministic and sampled outputs for a few prompts."""
import argparse
import json
from pathlib import Path
import torch
from transformers import GemmaTokenizer


def gen(model, tokenizer, prompt, do_sample=False, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    gen = model.generate(**inputs, do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=0.9, temperature=0.8, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(gen[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--adapter_dir', type=str, default=None)
    args = parser.parse_args()

    tokenizer = GemmaTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True, device_map='auto', attn_implementation='eager')

    prompts = [
        "Q: What is the best opening move for White and why?\nA:",
        "Q: Explain the concept of zugzwang and give a short example.\nA:",
        "Q: Give three practical endgame tips for rook and pawn endgames.\nA:",
    ]

    out = {'base': []}
    for p in prompts:
        det = gen(model, tokenizer, p, do_sample=False, max_new_tokens=128)
        samp = gen(model, tokenizer, p, do_sample=True, max_new_tokens=128)
        out['base'].append({'prompt': p, 'deterministic': det, 'sampled': samp})

    if args.adapter_dir:
        from peft import PeftModel
        peft_path = args.adapter_dir
        # apply adapter
        model = PeftModel.from_pretrained(model, peft_path)
        out['tuned'] = []
        for p in prompts:
            det = gen(model, tokenizer, p, do_sample=False, max_new_tokens=128)
            samp = gen(model, tokenizer, p, do_sample=True, max_new_tokens=128)
            out['tuned'].append({'prompt': p, 'deterministic': det, 'sampled': samp})

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
