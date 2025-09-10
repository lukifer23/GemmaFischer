#!/usr/bin/env python3
"""Simple inference test for adapter evaluation."""
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--adapter_dir', type=str, default=None)
    parser.add_argument('--prompt', type=str, required=True)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True, device_map='auto', attn_implementation='eager')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    if args.adapter_dir:
        print(f"Loading adapter from {args.adapter_dir}")
        model = PeftModel.from_pretrained(model, args.adapter_dir)

    print(f"Prompt: {args.prompt}")
    inputs = tokenizer(args.prompt, return_tensors='pt').to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, top_p=0.9, temperature=0.8, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Response: {response}")

if __name__ == "__main__":
    main()
