"""Inference example. By default this will not download models.
Use --download to fetch the base model and a (possibly existing) adapter.
"""
import argparse


def run_inference(download: bool):
    if not download:
        print("inference.py: safe mode (no downloads). Use --download to fetch model and adapter.")
        return

    from unsloth import FastLanguageModel
    from peft import PeftModel

    MODEL_NAME = "unsloth/gemma-3-270m-it"
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=MODEL_NAME)

    # Example assumes adapter dir 'gemma-chess-lora' exists
    try:
        model_lora = PeftModel.from_pretrained(model, "gemma-chess-lora")
    except Exception as e:
        print("Could not load adapter 'gemma-chess-lora':", e)
        return

    prompt = [
        {"role": "system", "content": "Given an incomplete set of chess moves and the game's final score, write the last missing chess move."},
        {"role": "user", "content": '{"moves": ["e2e4","e7e5","g1f3","g8f6","f1c4","b8c6","?"], "result": "1-0"}'}
    ]

    input_ids = tokenizer.apply_chat_template(prompt, tokenize=True)
    outputs = model_lora.generate(**input_ids, max_new_tokens=10)
    print(tokenizer.decode(outputs[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download model and adapter for a demo")
    args = parser.parse_args()
    run_inference(args.download)
