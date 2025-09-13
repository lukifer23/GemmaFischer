"""Prepare the Thytu/ChessInstruct dataset into chat-style `conversations` format.
This script only transforms and saves a small sample by default to avoid heavy downloads.
Run with --full to process entire dataset (downloads ~100k examples).
"""
import argparse
from pathlib import Path
import json
from itertools import islice


def convert_to_chat_format(example):
    return {
        "conversations": [
            {"role": "system", "content": example["task"]},
            {"role": "user", "content": str(example["input"])},
            {"role": "assistant", "content": example["expected_output"]},
        ]
    }


def convert_and_save(output_dir: str, full: bool = False):
    from datasets import load_dataset

    ds_name = "Thytu/ChessInstruct"
    print(f"Loading dataset {ds_name} (this may download files)...")

    if full:
        ds = load_dataset(ds_name, split="train")
        print("Dataset loaded, example count:", len(ds))
        n = len(ds)
        iterator = map(convert_to_chat_format, ds)
    else:
        ds = load_dataset(ds_name, split="train", streaming=True)
        print("Dataset loaded in streaming mode")
        n = 100
        iterator = map(convert_to_chat_format, islice(ds, n))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "chess_conversations.json"
    with out_file.open("w", encoding="utf-8") as f:
        for item in iterator:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {n} examples to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/processed", help="Where to save processed data")
    parser.add_argument("--full", action="store_true", help="Process full dataset (may be large)")
    args = parser.parse_args()
    convert_and_save(args.output_dir, args.full)
