"""Prepare the Thytu/ChessInstruct dataset into chat-style `conversations` format.
This script only transforms and saves a small sample by default to avoid heavy downloads.
Run with --full to process entire dataset (downloads ~100k examples).
"""
import argparse
from pathlib import Path


def convert_and_save(output_dir: str, full: bool = False):
    from datasets import load_dataset

    ds_name = "Thytu/ChessInstruct"
    print(f"Loading dataset {ds_name} (this may download files)...")
    ds = load_dataset(ds_name, split="train")
    print("Dataset loaded, example count:", len(ds))

    def convert(example):
        return {
            "conversations": [
                {"role": "system", "content": example["task"]},
                {"role": "user", "content": str(example["input"])},
                {"role": "assistant", "content": example["expected_output"]},
            ]
        }

    # For safety, only process a small subset unless full=True
    n = len(ds) if full else min(100, len(ds))
    ds_small = ds.select(range(n))
    ds_conv = ds_small.map(convert)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "chess_conversations.json"
    with out_file.open("w", encoding="utf-8") as f:
        for item in ds_conv:
            f.write(str(item) + "\n")

    print(f"Wrote {n} examples to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/processed", help="Where to save processed data")
    parser.add_argument("--full", action="store_true", help="Process full dataset (may be large)")
    args = parser.parse_args()
    convert_and_save(args.output_dir, args.full)
