#!/usr/bin/env python3
"""
Fix UCI Training Data Format for Better Instruction Tuning

Converts the current UCI training data to proper instruction-response format
with clearer prompts and more detailed responses.
"""

import json
import re
from pathlib import Path

def fix_uci_sample(sample):
    """Convert a UCI training sample to better instruction tuning format."""

    # Extract FEN from the prompt
    fen_match = re.search(r'FEN:\s*([^\n]+)', sample['prompt'])
    if not fen_match:
        return sample

    fen = fen_match.group(1).strip()

    # Create clearer instruction prompt with better structure for instruction tuning
    new_prompt = f"""Task: Find the best chess move
Position: {fen}
Instruction: Analyze the position and respond with the best move in UCI format only.

Response:"""

    # Extract the UCI move from response and ensure it's clean
    uci_move = sample['response'].strip().split('\n')[0].strip()  # Take only first line
    # Ensure it's a valid UCI move format
    if len(uci_move) >= 4 and len(uci_move) <= 5:
        new_response = uci_move
    else:
        # Fallback to original if parsing fails
        new_response = sample['response'].strip()

    return {
        **sample,
        'prompt': new_prompt,
        'response': new_response
    }

def main():
    """Fix all UCI training data samples."""
    input_file = Path("data/standardized/standardized_uci_expert.jsonl")
    output_file = Path("data/standardized/standardized_uci_expert_fixed.jsonl")

    print("🔧 Fixing UCI training data format...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")

    fixed_count = 0
    total_count = 0

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            if line.strip():
                total_count += 1
                sample = json.loads(line.strip())
                fixed_sample = fix_uci_sample(sample)

                f_out.write(json.dumps(fixed_sample) + '\n')
                fixed_count += 1

                if total_count % 10000 == 0:
                    print(f"   Processed {total_count} samples...")

    print("\n✅ UCI training data fixed!")
    print(f"   Total samples: {total_count}")
    print(f"   Fixed samples: {fixed_count}")

    # Backup original and replace
    backup_file = input_file.with_suffix('.jsonl.backup')
    print(f"\n📦 Backing up original to: {backup_file}")
    input_file.rename(backup_file)

    print(f"📦 Replacing original with fixed version")
    output_file.rename(input_file)

if __name__ == "__main__":
    main()
