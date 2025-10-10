#!/usr/bin/env python3
"""
Complete Tutor Expert Training Script

Advanced training script for completing Tutor expert training to 1000+ steps
with enhanced data quality validation, progress monitoring, and stability improvements.

Uses instruction tuning (--use_instruction_collator) for proper chess explanations.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

def run_tutor_training(max_steps=1000, timeout_minutes=180, force_restart=False):
    """Run Tutor expert training with enhanced stability and monitoring."""

    print("=" * 60)
    print("🏆 Complete Tutor Expert Training")
    print("=" * 60)
    print(f"Target Steps: {max_steps}")
    print(f"Timeout: {timeout_minutes} minutes")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Training command with enhanced stability
    cmd = [
        sys.executable, "-m", "src.training.train_lora_poc",
        "--expert", "tutor",
        "--config", "auto",
        "--max_steps_override", str(max_steps),
        "--timeout_minutes", str(timeout_minutes),
        "--disable_eval",  # Disable eval for speed during main training
        "--use_instruction_collator"  # CRITICAL: Use instruction tuning format
    ]

    # Add force restart flag if requested
    if force_restart:
        cmd.append("--force_restart")

    print(f"🚀 Executing: {' '.join(cmd)}")
    print("-" * 60)

    # Run training with real-time monitoring
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Monitor training progress
        start_time = time.time()
        last_checkpoint_time = start_time
        checkpoint_count = 0

        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())

                # Monitor for checkpoint saves
                if "checkpoint saved" in output.lower() or "💾" in output:
                    checkpoint_count += 1
                    last_checkpoint_time = time.time()
                    elapsed = last_checkpoint_time - start_time
                    print(f"💾 Checkpoint saved after {elapsed:.1f} seconds")
        rc = process.poll()
        total_time = time.time() - start_time

        print("\n" + "=" * 60)
        if rc == 0:
            print("✅ Tutor Training Completed Successfully!")
            print(f"⏱️  Total training time: {total_time:.1f} seconds")
            print(f"📊 Average checkpoint interval: {(total_time / max(1, checkpoint_count)):.1f} seconds")
        else:
            print(f"❌ Tutor Training Failed (Exit Code: {rc})")
            print(f"⏱️  Training time before failure: {total_time:.1f} seconds")
        print("=" * 60)

        return rc == 0

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        if 'process' in locals():
            process.terminate()
        return False

def validate_tutor_data():
    """Validate Tutor training data quality."""
    print("\n🔍 Validating Tutor Training Data...")

    data_path = project_root / "data" / "standardized" / "standardized_tutor_expert_v2.jsonl"

    if not data_path.exists():
        print(f"❌ Tutor data file not found: {data_path}")
        return False

    try:
        import chess
        from src.inference.uci_utils import extract_fen

        valid_samples = 0
        invalid_samples = 0
        total_samples = 0

        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    total_samples += 1

                    # Validate required fields
                    if not all(key in sample for key in ['task', 'prompt', 'response', 'meta']):
                        invalid_samples += 1
                        continue

                    # Validate FEN and move
                    fen = extract_fen(sample['prompt'])
                    if not fen:
                        invalid_samples += 1
                        continue

                    board = chess.Board(fen)
                    response = sample['response'].strip()

                    # For tutor expert, response contains explanations but should have a valid UCI move
                    # Look for UCI move pattern in the response
                    import re
                    uci_pattern = r'\b([a-h][1-8][a-h][1-8][qrbn]?\b)'
                    uci_matches = re.findall(uci_pattern, response.lower())

                    valid_move_found = False
                    for potential_move in uci_matches:
                        try:
                            move_obj = chess.Move.from_uci(potential_move)
                            if move_obj in board.legal_moves:
                                valid_move_found = True
                                break
                        except ValueError:
                            continue

                    if not valid_move_found:
                        if line_num <= 5:  # Show first few errors
                            print(f"   Sample {line_num} error: no valid UCI move found in response: {repr(response[:100])}")
                        invalid_samples += 1
                        continue

                    valid_samples += 1

                    # Progress update every 1000 samples
                    if total_samples % 1000 == 0:
                        print(f"   Processed {total_samples} samples... ({valid_samples}/{total_samples} valid)")

                except Exception as e:
                    invalid_samples += 1
                    if total_samples <= 5:  # Show first few errors
                        print(f"   Sample {line_num} error: {e}")

        print("📊 Data Validation Results:")
        print(f"   Total Samples: {total_samples}")
        print(f"   Valid Samples: {valid_samples}")
        print(f"   Invalid Samples: {invalid_samples}")
        print(f"   Success Rate: {(valid_samples / max(total_samples, 1)):.1f}")
        return (valid_samples / max(total_samples, 1)) > 0.95  # Require 95% validity

    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        return False

def check_training_progress():
    """Check current UCI training progress."""
    print("\n📈 Checking Training Progress...")

    checkpoint_dir = project_root / "checkpoints" / "lora_uci"

    if not checkpoint_dir.exists():
        print("ℹ️  No existing UCI checkpoints found")
        return 0

    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    if not checkpoints:
        print("ℹ️  No checkpoints found")
        return 0

    # Find latest checkpoint
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
    step_count = int(latest_checkpoint.name.split("-")[1])

    # Check training summary if available
    summary_file = latest_checkpoint / "training_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                print("📊 Latest Training Summary:")
                print(f"   Steps Completed: {summary.get('total_steps', step_count)}")
                print(f"   Best Eval Loss: {summary.get('best_eval_loss', 'N/A')}")
                print(f"   Training Time: {summary.get('training_time_seconds', 0):.1f} seconds")
        except Exception as e:
            print(f"⚠️  Could not read training summary: {e}")

    print(f"🎯 Latest Checkpoint: {latest_checkpoint.name} ({step_count} steps)")
    return step_count

def main():
    parser = argparse.ArgumentParser(description="Complete UCI Expert Training")
    parser.add_argument('--max_steps', type=int, default=1600, help='Target training steps')
    parser.add_argument('--timeout_minutes', type=int, default=240, help='Training timeout in minutes')
    parser.add_argument('--skip_validation', action='store_true', help='Skip data validation')
    parser.add_argument('--force_restart', action='store_true', help='Force restart from beginning')

    args = parser.parse_args()

    print("🎯 ChessGemma Tutor Expert Training Completion Script")
    print("=" * 50)

    # Validate data quality
    if not args.skip_validation:
        if not validate_tutor_data():
            print("❌ Data validation failed. Please fix data quality issues before training.")
            sys.exit(1)
    else:
        print("⏭️  Skipping data validation")

    # Check current progress
    current_steps = check_training_progress()

    if current_steps >= args.max_steps and not args.force_restart:
        print(f"✅ Training already completed! ({current_steps} >= {args.max_steps} steps)")
        sys.exit(0)

    if args.force_restart:
        print("🔄 Force restart requested - beginning from step 0")
        current_steps = 0

    # Calculate remaining steps
    remaining_steps = args.max_steps - current_steps
    if remaining_steps <= 0:
        print(f"🎯 Target already reached! Current: {current_steps}, Target: {args.max_steps}")
        sys.exit(0)

    print(f"📍 Training Plan:")
    print(f"   Current Steps: {current_steps}")
    print(f"   Target Steps: {args.max_steps}")
    print(f"   Remaining Steps: {remaining_steps}")
    print(f"   Estimated Time: ~{remaining_steps * 2.5:.0f} seconds ({remaining_steps * 2.5 / 60:.1f} minutes)")

    # Confirm training start
    try:
        response = input("\n🚀 Start Tutor training? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ Training cancelled by user")
            sys.exit(0)
    except (KeyboardInterrupt, EOFError):
        if isinstance(__import__('sys').exc_info()[1], EOFError):
            print("🔄 Non-interactive mode - auto-confirming training start...")
        else:
            print("\n❌ Training cancelled by user")
            sys.exit(0)

    # Run training
    success = run_tutor_training(args.max_steps, args.timeout_minutes, args.force_restart)

    if success:
        print("\n🎉 Tutor Expert training completed successfully!")
        print("🔍 You can now evaluate the model performance and proceed with other experts.")
    else:
        print("\n❌ Tutor Expert training failed.")
        print("🔧 Check the logs above for error details.")
        print("💡 Try running again or check system resources.")

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
