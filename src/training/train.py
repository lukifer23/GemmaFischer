"""Safe train entrypoint. By default this script will NOT start training.
Set --do_train to actually run a small smoke training run (max_steps default 10).
"""
import argparse


def main(do_train: bool, max_steps: int):
    if not do_train:
        print("train.py: safe mode (no training). Use --do_train to run a small smoke training job.")
        return

    # Import heavy libraries only when training is requested
    import os
    from pathlib import Path
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print("Starting smoke training (this will download model weights)...")
    MODEL_NAME = os.environ.get("CHESSGEMMA_MODEL_PATH") or os.environ.get("CHESSGEMMA_MODEL_ID")
    if MODEL_NAME is None:
        local_snapshot = Path(__file__).resolve().parents[2] / "models" / "google-gemma-3-270m"
        MODEL_NAME = str(local_snapshot) if local_snapshot.exists() else "google/gemma-3-270m"
    # Cap CPU threads to 2 by default if not already constrained
    import os as _os
    _os.environ.setdefault('OMP_NUM_THREADS', '2')
    _os.environ.setdefault('MKL_NUM_THREADS', '2')
    _os.environ.setdefault('NUMEXPR_NUM_THREADS', '2')

    # Load tokenizer/model
    path_obj = Path(MODEL_NAME)
    using_local = path_obj.exists()
    load_target = str(path_obj) if using_local else MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(load_target, local_files_only=using_local, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = torch.float16
    if torch.backends.mps.is_available() or not torch.cuda.is_available():
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        load_target,
        local_files_only=using_local,
        device_map="auto",
        attn_implementation="eager",
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    model = get_peft_model(model, lora_config)

    # Load tiny dataset sample
    dataset = load_dataset("Thytu/ChessInstruct", split="train[:200]")

    def convert(example):
        return {
            "conversations": [
                {"role": "system", "content": example["task"]},
                {"role": "user", "content": str(example["input"])},
                {"role": "assistant", "content": example["expected_output"]},
            ]
        }

    dataset = dataset.map(convert)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            max_seq_length=512,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=max_steps,
            learning_rate=5e-5,
            fp16=False,
            bf16=False,
            logging_steps=5,
            optim="adamw_hf",
            weight_decay=0.01,
            seed=3407,
        ),
    )

    print("Running trainer.train() ...")
    stats = trainer.train()
    print("Training finished. Stats:", stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action="store_true", help="Run a tiny smoke training run")
    parser.add_argument("--max_steps", type=int, default=10, help="Max steps for smoke run")
    args = parser.parse_args()
    main(args.do_train, args.max_steps)
