"""Safe train entrypoint. By default this script will NOT start training.
Set --do_train to actually run a small smoke training run (max_steps default 10).
"""
import argparse


def main(do_train: bool, max_steps: int):
    if not do_train:
        print("train.py: safe mode (no training). Use --do_train to run a small smoke training job.")
        return

    # Import heavy libraries only when training is requested
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    print("Starting smoke training (this will download model weights)...")
    MODEL_NAME = "unsloth/gemma-3-270m-it"
    # Cap CPU threads to 2 by default if not already constrained
    import os as _os
    _os.environ.setdefault('OMP_NUM_THREADS', '2')
    _os.environ.setdefault('MKL_NUM_THREADS', '2')
    _os.environ.setdefault('NUMEXPR_NUM_THREADS', '2')

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
        full_finetuning=False,
    )

    # Wrap with LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        use_rslora=False,
        random_state=3407,
    )

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
