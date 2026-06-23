"""QLoRA supervised fine-tuning of Llama-3-8B for BG market-making imitation.

Recipe mirrors the SFT paper's Appendix C (adapted to Llama-3-8B / H200):
  - 4-bit NF4 quantization, bfloat16 compute (QLoRA, Dettmers et al.)
  - LoRA rank r=64, alpha=64, dropout=0.1, on all attention + MLP projections
  - Paged AdamW 32-bit, lr 2e-4, cosine schedule, warmup 0.05, 3 epochs
  - loss restricted to ACTION tokens (handled in sft_data: labels=-100 elsewhere)

Assumes the base model artifacts are present at MODEL_PATH in the repo. The new special
tokens (<OBS>/<ACT>/<REW>/...) are added and embeddings resized so the schema is stable
single-token structure.

Run:  python -m ftp.sft_train --data /mnt/user-data/outputs/sft_data --out ./ckpt
"""
from __future__ import annotations

import argparse
import os

import torch

from .sft_data import (PadCollator, SPECIAL_TOKENS, WindowedSFTDataset, load_jsonl)

MODEL_PATH = os.environ.get("LLAMA3_8B_PATH", "./models/llama-3-8b")


def build_model_and_tokenizer(model_path=MODEL_PATH, lora_r=64, lora_alpha=64,
                              lora_dropout=0.1):
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # add schema special tokens so <OBS>/<ACT>/<REW> are atomic and stable
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto")
    model.resize_token_embeddings(len(tok))
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tok


def train(data_dir, out_dir, max_len=2048, epochs=3, lr=2e-4, batch_size=2,
          grad_accum=8, lora_r=64, stride_frac=0.5, eval_frac=0.1, seed=11):
    from transformers import Trainer, TrainingArguments

    model, tok = build_model_and_tokenizer(lora_r=lora_r)

    train_recs = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    # hold out a validation slice of TRAIN episodes (test.jsonl is for the gap eval)
    n_val = max(int(len(train_recs) * eval_frac), 1)
    val_recs, tr_recs = train_recs[:n_val], train_recs[n_val:]

    train_ds = WindowedSFTDataset(tr_recs, tok, max_len=max_len, stride_frac=stride_frac)
    val_ds = WindowedSFTDataset(val_recs, tok, max_len=max_len, stride_frac=1.0)
    collator = PadCollator(pad_token_id=tok.pad_token_id)
    print(f"train windows={len(train_ds)} val windows={len(val_ds)}")

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="paged_adamw_32bit",
        bf16=True,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=True,
        report_to="none",
        seed=seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collator)
    trainer.train()
    trainer.save_model(os.path.join(out_dir, "final"))
    tok.save_pretrained(os.path.join(out_dir, "final"))
    return os.path.join(out_dir, "final")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/user-data/outputs/sft_data")
    ap.add_argument("--out", default="./ckpt")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=64)
    args = ap.parse_args()
    path = train(args.data, args.out, max_len=args.max_len, epochs=args.epochs,
                 lr=args.lr, batch_size=args.batch_size, grad_accum=args.grad_accum,
                 lora_r=args.lora_r)
    print("saved adapter to", path)
