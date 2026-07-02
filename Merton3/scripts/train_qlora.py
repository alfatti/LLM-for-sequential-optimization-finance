#!/usr/bin/env python3
"""QLoRA SFT trainer (GPU), shared by Q1 and Q4.

Faithful to the target paper's recipe (Appendix C), transplanted to Llama-3-8B:
  * QLoRA, 4-bit NF4 double quantization, bf16 compute
  * LoRA r=64, alpha=64, dropout 0.1 on all attention + MLP projections
  * Paged AdamW 32-bit, lr 2e-4, cosine schedule, warmup ratio 0.05, 3 epochs
  * validation split BY TASK (10%), done upstream in datagen

Loss masking (pre-registered ablation, --mask-mode):
  decision : loss only on the action tokens after '-> a=' in the QUERY section
  query    : loss on the whole QUERY section
  full     : loss on the full sequence
Character spans come from merton_meta.serialize.decision_char_spans and are
mapped to tokens via the fast tokenizer's offset_mapping (robust to subword
tokenization).

Usage:
  python scripts/train_qlora.py --data data_main/q4/ntask_03200 \
      --out runs/q4_n3200 --max-len 4096
  python scripts/train_qlora.py --data data_main/q1 --out runs/q1 --max-len 640
  # smoke: --smoke caps optimizer steps and dataset size
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from merton_meta.serialize import decision_char_spans  # noqa: E402


def load_rows(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def build_labels(offsets, spans, input_ids, pad_token_id):
    """Token i receives loss iff its char span overlaps any target span."""
    labels = []
    for (a, b), tid in zip(offsets, input_ids):
        if tid == pad_token_id or (a == 0 and b == 0):
            labels.append(-100)
            continue
        keep = any(a < s_end and b > s_start for s_start, s_end in spans)
        labels.append(tid if keep else -100)
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with train.jsonl / val.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--mask-mode", choices=["decision", "query", "full"],
                    default="decision")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--per-device-bsz", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--smoke", action="store_true",
                    help="cap at 32 train rows / 20 optimizer steps")
    args = ap.parse_args()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig, Trainer, TrainingArguments)
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            f"missing GPU dependency ({e}); install: pip install torch transformers "
            "peft bitsandbytes datasets accelerate") from e

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    assert tok.is_fast, "fast tokenizer with offset_mapping is required"

    def encode(row):
        enc = tok(row["text"], truncation=True, max_length=args.max_len,
                  return_offsets_mapping=True, add_special_tokens=True)
        spans = decision_char_spans(row["text"], args.mask_mode)
        labels = build_labels(enc["offset_mapping"], spans, enc["input_ids"],
                              tok.pad_token_id)
        if all(l == -100 for l in labels):
            raise ValueError("example lost all supervised tokens after truncation; "
                             "increase --max-len")
        return {"input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"], "labels": labels}

    limit = 32 if args.smoke else None
    train_rows = load_rows(os.path.join(args.data, "train.jsonl"), limit)
    val_rows = load_rows(os.path.join(args.data, "val.jsonl"),
                         8 if args.smoke else None)
    train_ds = Dataset.from_list(train_rows).map(
        encode, remove_columns=list(train_rows[0].keys()))
    val_ds = Dataset.from_list(val_rows).map(
        encode, remove_columns=list(val_rows[0].keys()))

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id

        def _pad(seq, val):
            return seq + [val] * (maxlen - len(seq))
        return {
            "input_ids": torch.tensor([_pad(b["input_ids"], pad) for b in batch]),
            "attention_mask": torch.tensor(
                [_pad(b["attention_mask"], 0) for b in batch]),
            "labels": torch.tensor([_pad(b["labels"], -100) for b in batch]),
        }

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=(20 if args.smoke else -1),
        per_device_train_batch_size=args.per_device_bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        optim="paged_adamw_32bit",
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collate)
    trainer.train()
    model.save_pretrained(os.path.join(args.out, "adapter_final"))
    tok.save_pretrained(os.path.join(args.out, "adapter_final"))
    with open(os.path.join(args.out, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[train_qlora] adapter saved to {args.out}/adapter_final")


if __name__ == "__main__":
    main()
