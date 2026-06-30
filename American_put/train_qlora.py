"""
QLoRA SFT for the single American-put exercise policy. RUN ON GPU (H200).

Faithful to Paper 1 (Zhang/Aghaei/Saghafian, App. C): Llama-2-7B base,
QLoRA / 4-bit NF4, bf16 compute, Paged AdamW 32-bit, lr 2e-4, cosine schedule,
warmup 0.05, LoRA r=64 / alpha=64 / dropout 0.1 on all attention + MLP proj
layers, 3 epochs.

Two loss modes (the chosen fork):
  --loss full    : full-sequence LM loss over every token (Paper 1 Eq. 1).
                   THIS IS THE CORE RUN.
  --loss masked  : loss only on decision tokens (STOP/GO spans), state tokens
                   masked to -100. Ablation: tests whether the continuous-state
                   token burden taxes the faithful method.

Consumes sft_text.jsonl (one {"text": ...} per line) from the rollout generator.
Writes a LoRA adapter to --out, plus a small run_config.json.

This script trains only. Evaluation (ICL-only / SFT / random, the three metrics)
is a separate harness that loads this adapter and rolls policies forward on
fresh lattice paths -- mirroring proxy_sft.py but with the LLM in the policy slot.

Dependencies (install on the GPU box):
  pip install "transformers>=4.40" "peft>=0.10" "trl>=0.8" accelerate \
              bitsandbytes datasets
Llama-2-7B requires accepting Meta's license and an HF token (huggingface-cli login).
"""

import argparse, json, os
import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments,
                          DataCollatorForLanguageModeling, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# --------------------------------------------------------------------------
# Paper 1, Appendix C configuration
# --------------------------------------------------------------------------
DEFAULTS = dict(
    base_model="meta-llama/Llama-2-7b-hf",
    lora_r=64, lora_alpha=64, lora_dropout=0.1,
    lr=2e-4, epochs=3, warmup_ratio=0.05,
    batch_size=8, grad_accum=2, max_len=1024,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],   # all attn + MLP proj
)


def load_texts(path):
    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    return texts


def tokenize_example(text, tokenizer, max_len, loss_mode):
    """
    Tokenize one serialized trajectory. Returns dict(input_ids, attention_mask,
    labels). For 'full', labels == input_ids (standard causal LM, with pads
    -100'd by the collator). For 'masked', labels are -100 except on decision
    token spans, located via character offsets of the '-> DECISION' regions.
    """
    enc = tokenizer(text, truncation=True, max_length=max_len,
                    return_offsets_mapping=True, add_special_tokens=True)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    attn = enc["attention_mask"]

    if loss_mode == "full":
        labels = list(input_ids)
    else:
        # find char spans of each decision: substring after "-> " up to " ;" or end
        decision_spans = []
        i = 0
        marker = "-> "
        while True:
            p = text.find(marker, i)
            if p == -1:
                break
            start = p + len(marker)
            end = text.find(" ;", start)
            if end == -1:
                end = len(text)
            decision_spans.append((start, end))
            i = end
        # a token is "decision" if its char span overlaps any decision span
        def in_decision(a, b):
            for (s, e) in decision_spans:
                if a < e and s < b:   # overlap
                    return True
            return False
        labels = []
        for tok_id, (a, b) in zip(input_ids, offsets):
            # offset (0,0) marks special tokens -> never a decision
            if b > a and in_decision(a, b):
                labels.append(tok_id)
            else:
                labels.append(-100)

    return dict(input_ids=input_ids, attention_mask=attn, labels=labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sft_text.jsonl")
    ap.add_argument("--out", default="adapter_put_sft")
    ap.add_argument("--loss", choices=["full", "masked"], default="full",
                    help="full = Paper 1 Eq.1 core run; masked = decision-token ablation")
    ap.add_argument("--base_model", default=DEFAULTS["base_model"])
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--grad_accum", type=int, default=DEFAULTS["grad_accum"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--max_len", type=int, default=DEFAULTS["max_len"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ---- tokenizer ----
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token   # Llama-2 has no pad; use eos for padding
    tok.padding_side = "right"

    # ---- data ----
    texts = load_texts(args.data)
    print(f"[data] {len(texts)} trajectories from {args.data}")
    # truncation report: clipping a 50-step path drops its terminal decision,
    # a silent data-integrity bug. Surface it before training.
    n_trunc = 0
    max_tok = 0
    for t in texts:
        ln = len(tok(t, add_special_tokens=True)["input_ids"])
        max_tok = max(max_tok, ln)
        if ln > args.max_len:
            n_trunc += 1
    print(f"[len] longest trajectory = {max_tok} tokens; max_len = {args.max_len}; "
          f"truncated = {n_trunc}/{len(texts)}")
    if n_trunc > 0:
        print(f"[len] WARNING: {n_trunc} trajectories exceed max_len and will be "
              f"clipped (losing their terminal decision). Raise --max_len.")
    feats = [tokenize_example(t, tok, args.max_len, args.loss) for t in texts]
    # quick mask sanity for the ablation
    if args.loss == "masked":
        n_sup = sum(sum(1 for x in f["labels"] if x != -100) for f in feats)
        n_tot = sum(len(f["labels"]) for f in feats)
        print(f"[mask] supervised tokens = {n_sup}/{n_tot} "
              f"({100*n_sup/n_tot:.1f}%) -- decisions only")
        # show one decoded example's supervised positions
        ex = feats[0]
        sup_toks = tok.decode([i for i, l in zip(ex["input_ids"], ex["labels"])
                               if l != -100])
        print(f"[mask] example supervised tokens decode to: {sup_toks!r}")
    ds = Dataset.from_list(feats)

    # ---- model: 4-bit NF4 QLoRA ----
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    lcfg = LoraConfig(
        r=DEFAULTS["lora_r"], lora_alpha=DEFAULTS["lora_alpha"],
        lora_dropout=DEFAULTS["lora_dropout"], bias="none",
        task_type="CAUSAL_LM", target_modules=DEFAULTS["target_modules"],
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    # ---- collator: pads input_ids and labels, sets pad labels to -100 ----
    # We pass pre-tokenized examples with explicit labels, so use a collator that
    # pads labels with -100. DataCollatorForLanguageModeling(mlm=False) would
    # overwrite labels; instead use a light custom pad.
    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad_id = tok.pad_token_id
        input_ids, attn, labels = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * n)
            attn.append(b["attention_mask"] + [0] * n)
            labels.append(b["labels"] + [-100] * n)
        return dict(
            input_ids=torch.tensor(input_ids),
            attention_mask=torch.tensor(attn),
            labels=torch.tensor(labels),
        )

    # ---- training ----
    steps_per_epoch = max(1, len(ds) // (args.batch_size * args.grad_accum))
    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=DEFAULTS["warmup_ratio"],
        optim="paged_adamw_32bit",
        bf16=True,
        logging_steps=max(1, steps_per_epoch // 10),
        save_strategy="epoch",
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=collate)
    trainer.train()

    # ---- save adapter + config ----
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump(dict(loss_mode=args.loss, base_model=args.base_model,
                       epochs=args.epochs, lr=args.lr, seed=args.seed,
                       lora=dict(r=DEFAULTS["lora_r"], alpha=DEFAULTS["lora_alpha"],
                                 dropout=DEFAULTS["lora_dropout"]),
                       n_train=len(texts)), f, indent=2)
    print(f"[done] adapter + config saved to {args.out}/  (loss={args.loss})")


if __name__ == "__main__":
    main()
