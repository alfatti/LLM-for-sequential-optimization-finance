"""
QLoRA fine-tuning of Llama-3-8B for in-context Merton decision-making.
RUN THIS ON A GPU BOX WITH HF ACCESS (this sandbox has neither).

Config mirrors Paper 1 Appendix C, adapted to Llama-3-8B:
  4-bit NF4 quant, bfloat16 compute; LoRA r=64, alpha=64, dropout=0.1 on all attn+MLP
  projections; Paged AdamW 32-bit; lr 2e-4; cosine schedule; warmup 0.05; 3 epochs.
Loss is masked to the QUERY span (everything before query_char_start is context -> -100),
so the model is trained to PREDICT the optimal query actions GIVEN the noisy support
demonstrations -- i.e. to perform the in-context inference, not to memorize.

Usage:
  python train_qlora.py --train data_benign_train.jsonl --out adapters/benign
  python train_qlora.py --train data_sharp_train.jsonl  --out adapters/sharp
"""
import argparse, json
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL = "meta-llama/Meta-Llama-3-8B"
MAX_LEN = 4096   # Phase-0 context budget: K_demo*T*~38 tok fits comfortably

def build(args):
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                                 device_map="auto", torch_dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=64, lora_alpha=64, lora_dropout=0.1, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj","k_proj","v_proj","o_proj",
                                      "gate_proj","up_proj","down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return tok, model

def make_collate(tok):
    def collate(batch):
        texts = [b["text"] for b in batch]
        starts = [b["query_char_start"] for b in batch]
        enc = tok(texts, truncation=True, max_length=MAX_LEN, padding=True,
                  return_offsets_mapping=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        # mask everything before the query span (char offset -> token), and padding
        for i, (offsets, start) in enumerate(zip(enc["offset_mapping"], starts)):
            for j, (a, b) in enumerate(offsets.tolist()):
                if b <= start or (a == 0 and b == 0):   # context token or special/pad
                    labels[i, j] = -100
        labels[enc["attention_mask"] == 0] = -100
        enc.pop("offset_mapping")
        enc["labels"] = labels
        return enc
    return collate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    args = ap.parse_args()

    tok, model = build(args)
    ds = load_dataset("json", data_files=args.train, split="train")
    # 90/10 split for per-epoch validation (Paper 1 App. C)
    ds = ds.train_test_split(test_size=0.10, seed=0)

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
        optim="paged_adamw_32bit", bf16=True, logging_steps=20,
        eval_strategy="epoch", save_strategy="epoch", report_to="none",
        gradient_checkpointing=True)
    trainer = Trainer(model=model, args=targs, data_collator=make_collate(tok),
                      train_dataset=ds["train"], eval_dataset=ds["test"])
    trainer.train()
    model.save_pretrained(args.out); tok.save_pretrained(args.out)
    print("saved adapter to", args.out)

if __name__ == "__main__":
    main()
