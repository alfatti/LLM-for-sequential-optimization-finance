"""
S3 outcome analysis with a real Llama model. RUN ON A GPU BOX WITH HF ACCESS.

Provides a predict_return(prefix) that greedy-decodes the step return the model expects
for a counterfactual action, then runs the (CPU-validated) analysis in outcome_analysis.py.

Usage:
  python run_outcome_analysis.py --eval data_sharp_eval_id.jsonl --adapter adapters/sharp
  python run_outcome_analysis.py --eval data_sharp_eval_id.jsonl                 # ICL-only
"""
import argparse, json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from core.outcome_analysis import analyze_eval_set
import re

MODEL = "meta-llama/Meta-Llama-3-8B"
_num = re.compile(r"\s*(-?\d+\.?\d*)")

def load(adapter=None):
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                                 device_map="auto", torch_dtype=torch.bfloat16)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model

def make_predict_return(tok, model):
    @torch.no_grad()
    def predict_return(prefix):
        ids = tok(prefix, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=6, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        m = _num.match(gen)
        return float(m.group(1)) if m else None
    return predict_return

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()
    tasks = [json.loads(l) for l in open(args.eval)]
    tok, model = load(args.adapter)
    res = analyze_eval_set(make_predict_return(tok, model), tasks, n_pts=13)
    tag = "SFT" if args.adapter else "ICL-only"
    print(f"[{tag}] {args.eval}: n={res['n']}")
    print(f"  drift sensitivity  slope_below = {res['mean_slope_below']:.5f}  "
          f"(theory {res['mean_slope_theory']:.5f})  err={res['mean_slope_err']:.5f}")
    print(f"  return-pred vs theory RMSE = {res['mean_pred_rmse']:.5f}")
    print(f"  borrowing kink drop = {res['mean_kink_drop']:.5f}  "
          f"(theory {res['mean_kink_drop_theory']:.5f})")
    print("  Interpretation: slope_below~theory and (sharp) kink_drop~theory => the model")
    print("  carries the transition kernel, not just a policy. Flat slope / no kink => shortcut.")

if __name__ == "__main__":
    main()
