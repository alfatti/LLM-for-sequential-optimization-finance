"""
Evaluate a fine-tuned (SFT) or base (ICL-only) Llama-3-8B on the Merton eval tasks.
RUN ON A GPU BOX WITH HF ACCESS.

For each eval task we PROBE the model: present the noisy support context, then a query
prefix ending at '<A>' at several current-wealth probe points, and read the model's
emitted action. c_hat is the OLS slope of (emitted action) on (probe wealth) -- i.e.
the model's in-context inference of the hidden coefficient. We then reuse the VALIDATED
harness (coeff MSE + CRN optimality gap) from eval_harness.py.

Baselines:
  --adapter PATH      -> SFT policy (load LoRA adapter on base model)
  (omit --adapter)    -> ICL-only baseline (base model, no fine-tuning)
  --random            -> random-coefficient control

Usage:
  python eval_llm.py --eval data_benign_eval_id.jsonl --adapter adapters/benign
  python eval_llm.py --eval data_benign_eval_id.jsonl                      # ICL-only
"""
import argparse, json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from merton_serialize import build_query_prefix, parse_first_action, fmt, SW_DECIMALS
from eval_harness import crn_gap_for_task

MODEL = "meta-llama/Meta-Llama-3-8B"
SEP = " <SEP> "
PROBE_WEALTHS = [0.8, 1.0, 1.2, 1.4]   # probe points to read off the implied coefficient

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

@torch.no_grad()
def model_action(tok, model, context, wealth, price=1.0):
    prefix = context + SEP + build_query_prefix([price], [wealth], [], [])
    ids = tok(prefix, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=8, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_first_action("<A> " + gen)

def model_estimator(tok, model):
    def est(support_text):
        Ws, As = [], []
        for w in PROBE_WEALTHS:
            a = model_action(tok, model, support_text, w)
            if a is not None:
                Ws.append(w); As.append(a)
        if len(Ws) < 2:
            return np.nan
        Ws, As = np.array(Ws), np.array(As)
        return float(np.sum(As*Ws)/np.sum(Ws*Ws))   # implied coefficient
    return est

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--n_gap_paths", type=int, default=8000)
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(args.eval)]
    if args.random:
        rng = np.random.default_rng(0)
        est = lambda s: float(rng.uniform(0.1, 2.0))
        tag = "random"
    else:
        tok, model = load(args.adapter)
        est = model_estimator(tok, model)
        tag = "SFT" if args.adapter else "ICL-only"

    sq, gaps = [], []
    for t in tasks:
        ch = est(t["support_text"])
        if np.isnan(ch): continue
        sq.append((ch - t["c_star"])**2)
        gaps.append(abs(crn_gap_for_task(t, ch, args.n_gap_paths, seed=t["task_id"])))
    gaps = np.array(gaps)
    wins = np.clip(gaps, None, np.percentile(gaps, 95))
    print(f"[{tag}] {args.eval}: n={len(sq)}  coeff RMSE={np.sqrt(np.mean(sq)):.4f}  "
          f"gap median={100*np.median(gaps):.3f}%  gap wins-mean={100*np.mean(wins):.3f}%")

if __name__ == "__main__":
    main()
