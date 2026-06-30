"""
Shared GPU model utilities (used by fixed_mdp and many_tasks eval). RUN ON GPU + HF.
  load(adapter)            -> (tokenizer, model); adapter=None gives the base (ICL) model
  make_coeff_probe(...)    -> estimator(support_text)->c_hat by probing actions at probe wealths
  make_return_predictor()  -> predict_return(prefix)->float (for S3 outcome analysis)
"""
import re, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from core.serialize import build_query_prefix, parse_first_action, SEP

MODEL = "meta-llama/Meta-Llama-3-8B"
PROBE_WEALTHS = [0.8, 1.0, 1.2, 1.4]
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

@torch.no_grad()
def _action_at(tok, model, context, wealth, price=1.0, use_context=True):
    head = (context + SEP) if (use_context and context) else ""
    prefix = head + build_query_prefix([price], [wealth], [], [])
    ids = tok(prefix, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_first_action("<A> " + gen)

def make_coeff_probe(tok, model, use_context=True):
    """Estimator: probe the model's action at several wealths, OLS slope = implied c_hat."""
    def est(support_text):
        Ws, As = [], []
        for w in PROBE_WEALTHS:
            a = _action_at(tok, model, support_text, w, use_context=use_context)
            if a is not None:
                Ws.append(w); As.append(a)
        if len(Ws) < 2: return np.nan
        Ws, As = np.array(Ws), np.array(As)
        return float(np.sum(As*Ws)/np.sum(Ws*Ws))
    return est

def make_return_predictor(tok, model):
    @torch.no_grad()
    def predict_return(prefix):
        ids = tok(prefix, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=6, do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        m = _num.match(gen)
        return float(m.group(1)) if m else None
    return predict_return
