# `ftp/` — Fine-Tuning an LLM to Recover the Bergault–Guéant Market-Making Oracle

Tests whether a QLoRA-fine-tuned **Llama-3-8B** can recover near-optimal RFQ market-making
behavior under the Bergault–Guéant Fair-Transfer-Price objective, on a synthetic
single-CUSIP environment with a **censored MMPP liquidity regime** (a POMDP-in-disguise).

The oracle is the BG quadratic-approximation HJB quote map (closed-form, exact). The LLM is
trained on oracle-labeled trajectories and benchmarked against it.

---

## Pipeline at a glance

```
config.py (BG-calibrated params)
        │
        ▼
  oracle.py ── BG quadratic-approx HJB quote map  δ*(t, q, regime)   [the labels]
        │
        ▼
   env.py ──── single-CUSIP RFQ episodes (lifts engine.py fill mechanics)
        │
        ▼
 rollout.py ── thread inventory, run a policy, accumulate the BG objective
        │
        ▼
 pipeline.py ─ serialize oracle rollouts → train/test JSONL (held-out CUSIPs)
        │
        ▼
 sft_data.py ─ window episodes + mask loss to ACTION tokens only
 sft_train.py  QLoRA fine-tune Llama-3-8B
        │
        ▼
 llm_policy.py  wrap the fine-tuned model as a policy (restricted decoding)
 evaluate.py    pooled optimality-gap decomposition
 trace.py       per-decision logging  → outcome_analysis.ipynb
```

The MMPP regime is generated but **censored** from the agent's observation; the agent infers
it from sector-flow context. The **belief-optimal** policy (Bayesian filter + oracle value
function) is the *best possible flow-only policy* — the honest ceiling for a flow-only LLM.

---

## End-to-end run

```bash
# 0. (once) point at the base model — see "Llama-3-8B artifacts" below
export LLAMA3_8B_PATH=/abs/path/to/llama-3-8b

# 1. generate SFT data (1-day episodes, ~100 RFQs each)
python -m ftp.pipeline                     # writes /mnt/user-data/outputs/sft_data/{train,test}.jsonl

# 2. QLoRA fine-tune
python -m ftp.sft_train --data /mnt/user-data/outputs/sft_data --out ./ckpt

# 3. benchmark (scalar gap decomposition)
python -m ftp.evaluate --adapter ./ckpt/final --with_icl

# 4. full outcome analysis → open outcome_analysis.ipynb, set HAVE_LLM=True, run
```

---

## Llama-3-8B artifacts — getting this right

This is the part most likely to break. Read carefully.

### Where the code looks
Both `sft_train.py` and `llm_policy.py` read the base model from the env var
**`LLAMA3_8B_PATH`** (default `./models/llama-3-8b`). That path is passed directly to
HuggingFace `from_pretrained`, so it must be a **local directory containing a complete HF
checkpoint** (not a `.pth`, not a single safetensors file, not a GGUF).

### Required files in `$LLAMA3_8B_PATH`
A valid HF directory for `meta-llama/Meta-Llama-3-8B` contains:

```
config.json
generation_config.json
model-00001-of-00004.safetensors        (the sharded weights)
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json            (the shard index — REQUIRED)
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

Get them with either:

```bash
# option A: huggingface-cli (requires `huggingface-cli login` + access granted on the model page)
huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir $LLAMA3_8B_PATH

# option B: in Python
from huggingface_hub import snapshot_download
snapshot_download("meta-llama/Meta-Llama-3-8B", local_dir="...", local_dir_use_symlinks=False)
```

**Use the base (`Meta-Llama-3-8B`), not the Instruct variant.** This is supervised imitation
on a fixed schema, not chat — the base model is the right starting point and matches the SFT
paper's use of a base (non-chat) Llama-2.

### Pitfall 1 — tokenizer / embedding resize (the silent corruptor)
The training code adds 5 schema special tokens and **resizes the embedding matrix**:

```python
tok.add_special_tokens({"additional_special_tokens":
                        ["<EPISODE>","<OBS>","<ACT>","<REW>","<END>"]})
model.resize_token_embeddings(len(tok))     # vocab grows by 5
```

At **eval** time (`llm_policy.load_sft_model`) the SAME resize must happen **before** the LoRA
adapter is attached, and the tokenizer must be loaded from the **adapter dir** (which has the
added tokens), not the base dir:

```python
tok   = AutoTokenizer.from_pretrained(ADAPTER_PATH)   # has the 5 extra tokens
model = AutoModelForCausalLM.from_pretrained(BASE)    # base vocab
model.resize_token_embeddings(len(tok))               # grow to match BEFORE PeftModel
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
```

This is already wired correctly in `load_sft_model` — but if you hand-roll loading, getting
the order wrong (attaching the adapter before resizing, or loading the tokenizer from the base
dir) produces a **shape mismatch or, worse, silently shifted token ids** and garbage actions.
`sft_train` saves the resized tokenizer next to the adapter (`trainer.save_model` +
`tok.save_pretrained`), so always load the tokenizer from `./ckpt/final`.

### Pitfall 2 — action tokens must be single tokens
Actions are the digits `0`–`8`. The reward parser and restricted decoder assume each is one
token id (`tok.convert_tokens_to_ids("4")`). Llama-3's tokenizer encodes single digits as
single tokens, so this holds — but if you change the action encoding (e.g. multi-digit bins),
update `llm_policy.act_token_ids` and the `<ACT>`-target locator in `sft_data.build_windows`.

### Pitfall 3 — quantization deps
QLoRA needs `bitsandbytes` with a CUDA build matching your torch. On the H200:

```bash
pip install "torch>=2.3" transformers peft accelerate bitsandbytes datasets
python -c "import bitsandbytes as bnb; print(bnb.__version__)"   # sanity
```

If `bitsandbytes` can't find CUDA, training falls back to CPU and will be unusably slow — check
the import succeeds *before* launching a run.

### Pitfall 4 — gated-model access
`Meta-Llama-3-8B` is gated. You must (a) request access on the HF model page and be approved,
and (b) `huggingface-cli login` with a token that has that access, *before* `snapshot_download`.
A 401/403 here is an access problem, not a code problem.

---

## QLoRA recipe (matches the SFT paper, Appendix C, adapted to Llama-3-8B / H200)

| setting | value |
|---|---|
| quantization | 4-bit NF4, double-quant, bf16 compute |
| LoRA rank / α / dropout | 64 / 64 / 0.1 |
| target modules | `q,k,v,o_proj`, `gate,up,down_proj` |
| optimizer | paged AdamW 32-bit |
| lr / schedule / warmup | 2e-4 / cosine / 0.05 |
| epochs | 3 |
| loss | next-token, **masked to action tokens only** |
| max_len (window) | 2048 (raise to 4096–8192 on H200 for fewer windows) |

H200 (141 GB) runs 8B QLoRA comfortably; you can raise `--batch_size` and `--max_len` well
beyond the defaults.

---

## Module reference

| file | role | runs without GPU? |
|---|---|---|
| `oracle.py` | BG quadratic-approx HJB quote map (the labels) | ✅ |
| `env.py` | single-CUSIP RFQ episode generator (lifts `engine.py` mechanics) | ✅ |
| `rollout.py` | inventory threading, policy execution, BG objective | ✅ |
| `belief.py` | MMPP belief filter + belief-optimal policy (the ceiling) | ✅ |
| `sector.py` | windowed sector-flow context (regime-inference signal) | ✅ |
| `serialize.py` | `<OBS>/<ACT>/<REW>` encoder, action bins | ✅ |
| `pipeline.py` | generate train/test SFT JSONL (held-out CUSIPs) | ✅ |
| `sft_data.py` | episode windowing + **action-token loss masking** | ✅ (logic) |
| `sft_train.py` | QLoRA fine-tune Llama-3-8B | ❌ GPU + model |
| `llm_policy.py` | fine-tuned / ICL model as a `(step,q,t)->action` policy | ❌ GPU + model |
| `evaluate.py` | pooled optimality-gap decomposition | ❌ (LLM); ✅ for oracle/belief/random |
| `trace.py` | per-decision oracle-vs-policy logging for analysis | ✅ |
| `outcome_analysis.ipynb` | benchmark figures (gap, PnL, confusion, quote curves, inference) | ✅ non-LLM cells |

---

## Reading the result honestly

The agent is **flow-only** (it never sees the regime). So is the belief-optimal policy. The
honest headline is therefore *"the SFT LLM closes X% of the **closable** (belief-optimal)
band,"* not *"X% of the full oracle gap"* — because the full-info oracle's information is not
attainable by any flow-only policy. The notebook reports the full decomposition
(full-info oracle = 0, belief-optimal = irreducible inference loss, random = 1) so this framing
stays explicit. Section 6 (action-correctness vs belief confidence) is the key evidence for
whether the LLM learned *inference* (correct when the regime is identifiable) vs pattern-matching.

## Key configuration knobs

- `GenConfig.horizon_days` (default **1.0**) — episode length. 1 day ≈ 100 RFQs ≈ 3.8k tokens
  (single-window trainable, ICL-feasible). Longer = the paper's horizon axis but heavier.
- `GenConfig.gamma` (default 0.5) — inventory risk aversion. Sets how much of the optimal policy
  is inventory-driven vs regime-driven. (BG-calibrated γ is tiny → near-myopic; 0.5 keeps
  inventory a live signal.)
- `GenConfig.sector_halflife` — sector-flow observation window = the regime-inference fidelity
  knob (the analogue of the SFT paper's observation quality `q`).
- `GenConfig.{n_train_cusips, episodes_per_cusip}` — scale toward the paper's training-task axis.
```
