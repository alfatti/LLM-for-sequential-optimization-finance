# Merton replication pipeline — README

Tests whether the claims of the SFT-ICL sequential-decision paper replicate on the
**Merton portfolio problem** (exact closed-form oracle, continuous action/state). See
`RESEARCH_NOTES.md` for the full Phase-0 findings, including the **flatness result**
and the two-regime design.

## What runs where

| Stage | File | Where | Status |
|-------|------|-------|--------|
| Dynamics, oracles, utilities | `merton.py`, `rollout.py` | CPU | ✓ validated |
| Closed-form in-context estimator | `estimator.py` | CPU | ✓ validated |
| Serialization + parser | `merton_serialize.py` | CPU | ✓ round-trips |
| Dataset generation | `gen_dataset.py` | CPU | ✓ datasets built |
| **Eval harness** (MSE + CRN gap) | `eval_harness.py` | CPU | ✓ validated on OLS stand-in |
| **QLoRA fine-tune** | `train_qlora.py` | **GPU + HF** | written, not run here |
| **LLM eval** (SFT / ICL / random) | `eval_llm.py` | **GPU + HF** | written, not run here |

This sandbox has **no GPU and HuggingFace is blocked**, so `train_qlora.py` and
`eval_llm.py` were authored and syntax-checked but must run on your own hardware.
Everything else was executed and validated here.

## Two regimes (see RESEARCH_NOTES.md, Findings 4–5)
- **benign** (`spread=0`): flat objective — tests whether the paper's gains *evaporate*
  when the objective is smooth (the flatness finding).
- **sharp** (`spread=0.4`, borrowing kink): curved objective (~7.6% gap at ±30% error)
  — the fair, high-contrast test of claim 1.

## Metrics
- **PRIMARY: coefficient-recovery RMSE** — in-context inference of the hidden scalar c.
  No flatness problem; this is what claim 2 is about.
- **SECONDARY: CRN optimality gap** (median + 95%-winsorized mean) — realized utility loss,
  evaluated with common random numbers. Report in both regimes.

## Run order (on your GPU box)

```bash
pip install -r requirements_gpu.txt
huggingface-cli login          # Llama-3-8B is gated

# 1. (Re)generate data at full scale — bump n_train toward 3200 (Paper 1 scale).
python gen_dataset.py --regime benign --n_train 3200 --n_eval 200
python gen_dataset.py --regime sharp  --n_train 3200 --n_eval 200   # slower: numerical oracle

# 2. Fine-tune (per regime)
python train_qlora.py --train data_benign_train.jsonl --out adapters/benign
python train_qlora.py --train data_sharp_train.jsonl  --out adapters/sharp

# 3. Evaluate: SFT vs ICL-only vs random, in-distribution and OOD
for split in id ood; do
  python eval_llm.py --eval data_benign_eval_$split.jsonl --adapter adapters/benign  # SFT
  python eval_llm.py --eval data_benign_eval_$split.jsonl                            # ICL-only
  python eval_llm.py --eval data_benign_eval_$split.jsonl --random                   # random
done
# repeat for sharp
```

## Claims and how each is read off
1. **SFT > ICL > random**: compare coeff RMSE (primary) and CRN gap across the three
   `eval_llm.py` modes. Expect the gap contrast to be *large in sharp, small in benign*
   (the flatness finding).
2. **Decomposition** (1/K_test, 1/N²): generate eval sets with larger `--k_test` and
   subsample (see `ktest_harness_check.py` for the CPU template), and vary `--n_train`.
   The closed-form estimator already shows slopes −0.90 (K_test) and −2.10 (N); the LLM
   question is whether Llama reproduces them.
3. **Exponential control**: add a `regime='exp'` path (state-independent schedule) — the
   "trivial control" expected to pass claim 1 but be insensitive to K_test.
4. **OOD**: `eval_*_ood.jsonl` use a disjoint market range (higher σ, higher μ). Compare
   RMSE/gap vs the ID sets.

## Key knobs
- `--label_noise 0.25` on support demos is REQUIRED for an observable 1/K_test curve
  (Finding 2). Query targets stay clean.
- `K_test ≲ 15` with the rich (price+wealth) serialization fits the 8k context (Phase 0
  context budget). Larger K_test needs a longer-context base model.
- Action quantized to 2 decimals — quantization floor is ~1e-3% (Finding: non-issue).

## Outcome analysis (Test S3 — world-model probe)
`outcome_analysis.py` (CPU-validated via mocks) + `run_outcome_analysis.py` (GPU, real Llama).
Probes the transition kernel with counterfactual actions: a dynamics-aware model reproduces the
return-response slope (μ−r)dt/W and, in the sharp regime, the borrowing-cost KINK at A=W; a
scalar-shortcut produces a flat response. Read in prediction space (flatness-free).
```bash
python run_outcome_analysis.py --eval data_sharp_eval_id.jsonl --adapter adapters/sharp  # SFT
python run_outcome_analysis.py --eval data_sharp_eval_id.jsonl                           # ICL-only
```
Validate the analysis logic without a GPU: `python validate_s3.py` (mock ideal/shortcut/no_kink).
