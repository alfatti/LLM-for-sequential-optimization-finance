# Merton replication — repository

Tests whether the claims of the SFT-vs-ICL sequential-decision paper hold on the Merton
portfolio problem (exact closed-form oracle, continuous action/state). See
`notes/RESEARCH_NOTES.md` for all findings (flatness, two-regime design, structural probes).

## Structure

```
core/                 # shared, experiment-agnostic (CPU except llm.py)
  dynamics.py         # Market, CRRA/exp oracles, utilities, optimality gap
  rollout.py          # vectorized rollout + per-regime oracle (benign/sharp)
  serialize.py        # <S> <W> <A> <R> schema + parser + SEP
  estimator.py        # closed-form in-context estimators (eq.5 shrinkage + OLS)
  metrics.py          # parse_support_pairs, estimator_ols, CRN optimality gap
  outcome_analysis.py # S3 world-model probe (counterfactual return-response)
  llm.py              # GPU: load base/adapter model, coeff probe, return predictor
  train_qlora.py      # GPU: shared QLoRA trainer (used by both experiments)

fixed_mdp/            # ★ WARM-UP: ICL vs SFT on ONE fixed MDP (this thread's scope)
  config.py           # the single fixed market + the two regimes
  gen_data.py         # rollouts of one MDP -> SFT + ICL/eval data
  analysis.py         # CPU: coeff recovery + gap, OLS stand-in baseline
  eval_llm.py         # GPU: ICL / SFT / random (+ --no_context SFT ablation)

many_tasks/           # the broader task-distribution version (Paper 1 faithful)
  gen_dataset.py      # markets sampled per task; ID + OOD splits
  eval_harness.py     # CPU: coeff RMSE + CRN gap over the distribution
  eval_llm.py         # GPU: ICL / SFT / random over tasks
  run_outcome_analysis.py  # GPU: S3 on the many-tasks model

phase0/               # exploration + validation scripts (historical)
data/{fixed_mdp,many_tasks}/   # generated jsonl
notes/                # RESEARCH_NOTES.md
```

**Imports use absolute paths from the repo root** (`from core.X import ...`). Run scripts as
modules from the repo root: `python -m fixed_mdp.gen_data ...`. `core/llm.py` and the
`*_llm.py` / `train_qlora.py` scripts require a GPU + HuggingFace access (torch, transformers,
peft); everything else runs on CPU.

## Fixed-MDP warm-up (start here)

One MDP, one optimal policy c*. The LLM sees optimal-policy rollouts of that same MDP. Question:
for THIS MDP, is ICL or SFT more effective at learning to act optimally? (Claim 1, isolated.)
Claims 2 (decomposition) and 4 (cross-task OOD) are OUT of scope here — they need a task
distribution; they live in `many_tasks/`.

```bash
# CPU: generate data + establish the OLS stand-in (in-context shortcut) baseline
python -m fixed_mdp.gen_data --regime benign --n_train 600 --n_eval 150
python -m fixed_mdp.gen_data --regime sharp  --n_train 600 --n_eval 150
python -m fixed_mdp.analysis                         # the null ICL/SFT must beat

# GPU: fine-tune and compare ICL vs SFT vs random (per regime)
python -m core.train_qlora --train data/fixed_mdp/sharp_train.jsonl --out adapters/fixed_sharp
python -m fixed_mdp.eval_llm --eval data/fixed_mdp/sharp_eval.jsonl --adapter adapters/fixed_sharp   # SFT
python -m fixed_mdp.eval_llm --eval data/fixed_mdp/sharp_eval.jsonl                                  # ICL-only
python -m fixed_mdp.eval_llm --eval data/fixed_mdp/sharp_eval.jsonl --random                         # random
python -m fixed_mdp.eval_llm --eval data/fixed_mdp/sharp_eval.jsonl --adapter adapters/fixed_sharp --no_context  # memorization ablation
```

**Stand-in baseline (CPU, validated):** with 4 noisy support rollouts the OLS in-context
shortcut recovers c* at RMSE ≈ 0.04 (benign c*=1.125, sharp c*=0.959). The warm-up question is
whether base-Llama ICL reaches this clean-regression bar and whether SFT beats it. Lead with
coefficient RMSE (flatness makes the utility gap tiny in benign; sharp has contrast).

## Many-tasks version
Same `core/`, markets sampled per task; tests the full apparatus incl. the K_test/N
decomposition and cross-task OOD. See `many_tasks/` and `notes/RESEARCH_NOTES.md`.

## Metrics (both experiments)
- PRIMARY: coefficient-recovery RMSE (flatness-free; what in-context inference is about).
- SECONDARY: CRN optimality gap (median + 95%-winsorized mean), reported per regime.
- STRUCTURAL: S3 world-model probe (`core.outcome_analysis`) — counterfactual return-response
  slope and (sharp) borrowing-kink; read in prediction space.
