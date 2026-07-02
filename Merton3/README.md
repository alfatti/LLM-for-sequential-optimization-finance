# merton_meta — Q1 / Q2 / Q4 on the Merton task family

Self-contained experimental apparatus testing three questions against the
SFT-vs-ICL sequential decision-making paper (Zhang, Aghaei, Saghafian), using
the Merton portfolio problem as a controlled testbed with exact analytic
oracles.

* **Q1** — fixed MDP: can SFT (behavioral cloning on noisy-optimal rollouts)
  recover the optimal policy? Plus an S3-style counterfactual state probe:
  did the model learn state-independence, or just the right mean action?
* **Q2** — fixed MDP: can ICL alone (base model, K_test noisy demos in
  context) do the same? Pre-registered slope: MSE ~ sigma^2/(K T), i.e. −1.
* **Q4** — task distribution (the paper's actual setting): meta-SFT across
  many Merton tasks, evaluated on held-out tasks with few-shot support demos.
  Replication axis = N_task sweep at K_test = 2 (their Fig. 2 protocol);
  extension axis = K_test sweep {1,2,4,8,16}, which their theory predicts
  (in-context term ~ 1/K_test) but their experiments never ran; mechanism
  axis = two-oracle tracking + tiered extrapolation (prior-vs-algorithm).

Q3 (learning from random rollouts) was considered and deliberately dropped.

## Design decisions (locked)

D1  Only mu varies. pi* = (mu−r)/(gamma sigma^2) ~ U[0.30, 0.80];
    r=0.02, sigma=0.20, gamma=2, dt=0.1, T=10 steps. Rationale:
    identifiability (demos reveal only the ratio) and the paper's fixed-
    covariance (fixed-Lambda) assumption, which varying sigma would violate.
D2  Context = noisy-optimal support demos: executed a_t = pi* + eps_t,
    eps ~ N(0, 0.25^2) iid per step (noise is load-bearing). Task parameters
    and the task name never appear in prompts (contamination guard); the Q1
    `--reveal-params` arm is the deliberate exception that MEASURES retrieval.
D3  Constrained scoring over a 0.01 action grid; grid extends to
    [−0.60, 1.20] — beyond the training-demonstrated range — so Tier-2/3
    clipping is model behavior, not a grid artifact.
D4  Number of support demos per TRAINING example ~ U{1..16}, so the whole
    K_test sweep is in-format. Training query labels are the CLEAN pi*
    rounded to the grid (DPT-faithful: predict the optimal action given
    in-context data ⇒ Bayes-optimal predictor is the posterior mean).
    Support demos carry the noise.
D5  Tiers: T1 interpolation [0.30,0.80]; T2 extrapolation
    [0.05,0.29] ∪ [0.81,1.00]; T3 sign-flip [−0.50,−0.125] (shorting never
    demonstrated). Mechanism test: RMSE(model → Bayes-under-training-prior)
    vs RMSE(model → prior-free sample mean) on CRN'd contexts.
D6  Primary metric RMSE(pi_hat, pi*) with Student-t CIs (over tasks for Q4,
    over CRN draws for Q1/Q2). Secondary: EXACT analytic log-CE gap
    0.5·gamma·sigma^2·dt·Σ_t (a_t − pi*)^2 (derived and MC-verified in
    env.py / proxy_validate; no Monte Carlo in the metric itself).

CRN discipline: all randomness keyed by string tags (datagen.rng_for). Demo
sets are nested in K_test; N_task training sets are nested prefixes of one
draw; query Brownian increments are policy-independent per row.

## Pre-registered predictions

1. Q1 SFT: pi_hat within a few grid steps of pi* = 0.55; probe spread ≈ 0
   (state-independence learned). Failure localizes to serialization, not
   decision-making.
2. Q2 ICL and Q4 tier-1: log-log slope of MSE vs K_test ≈ −1 (in-context
   estimation term); Q4 tier-1 RMSE approaches the Bayes floor as N_task
   grows; RMSE decreases in N_task (training-term replication).
3. SFT > ICL-only > random ordering (paper claim 1) at every matched setting.
4. Tier-2: prior-shrinkage (beta < 1, model tracks the Bayes oracle) at
   N_task = 100, attenuating toward sample-mean tracking by N_task = 3200
   (Raventós-style diversity transition).
5. Tier-3 (sign-flip): failure by clipping near the training-support
   boundary despite negative actions being expressible on the grid.
6. Scope condition vs the paper: their bound's in-context term GROWS with
   horizon T (Q-error compounding); here more steps per demo = more
   observations of a constant, so error FALLS with T. Not a contradiction —
   a designed switch-off of the compounding channel.

## Repo map

    merton_meta/config.py     presets (smoke, main) + all knobs
    merton_meta/env.py        exact log-Euler dynamics; analytic CE gap (derivation)
    merton_meta/tasks.py      training prior; deterministic eval tiers
    merton_meta/serialize.py  schema; round-trip parser; decision-span masking
    merton_meta/datagen.py    Q1/Q2/Q4 corpora + eval bundles; CRN registry
    merton_meta/oracles.py    truncated-normal Bayes posterior; sample mean
    merton_meta/metrics.py    RMSE/t-CIs; shrinkage; slopes; random baseline
    scripts/make_data.py      dataset/bundle generation
    scripts/proxy_validate.py CPU design validation (run BEFORE GPU spend)
    scripts/train_qlora.py    QLoRA trainer (paper Appendix-C recipe, Llama-3-8B)
    scripts/eval_llm.py       constrained-scoring eval; closed-loop; probe
    scripts/make_plots.py     figures
    scripts/run_all.sh        full run order

## Run order

CPU (here, already executed): `make_data --preset smoke` → `proxy_validate`.
GPU (H200): smoke train + smoke eval, then `scripts/run_all.sh` stages 3–6.
GPU deps: `pip install torch transformers peft bitsandbytes datasets accelerate`.

## Compute notes

* Training: 3,200 tasks × 4 examples ≈ 12.8k sequences ≈ 20–25M tokens/epoch
  × 3 epochs. Paper reports hours-to-days per run on an A100-80GB; expect
  less on an H200.
* Eval: cached scoring amortizes the prompt; per decision, only the ~5-token
  candidates are forwarded against the reused KV cache (chunked). Startup
  spot-check (`--verify-naive`) compares cached vs exact argmax; fall back to
  `--scoring naive` if it flags disagreement.
* Q4 bundle ≈ 780 closed-loop rollouts of 10 decisions × 181 candidates.

## Deviations / refinements embedded in code (flag in the paper)

* Action grid extended beyond training support (D3 rationale above).
* Clean (grid-rounded) query labels + noisy executed support demos —
  DPT-faithful; the paper's oracle labels are exact actions.
* Analytic CE-gap identity replaces Monte Carlo for the secondary metric.
* Q1 default trains on noisy-behavior rollouts (sigma 0.25); the noiseless
  arm (`--q1-sigma-label 0.0`) is the trivial-memorization control.
* Base-model arms are evaluated under the identical constrained-scoring
  protocol (the paper's ICL policy restricts outputs to action tokens).
