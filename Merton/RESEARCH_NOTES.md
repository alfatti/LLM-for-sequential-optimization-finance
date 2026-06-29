# Research log — Merton problem as a replication test case for SFT-ICL sequential decision-making

**Context.** Testing whether the premise/claims of *Zhang, Aghaei, Saghafian — "LLMs for
Sequential Decision-Making: Improving ICL via SFT"* (Paper 1) replicate when the task is the
**Merton portfolio problem** as set up in *Fathi — "Stochastic Optimal Control in Algorithmic
Trading"* (Paper 2). Merton is attractive because it supplies an **exact closed-form oracle**
(the labeling oracle Paper 1's framework depends on), is **continuous-action/continuous-state**
(a regime Paper 1 never tested — they use discrete actions), and has **clean theory to check
against** Paper 1's suboptimality bound.

Claims under test:
1. **SFT beats ICL-only and random** on the optimality gap.
2. **Error decomposition** (Paper 1 eq. 6): in-context estimation error (↓ with K_test) is
   *separable* from training-length bias (↓ with K / N).
4. **OOD robustness** — train on one market region, test on another.

Utilities (Paper 2): exponential `U(x) = -exp(-γx)` and CRRA power `U(x) = x^α/α`.

---

## 0. Setup and conventions

Discounted wealth SDE (Paper 2 eq. 16), `u_t` = **dollar amount** in the risky asset:

    dX_t = (u_t (μ - r) + r X_t) dt + u_t σ dW_t.

Two oracle controls:

- **Exponential (Paper 2 eq. 26):** `u*(t) = (λ/(γσ)) exp(-r(T-t))`, `λ=(μ-r)/σ`.
  **STATE-INDEPENDENT** — depends only on the clock and market params.
- **CRRA power, RRA ρ = 1-α:** `u*(t,x) = c·x`, with
  `c = (μ - r) / (σ² ρ)`. **STATE-DEPENDENT** (the Merton fraction). The per-task
  **hidden scalar** the model must infer from context is `c`.

A "task" τ = a market draw `(μ, σ, r)` with ρ (preference) fixed. Few-shot ICL must recover
the hidden `c` (CRRA) or the parameter set (exp) from demonstration trajectories.

Baseline market used throughout Phase 0: `μ=0.12, σ=0.20, r=0.03, ρ=2.0, T=1, 10 steps, x0=1`.
=> analytic `c* = (0.12-0.03)/(0.04·2) = 1.125`.

---

## FINDING 1 — Oracle is exact (validated)

Numerical EU-maximization over the policy class `u=c'x` peaks exactly at the analytic `c*`:

- Analytic `c* = 1.1250`; empirical grid argmax = `1.1250`; quadratic-fit peak `1.107` (within 1.6%).

**=> Labels are exact.** This removes a confound Paper 1 cannot fully escape in its tabular
experiments, where the "oracle" is itself an approximate backward-induction solution.

---

## FINDING 2 — Demonstration noise is REQUIRED for the K_test term to be observable
### (a design decision, settled empirically)

In CRRA-Merton the optimal demo pairs are `(x_i, c·x_i)`. With **noiseless labels**, exact OLS
recovers `c` from a **single** demonstration step — so K_test carries no information and the
in-context error term (Paper 1's 1/M) is at floor from K_test=1.

Measured MSE of ĉ vs K_test (GBM features, realistic Merton):

| condition / estimator        | K_test=1 | K_test=10 | K_test=50 | behavior |
|------------------------------|----------|-----------|-----------|----------|
| noiseless, OLS               | 2.4e-32  | 3.5e-32   | 2.4e-32   | **flat at machine zero** |
| label-noise (0.25), OLS      | 8.2e-3   | 8.9e-4    | 1.7e-4    | clean **~1/K_test** |
| label-noise (0.25), shrinkage| 8.9e-2   | 8.7e-3    | 1.8e-3    | clean **~1/K_test** |

**=> Inject multiplicative label noise on the demonstrated actions in the LLM phase.** This
restores the 1/K_test law regardless of which estimator the model emulates. NB "process noise"
is *not* a separable knob in Merton — the wealth path is GBM-stochastic either way; the only
real lever is observation/label noise on the action. This is faithful to Paper 1's theory: their
1/M term arises precisely because the in-context features are random draws with variance to
reduce (Lemma 1; their pairs `(x_i, y_i)`, `y_i=<w*,x_i>`).

---

## FINDING 3 — Paper 1's error decomposition is PRE-REPLICATED analytically (claim 2 theory)

Using the eq. (5) shrinkage predictor `ĉ = (1/M Σ y_i x_i)/[S(1+2/N)]` in the matched-Gaussian
harness (Paper 1's Appendix E.2 / Fig. 4 setup), with `S = E[x²]` known:

- **In-context term (1st term of eq. 6):** Var(ĉ) vs K_test under label noise →
  **log-log slope = -1.00** (theory: -1). Clean 1/K_test.
- **Training-length term (2nd term of eq. 6):** the predictor's bias is
  `E[ĉ] = c/(1+2/N)` ⇒ `bias = -2c/(N+2) ~ -2c/N` ⇒ `bias² ~ 1/N²`. Measured empirical bias
  matches `-2c/(N+2)` almost exactly; **log-log slope of bias² vs N = -2.10** (theory: -2).

**d=1 specialization of eq. (6) — a clean property of the Merton test.** With a scalar feature
(`x_i` = wealth), `Λ = E[x²] =: S` and the condition number `κ` collapses to 1, so Paper 1's
bound

    ε_Q(M,N) ≤ (d+1)tr(Λ)/M + (1+2d+d²κ)tr(Λ)/N²

reduces to `ε_Q ≤ 2S/M + (≈4)S/N²`. **Merton isolates the M and N scaling without the d²κ
confound present in their analysis.** Arguably a *cleaner* test of the decomposition than the
paper's own tabular experiments (whose optimal Q-function is not actually linear).

**Interpretation for the paper:** the theory (claim 2) is TRUE in this setting independent of any
LLM. The LLM phase therefore tests the separate question — *can Llama-3-8B realize this
estimator in-context?* This is a useful decomposition of "is the theory right" vs "can the model
do it."

---

## FINDING 4 — ★ THE FLATNESS FINDING (headline; a refinement of Paper 1's thesis) ★

**The smooth Merton optimality gap is intrinsically flat near the optimum**, which makes the
gap a *poor discriminator* for claim 1 — not because SFT fails, but because the metric is
**objective-shape-dependent**.

Quantified (CRN evaluation, gap vs relative coefficient error, baseline CRRA ρ=2):

| ĉ/c* | 0.5 | 0.8 | 0.9 | 1.1 | 1.2 | 1.5 |
|------|-----|-----|-----|-----|-----|-----|
| optimality gap | 1.25% | 0.20% | 0.05% | 0.06% | 0.22% | 1.35% |

A ±20% coefficient error costs only ~0.2%; a 50% error ~1.3%. Contrast Paper 1's *discrete*
tasks (random 50% → ICL 30% → SFT 15%), where wrong actions are costly. **In smooth Merton,
wrong-but-close actions are nearly free** because the certainty-equivalent loss from a
suboptimal constant fraction is second-order small (classical "Merton's objective is flat").

### Can α (CRRA curvature) fix it? — No.
Sweeping α (RRA ρ=1-α), measuring contrast = gap at ±30% error, and the leverage/degeneracy
boundary (`c*` explodes, wealth crosses zero in discrete time):

| α | ρ | c* | contrast@30% | % wealth ≤ 0 |
|------|-----|-------|------|------|
| -5.0 | 6.0 | 0.375 | 0.80% | 0% |
| -2.0 | 3.0 | 0.750 | 0.63% | 0% |
| -1.0 | 2.0 | 1.125 | 0.47% | 0% |
| -0.5 | 1.5 | 1.500 | 0.33% | 0% |
| 0.2 | 0.8 | 2.812 | 0.34% | 0% |
| 0.5 | 0.5 | 4.500 | 2.40% | **2%** |
| 0.7 | 0.3 | 7.500 | (degenerate) | **29%** |
| 0.85| 0.15| 15.00 | (degenerate) | **65%** |

**=> In the non-degenerate band (α ≤ 0.2) contrast stays sub-1% regardless of risk aversion.**
The only place contrast rises is exactly where the discrete-time wealth process breaks down
(unbounded leverage). **The flatness is intrinsic to the non-degenerate smooth Merton objective;
α cannot escape it.**

### Quadratic position cost? — Makes it WORSE.
Adding `-(κ/2)Σu²dt` shrinks `c*` toward 0 (penalizes holding the asset), landing in an *even
flatter* region: contrast drops 0.49% → 0.01% as κ: 0 → 20. Wrong mechanism — it shrinks the
position rather than penalizing deviations.

### Paper-ready statement of the finding
> The magnitude of SFT's advantage over ICL — and indeed of *any* policy's optimality gap —
> depends on the **curvature of the objective around the optimum**, not only on horizon,
> observability, or model ambiguity (the axes Paper 1 varies). In smooth-objective domains the
> reported gains compress toward zero even when in-context *parameter recovery* differs
> substantially across methods. This predicts that Paper 1's headline gains are largest in
> "sharp" discrete tasks and smallest in "flat" smooth-control tasks, and recommends reporting
> **parameter-recovery error** alongside the optimality gap.

---

## FINDING 5 — A faithful curvature knob: the borrowing spread

To give claim 1 a fair high-contrast test *without* leaving Merton or inducing leverage
degeneracy: charge `r` on cash but `r+s` when levered (`u > x`):

    dX = (u(μ-r) + r x - s·max(0, u - x)) dt + u σ dW.

For `u=c x` the penalty is `s·x·max(0, c-1)` — a **kink at c=1** (fully invested). Over-leverage
becomes **first-order** costly, breaking flatness. Results:

- `c*` **stabilizes at ≈ 0.91** (just below the kink) — bounded, non-degenerate.
- Oracle **stays linear** (`u=c*x`; numerically, best intercept b=0 for all s>0) — the
  "infer a hidden scalar" task structure is **preserved**.
- Contrast is a **smooth, tunable** function of the spread:

  | spread s | 0.0 | 0.05 | 0.1 | 0.2 | 0.4 | 0.7 | 1.0 |
  |----------|-----|------|-----|-----|-----|-----|-----|
  | gap@±30% | 0.47% | 1.06% | 1.98% | 3.84% | **7.59%** | 13.6% | 20.0% |

- Sharpness is **one-sided** (over-leverage punished, under-investing gently costly) — correct
  economically, and an interesting asymmetry to report.
- Economic faithfulness: two interest rates = realistic margin/financing. Doubles as a bridge to
  Paper 2's *other* problem (price impact / execution cost).

`s ≈ 0.4` gives ~7.6% contrast, comparable to Paper 1's operating regime.

---

## Practical constraints carried into the LLM phase

- **Primary metric = coefficient-recovery MSE** (no flatness problem; directly measures
  in-context inference; is what claim 2 is about). **Optimality gap = secondary**, reported in
  both regimes via **common random numbers** (CRN).
- **CRN is essential for gap eval.** Without it the gap is lost in MC noise at ~0.5% (Paper 1
  uses only 30–90 eval rollouts/task — their reported gaps carry similar noise).
- **Action quantization is a non-issue**: rounding to 1 decimal → 0.0008% gap floor; even integer
  rounding → 0.14% (flatness again). 1–2 decimals suffices for the LLM's number output.
- **Context budget binds K_test.** Llama-3-8B 8k window; per-step serialization with history
  depth h≈3 ≈ 38 tok/step ≈ 380 tok/trajectory (T=10). K_test=20 ≈ 8k tokens (near limit),
  K_test=50 overflows. **=> cap K_test ≈ 15 with h=3.**

---

## Two-regime experimental design (locked)

| | **Benign (s=0)** | **Sharp (s≈0.4)** |
|---|---|---|
| objective near optimum | flat (~0.5% gap at ±30%) | curved (~7.6%) |
| hidden scalar c* | 1.125 | 0.913 |
| dynamics | plain Merton | Merton + borrowing kink |
| what it tests | do Paper 1's gains **evaporate** when objective is smooth? (the flatness finding) | does SFT beat ICL when mis-estimation is **costly**? (fair claim-1 test) |

Plus the **exponential-utility** run as a deliberate "trivial control": state-independent oracle,
expected to satisfy claim 1 easily but be **insensitive to K_test** (nothing state-dependent to
infer) — a built-in ablation.

---

## What the LLM phase will test (open questions)

- Can Llama-3-8B **recover the hidden scalar c in-context** (MSE), and does **SFT improve recovery
  over ICL-only**? (claim 1, in the sensitive metric)
- Does the LLM's MSE show the **1/K_test** and **1/N²** scalings the closed-form estimator does?
  (claim 2, realization question)
- Does the **optimality-gap** advantage appear in the **sharp** regime but **vanish** in the
  **benign** regime? (the flatness finding, confirmed or refuted with a real model)
- **OOD**: train on `(μ,σ)` region A, test on region B. (claim 4)

## File map (Phase 0, in /home/claude)
- `merton.py` — dynamics, oracles (exp + CRRA), utilities, optimality gap, simulator.
- `estimator.py` — closed-form in-context estimators (eq. 5 shrinkage + OLS).
- `verify_oracle.py` — Finding 1.
- `phase0_noise_pilot.py` — Finding 2 (K_test sweep, GBM features).
- `phase0_bias_gaussian.py` — Finding 3 (matched-Gaussian, both decomposition terms).
- `phase0_gap_crn.py`, `phase0_alpha_sweep.py`, `phase0_quadcost.py`, `phase0_borrow.py` —
  Finding 4 + 5 (flatness, α-degeneracy, quadratic-cost failure, borrowing-spread fix).
- `phase0_grid_context.py` — quantization floor + context budget.
- Figures: `phase0_pilot.png` (4-panel: design decision, both decomposition terms, flatness bowl),
  `phase0_regimes.png` (curvature knob + benign-vs-sharp).

---

# PHASE 1 — Pipeline build status (validated half)

The data/serialization/eval pipeline is built and **validated end-to-end on CPU** using the
closed-form OLS estimator as a stand-in for the LLM (the LLM half — `train_qlora.py`,
`eval_llm.py` — is authored and syntax-checked but requires a GPU + HuggingFace access, both
unavailable in the build sandbox).

**Serialization** (rich state, Paper 1 schema): `<S> price <W> wealth <A> action <R> return`
per step; action 2-decimals. Support demos carry label noise 0.25; query targets clean.

**Datasets generated** (ID range μ∈[.06,.14] σ∈[.15,.25]; OOD μ∈[.14,.20] σ∈[.25,.35], disjoint;
ρ=2, T=10). Note: original OOD range (low σ) induced leverage degeneracy (c* up to 9.5, CRRA
U=−1/x blowups) — fixed by moving OOD to higher σ so c* stays bounded. Sharp regime was immune
(borrowing kink caps leverage) — a useful consistency check.

**Harness validation (OLS stand-in):**

| regime/split | coeff RMSE | gap median | gap wins-mean |
|--------------|-----------|-----------|---------------|
| benign / id  | 0.047 | 0.005% | 0.009% |
| benign / ood | 0.031 | 0.006% | 0.011% |
| sharp / id   | 0.034 | 0.008% | 0.059% |
| sharp / ood  | 0.027 | 0.009% | 0.020% |

Regime contrast confirmed: at comparable RMSE, **sharp gap (0.059%) > benign gap (0.009%)** —
the sharp objective amplifies the same coefficient error into a larger utility loss, as designed.
(OLS RMSE is small because OLS is near-perfect; a real LLM, esp. ICL-only, will have larger RMSE,
amplifying the contrast.)

**Harness measures claim 2 on real data:** subsampling K_test=1..12 support trajectories from a
generated eval set, coeff MSE vs K_test has **log-log slope −0.90** (theory −1). The full
pipeline (generate→serialize→parse→estimate→metric) reproduces the in-context term.

**Decisions locked into the pipeline:** primary metric = coeff RMSE; CRN for all gap eval (median
+ 95%-winsorized mean for CRRA tail robustness); K_test ≲ 15 with rich serialization (8k budget);
label noise on support only; OOD via disjoint higher-σ range.

**Remaining for the LLM phase (on GPU):** run `train_qlora.py` (benign, sharp) and `eval_llm.py`
(SFT / ICL-only / random; ID / OOD). Add the `exp` regime path for the trivial-control ablation.
Open questions unchanged (can Llama recover c in-context; does it reproduce the −0.90/−2.10 slopes;
does the gap advantage appear in sharp but vanish in benign).

## Pipeline file map (/home/claude)
- `merton_serialize.py` — schema + parser.   `rollout.py` — vectorized rollout + per-regime oracle.
- `gen_dataset.py` — dataset generator.       `eval_harness.py` — coeff MSE + CRN gap (validated).
- `ktest_harness_check.py` — 1/K_test validation on real data.
- `train_qlora.py` — QLoRA fine-tune (GPU).   `eval_llm.py` — SFT/ICL/random eval (GPU).
- `requirements_gpu.txt`, `README_PIPELINE.md`.
- Data: `data_{benign,sharp}_{train,eval_id,eval_ood}.jsonl`, `data_benign_eval_kt12.jsonl`.

---

# STRUCTURAL PROBES — does the LLM learn MDP structure, or just the scalar shortcut?

**Discriminator principle.** A structural test is diagnostic only if the *scalar-shortcut*
hypothesis (recover c, act u=cx) and the *structure* hypothesis predict measurably different
outcomes. The closed-form OLS estimator IS the shortcut null model — every probe is compared
against it.

## ★ Second-order consequence of flatness (a finding) ★
**The flatness that compresses the SFT-vs-ICL gap also compresses the structure-vs-shortcut gap
in UTILITY space.** Demonstrated with a non-stationary probe (σ(t) ramp so c*(t) varies):

- c*(t) ramps **2.0 → 0.5** across the episode (4× swing in the optimal coefficient).
- Constant-c shortcut vs time-varying optimum: utility gap only **0.81%** (benign) — flatness
  absorbs even a 4× temporal variation; the test is nearly vacuous in utility space.
- **Same manipulation in POLICY space: RMSE(constant vs c*(t)) = 0.47 (47% relative)** —
  trivially detectable.

**=> Read structural probes in policy / prediction space, not utility space.** This is the same
escape that made claim 2 work (coefficient RMSE is flatness-free). Utility-space metrics are
flatness-limited; policy/prediction-space metrics are not.

## Probe menu (all read in prediction/policy space)
| Probe | MDP facet | manipulation | shortcut | structure | read-out |
|-------|-----------|--------------|----------|-----------|----------|
| S1 | Bellman/temporal | σ(t) ramp → c*(t) | flat ĉ(t) | ĉ(t) tracks c*(t) | coeff RMSE vs t |
| S2 | compounding credit assignment | autonomous closed-loop rollout | closed≈open | closed-loop **state** drift grows | state divergence over horizon |
| S3 | learned dynamics (world model) | counterfactual action → predict return | flat response | slope=(μ−r)dt/W (+ sharp kink) | return-prediction (IMPLEMENTED) |
| S4 | pure temporal (no scalar) | exponential utility, u*(t) state-indep | can't/flat | tracks u*(t) | action-vs-schedule RMSE |

S2 is the most central to "sequential": only the *closed-loop* model reveals whether it handles
the states its OWN errors produce (distribution shift) — something a one-shot estimator cannot
exhibit. S4 (exp utility) is the case with NO scalar to shortcut to, so success necessarily means
temporal structure was used — it is the *opposite* of a trivial control.

## S3 implemented (outcome analysis) — validated with mocks
`outcome_analysis.py` probes the transition kernel via counterfactual actions. Merton signature:
`E[R|W,A] = (A(μ−r)+rW−pen)dt/W`, slope `(μ−r)dt/W`, KINKS at A=W in the sharp regime
(slope drops by `s·dt`). Validated on mock models (CPU, no LLM):

| regime | mock | slope_below (theory) | pred RMSE | kink_drop (theory) |
|--------|------|----------------------|-----------|--------------------|
| benign | ideal | 0.00734 (0.00734) | 0.000 | 0.000 (0.000) |
| benign | shortcut | 0.000 | 0.005 | 0.000 |
| sharp | ideal | — | 0.000 | **0.040 (0.040)** |
| sharp | shortcut | 0.000 | 0.015 | 0.000 |
| sharp | no_kink | 0.00723 (0.00723) | 0.018 | **0.000 (0.040)** |

Three-way separation confirmed: no-dynamics (shortcut) vs linear-dynamics-only (no_kink) vs
full-dynamics-with-cost-structure (ideal). The sharp-regime **kink_drop** distinguishes a model
that learned the borrowing-cost structure from one that learned only the linear part — a signal
unique to the sharp regime. GPU runner: `run_outcome_analysis.py` (SFT / ICL-only).

## Caveat
These probes separate *structure-of-some-kind* from *scalar shortcut* (the live question). They do
NOT isolate "learned value iteration specifically" — that needs return-conditioning
(Decision-Transformer style), a larger build, noted as future work.
