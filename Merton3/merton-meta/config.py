"""Experiment configuration for the Q4 meta-Merton experiment.

Design decisions (locked in-conversation, D1-D6):
  D1  Task prior directly on the identified scalar pi* = (mu-r)/(gamma sigma^2),
      Uniform[0.30, 0.80]. Only mu varies; r, sigma, gamma fixed. This keeps the
      in-context feature covariance approximately fixed across tasks (the paper's
      fixed-Lambda assumption) and makes the Bayes oracle exact.
  D2  Context = K_test noisy-optimal support demos, executed action a_t = pi* + eps_t,
      eps ~ N(0, sigma_label^2) iid per step. Task parameters are NEVER printed in
      the prompt (contamination guard). The word "Merton" never appears in prompts.
  D3  Output protocol = discretized action grid, constrained scoring over all
      candidate action strings, argmax. Grid extends BEYOND the training-support
      range ([-0.60, 1.20], step 0.01) so that Tier-2/3 clipping, if observed, is a
      model behavior rather than a grid artifact.
  D4  Sweeps: N_task in {100, 400, 1600, 3200} at K_test = 2 (replication of the
      paper's Fig. 2 protocol); K_test in {1,2,4,8,16} eval-only on each checkpoint.
      Number of support demos per TRAINING example is randomized Uniform{1..16} so
      the whole K_test sweep is in-format.
  D5  Generalization tiers: T1 interpolation pi* in [0.30,0.80]; T2 extrapolation
      [0.05,0.29] u [0.81,1.00]; T3 sign-flip [-0.50,-0.125]. Mechanism test =
      two-oracle comparison (truncated-normal Bayes posterior mean under the
      training prior vs prior-free sample mean) on CRN'd contexts.
  D6  Primary metric RMSE(pi_hat, pi*) with t-CIs; mechanism metric RMSE to each
      oracle; secondary metric = EXACT analytic certainty-equivalent gap (no MC
      needed for constant-parameter tasks; see env.py derivation).

Query labels in training are the CLEAN optimal action pi* rounded to the grid
(DPT-faithful: predict the optimal action given in-context data => the Bayes-
optimal predictor is the posterior over pi*). Support demos carry the noise.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class MarketConfig:
    """Fixed market/utility parameters (D1: only mu varies across tasks)."""
    r: float = 0.02            # risk-free rate (annual)
    sigma: float = 0.20        # risky-asset volatility (annual)
    gamma: float = 2.0         # CRRA coefficient
    dt: float = 0.1            # step size in years
    n_steps: int = 10          # steps per trajectory (horizon = n_steps * dt years)


@dataclass
class PriorConfig:
    """Training task prior over pi* (D1)."""
    lo: float = 0.30
    hi: float = 0.80


@dataclass
class GridConfig:
    """Action grid for constrained scoring (D3)."""
    lo: float = -0.60
    hi: float = 1.20
    step: float = 0.01


@dataclass
class DataConfig:
    sigma_label: float = 0.25       # demo label noise (load-bearing; established)
    n_support_min: int = 1          # randomized supports per training example (D4)
    n_support_max: int = 16
    examples_per_task: int = 4      # training (supports, query) prompts per task
    ntask_sweep: List[int] = field(default_factory=lambda: [100, 400, 1600, 3200])
    val_task_frac: float = 0.10     # validation split BY TASK (paper: 10% tasks)


@dataclass
class FixedTaskConfig:
    """Q1 (fixed-MDP SFT) and Q2 (fixed-MDP ICL) share this single task.
    pi* = 0.55: center of the training prior, interior (truncation never binds)."""
    pistar: float = 0.55
    n_train_rollouts: int = 512     # Q1 SFT corpus size (one trajectory per example)
    sigma_label_q1: float = 0.25    # Q1 behavior-policy noise; set 0.0 for the
                                    # noiseless arm (trivial-memorization control)
    reveal_params: bool = False     # Q1 contamination arm: print (mu,sigma,r,gamma)
    n_eval_paths: int = 50          # Q1 closed-loop CRN query paths (CIs across paths)
    probe_t: int = 6                # S3-style counterfactual state probe: step index
    probe_x_lo: float = -0.30       # counterfactual log-wealth grid
    probe_x_hi: float = 0.30
    probe_n_x: int = 13
    probe_n_histories: int = 3      # distinct CRN histories for the probe


@dataclass
class Q2Config:
    """Q2 = ICL-only (base model) on the fixed task. One task, so CIs come from
    repeated CRN context draws at each K_test."""
    n_context_draws: int = 50


@dataclass
class EvalConfig:
    ktest_sweep: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])  # Tier 1
    tier23_ktests: List[int] = field(default_factory=lambda: [2, 8])
    n_tier1: int = 100
    n_tier2_low: int = 50            # pi* in [0.05, 0.29]
    n_tier2_high: int = 50           # pi* in [0.81, 1.00]
    n_tier3: int = 40                # pi* in [-0.50, -0.125]
    tier2_low_range: List[float] = field(default_factory=lambda: [0.05, 0.29])
    tier2_high_range: List[float] = field(default_factory=lambda: [0.81, 1.00])
    tier3_range: List[float] = field(default_factory=lambda: [-0.50, -0.125])
    n_query_paths: int = 1           # closed-loop query rollouts per (task, ktest)


@dataclass
class ExperimentConfig:
    market: MarketConfig = field(default_factory=MarketConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    fixed: FixedTaskConfig = field(default_factory=FixedTaskConfig)
    q2: Q2Config = field(default_factory=Q2Config)
    seed: int = 20260702

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_json(path: str) -> "ExperimentConfig":
        with open(path) as f:
            d = json.load(f)
        return ExperimentConfig(
            market=MarketConfig(**d["market"]),
            prior=PriorConfig(**d["prior"]),
            grid=GridConfig(**d["grid"]),
            data=DataConfig(**d["data"]),
            eval=EvalConfig(**d["eval"]),
            fixed=FixedTaskConfig(**d.get("fixed", {})),
            q2=Q2Config(**d.get("q2", {})),
            seed=d.get("seed", 20260702),
        )


def smoke_config() -> ExperimentConfig:
    """Tiny preset for CPU validation and GPU smoke tests."""
    cfg = ExperimentConfig()
    cfg.data.ntask_sweep = [8]
    cfg.data.examples_per_task = 2
    cfg.eval.n_tier1 = 12
    cfg.eval.n_tier2_low = 6
    cfg.eval.n_tier2_high = 6
    cfg.eval.n_tier3 = 6
    cfg.eval.ktest_sweep = [1, 4, 16]
    cfg.eval.tier23_ktests = [4]
    cfg.fixed.n_train_rollouts = 16
    cfg.fixed.n_eval_paths = 6
    cfg.fixed.probe_n_histories = 2
    cfg.q2.n_context_draws = 6
    return cfg


def main_config() -> ExperimentConfig:
    return ExperimentConfig()
