"""Task prior and evaluation tiers (D1, D5).

Training tasks: pi* ~ Uniform[prior.lo, prior.hi], sampled iid.
Evaluation tiers use DETERMINISTIC linspaces so shrinkage regressions
(pi_hat vs pi*) are well-conditioned and reproducible:
    tier1       : interpolation, linspace over [prior.lo, prior.hi]
    tier2_low   : extrapolation below support
    tier2_high  : extrapolation above support
    tier3       : sign-flip (pi* < 0; shorting never demonstrated in training)
"""
from __future__ import annotations

from typing import List

import numpy as np

from .config import ExperimentConfig
from .env import Task


def sample_training_tasks(cfg: ExperimentConfig, n_tasks: int,
                          rng: np.random.Generator) -> List[Task]:
    ps = rng.uniform(cfg.prior.lo, cfg.prior.hi, size=n_tasks)
    return [Task(task_id=f"train_{i:05d}", tier="train", pistar=float(p))
            for i, p in enumerate(ps)]


def build_eval_tiers(cfg: ExperimentConfig) -> List[Task]:
    e = cfg.eval
    tasks: List[Task] = []

    def add(tier: str, lo: float, hi: float, n: int) -> None:
        for i, p in enumerate(np.linspace(lo, hi, n)):
            tasks.append(Task(task_id=f"{tier}_{i:04d}", tier=tier, pistar=float(p)))

    add("tier1", cfg.prior.lo, cfg.prior.hi, e.n_tier1)
    add("tier2_low", e.tier2_low_range[0], e.tier2_low_range[1], e.n_tier2_low)
    add("tier2_high", e.tier2_high_range[0], e.tier2_high_range[1], e.n_tier2_high)
    add("tier3", e.tier3_range[0], e.tier3_range[1], e.n_tier3)
    return tasks
