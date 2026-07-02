"""Data generation for Q1 (fixed-MDP SFT), Q2 (fixed-MDP ICL), Q4 (meta-SFT).

CRN discipline: every stochastic object is generated from a deterministic RNG
keyed by string tags via `rng_for(cfg.seed, *tags)`. Consequences:
  * Q4 demo sets are NESTED in K_test (demo j of task i is identical whether
    the context uses 1, 2, ..., or 16 demos), so K_test comparisons are paired.
  * Q4 training task sets are NESTED across the N_task sweep (prefixes of one
    3200-task draw), so the N_task curve is not confounded by task resampling.
  * Query-path Brownian increments are fixed per (row), independent of the
    acting policy, so model / oracles / baselines face identical shocks.

Bundle row schema (jsonl), shared by the unified eval harness:
  {row_id, exp, tier, task_id, pistar, ktest, draw, header, demos_text,
   demo_actions[], bayes_mean, sample_mean, query_seed_tag}
Q1 rows have ktest=0 and demos_text="". Probe rows (Q1 only) have their own
schema: {row_id, exp:"q1_probe", pistar, t_probe, history_text, x_values[]}.

Training row schema: {text, task_id, pistar, n_support, split_tag}.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import ExperimentConfig
from .env import Task, pistar_to_mu, rollout
from .oracles import bayes_posterior_mean, sample_mean_oracle
from .serialize import (params_header, parse_demo_actions, render_demos_block,
                        render_training_text, render_traj_lines, QUERY_HEADER)
from .tasks import build_eval_tiers, sample_training_tasks


# ----------------------------------------------------------------------------- 
# CRN registry
# -----------------------------------------------------------------------------

def rng_for(seed: int, *tags) -> np.random.Generator:
    key = "|".join(str(t) for t in tags)
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little") ^ (seed & (2**64 - 1)))


def query_z(cfg: ExperimentConfig, tag: str) -> np.ndarray:
    """Brownian increments for a closed-loop query path (policy-independent CRN)."""
    return rng_for(cfg.seed, "queryz", tag).standard_normal(cfg.market.n_steps)


# -----------------------------------------------------------------------------
# Trajectory generators
# -----------------------------------------------------------------------------

def gen_noisy_demo(cfg: ExperimentConfig, task: Task, sigma_label: float,
                   *tags) -> Tuple[np.ndarray, np.ndarray]:
    """Support demo: EXECUTED actions a_t = pi* + eps_t (unclipped), consistent
    wealth path. Returns (xs[T+1], actions[T])."""
    m = cfg.market
    rng = rng_for(cfg.seed, "demo", *tags)
    eps = rng.normal(0.0, sigma_label, size=m.n_steps) if sigma_label > 0 else np.zeros(m.n_steps)
    acts = task.pistar + eps
    z = rng.standard_normal(m.n_steps)
    xs = rollout(acts, z, task.mu(m), m)
    return xs, acts


def grid_round(a: float) -> float:
    return float(f"{a:+.2f}")


def gen_clean_query(cfg: ExperimentConfig, task: Task,
                    *tags) -> Tuple[np.ndarray, np.ndarray]:
    """Training query: executed and labeled with the CLEAN pi* rounded to the
    grid (DPT-faithful: the target is the optimal action)."""
    m = cfg.market
    rng = rng_for(cfg.seed, "query", *tags)
    a = grid_round(task.pistar)
    acts = np.full(m.n_steps, a)
    z = rng.standard_normal(m.n_steps)
    xs = rollout(acts, z, task.mu(m), m)
    return xs, acts


# -----------------------------------------------------------------------------
# Q4: meta training corpus + eval bundle
# -----------------------------------------------------------------------------

def build_q4_training(cfg: ExperimentConfig, out_dir: str) -> Dict[str, int]:
    d = cfg.data
    n_max = max(d.ntask_sweep)
    all_tasks = sample_training_tasks(cfg, n_max, rng_for(cfg.seed, "q4tasks"))
    counts = {}
    for n_task in d.ntask_sweep:
        tasks = all_tasks[:n_task]                      # nested prefixes
        n_val = max(1, int(round(d.val_task_frac * n_task)))
        val_ids = {t.task_id for t in tasks[-n_val:]}   # split BY TASK
        sub = os.path.join(out_dir, "q4", f"ntask_{n_task:05d}")
        os.makedirs(sub, exist_ok=True)
        n_rows = {"train": 0, "val": 0}
        with open(os.path.join(sub, "train.jsonl"), "w") as ftr, \
             open(os.path.join(sub, "val.jsonl"), "w") as fva:
            for task in tasks:
                for e in range(d.examples_per_task):
                    rng = rng_for(cfg.seed, "nsup", task.task_id, e)
                    n_sup = int(rng.integers(d.n_support_min, d.n_support_max + 1))
                    demos = [gen_noisy_demo(cfg, task, d.sigma_label,
                                            "q4", task.task_id, e, j)
                             for j in range(n_sup)]
                    q_xs, q_acts = gen_clean_query(cfg, task, "q4", task.task_id, e)
                    text = render_training_text(demos, q_xs, q_acts)
                    row = {"text": text, "task_id": task.task_id,
                           "pistar": task.pistar, "n_support": n_sup}
                    split = "val" if task.task_id in val_ids else "train"
                    (fva if split == "val" else ftr).write(json.dumps(row) + "\n")
                    n_rows[split] += 1
        counts[f"ntask_{n_task:05d}"] = n_rows
    return counts


def _bundle_row(cfg: ExperimentConfig, exp: str, task: Task, ktest: int,
                draw: int, demos: Sequence[Tuple[np.ndarray, np.ndarray]],
                header: str = "") -> dict:
    demos_text = render_demos_block(demos[:ktest]) if ktest > 0 else ""
    demo_acts = parse_demo_actions(demos_text + "\n" + QUERY_HEADER) \
        if ktest > 0 else np.array([])
    row = {
        "row_id": f"{exp}_{task.task_id}_k{ktest:02d}_d{draw:02d}",
        "exp": exp, "tier": task.tier, "task_id": task.task_id,
        "pistar": task.pistar, "ktest": ktest, "draw": draw,
        "header": header, "demos_text": demos_text,
        "demo_actions": [float(a) for a in demo_acts],
        "sample_mean": (sample_mean_oracle(demo_acts) if ktest > 0 else None),
        "bayes_mean": (bayes_posterior_mean(demo_acts, cfg.data.sigma_label,
                                            cfg.prior.lo, cfg.prior.hi)
                       if ktest > 0 else None),
        "query_seed_tag": f"{exp}|{task.task_id}|k{ktest}|d{draw}",
    }
    return row


def build_q4_eval_bundle(cfg: ExperimentConfig, out_dir: str) -> int:
    e = cfg.eval
    tasks = build_eval_tiers(cfg)
    os.makedirs(os.path.join(out_dir, "q4"), exist_ok=True)
    path = os.path.join(out_dir, "q4", "eval_bundle.jsonl")
    n = 0
    with open(path, "w") as f:
        for task in tasks:
            k_max = max(max(e.ktest_sweep), max(e.tier23_ktests))
            demos = [gen_noisy_demo(cfg, task, cfg.data.sigma_label,
                                    "q4eval", task.task_id, j)      # nested in j
                     for j in range(k_max)]
            ktests = e.ktest_sweep if task.tier == "tier1" else e.tier23_ktests
            for k in ktests:
                for d in range(e.n_query_paths):
                    f.write(json.dumps(_bundle_row(cfg, "q4", task, k, d, demos)) + "\n")
                    n += 1
    return n


# -----------------------------------------------------------------------------
# Q1: fixed-MDP SFT corpus + eval bundle + counterfactual state probe
# -----------------------------------------------------------------------------

def _fixed_task(cfg: ExperimentConfig) -> Task:
    return Task(task_id="fixed", tier="q1_fixed", pistar=cfg.fixed.pistar)


def build_q1_training(cfg: ExperimentConfig, out_dir: str) -> Dict[str, int]:
    fx = cfg.fixed
    task = _fixed_task(cfg)
    m = cfg.market
    header = params_header(task.mu(m), m.sigma, m.r, m.gamma) if fx.reveal_params else ""
    sub = os.path.join(out_dir, "q1")
    os.makedirs(sub, exist_ok=True)
    n_val = max(1, int(round(cfg.data.val_task_frac * fx.n_train_rollouts)))
    n_rows = {"train": 0, "val": 0}
    with open(os.path.join(sub, "train.jsonl"), "w") as ftr, \
         open(os.path.join(sub, "val.jsonl"), "w") as fva:
        for i in range(fx.n_train_rollouts):
            xs, acts = gen_noisy_demo(cfg, task, fx.sigma_label_q1, "q1", i)
            text = render_training_text([], xs, acts, header=header)
            row = {"text": text, "task_id": task.task_id,
                   "pistar": task.pistar, "n_support": 0}
            split = "val" if i >= fx.n_train_rollouts - n_val else "train"
            (fva if split == "val" else ftr).write(json.dumps(row) + "\n")
            n_rows[split] += 1
    return n_rows


def build_q1_eval_bundle(cfg: ExperimentConfig, out_dir: str) -> int:
    fx = cfg.fixed
    task = _fixed_task(cfg)
    m = cfg.market
    header = params_header(task.mu(m), m.sigma, m.r, m.gamma) if fx.reveal_params else ""
    sub = os.path.join(out_dir, "q1")
    os.makedirs(sub, exist_ok=True)
    n = 0
    with open(os.path.join(sub, "eval_bundle.jsonl"), "w") as f:
        for d in range(fx.n_eval_paths):
            row = _bundle_row(cfg, "q1", task, 0, d, [], header=header)
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


def build_q1_probe_bundle(cfg: ExperimentConfig, out_dir: str) -> int:
    """S3-style counterfactual state probe: a fixed optimal-policy history up to
    step t_probe-1, then the current log-wealth x is REPLACED by grid values.
    A state-independent learned policy shows ~zero action spread across x."""
    fx = cfg.fixed
    task = _fixed_task(cfg)
    m = cfg.market
    header = params_header(task.mu(m), m.sigma, m.r, m.gamma) if fx.reveal_params else ""
    a = grid_round(task.pistar)
    x_values = [float(v) for v in np.linspace(fx.probe_x_lo, fx.probe_x_hi, fx.probe_n_x)]
    sub = os.path.join(out_dir, "q1")
    os.makedirs(sub, exist_ok=True)
    n = 0
    with open(os.path.join(sub, "probe_bundle.jsonl"), "w") as f:
        for h in range(fx.probe_n_histories):
            rng = rng_for(cfg.seed, "probe", h)
            z = rng.standard_normal(m.n_steps)
            acts = np.full(fx.probe_t - 1, a)
            xs = rollout(np.concatenate([acts, np.full(m.n_steps - fx.probe_t + 1, a)]),
                         z, task.mu(m), m)[: fx.probe_t]  # x_0..x_{t_probe-1}
            hist_lines = render_traj_lines(xs, acts)
            parts = ([header] if header else []) + [QUERY_HEADER] + hist_lines
            row = {"row_id": f"q1probe_h{h:02d}", "exp": "q1_probe",
                   "pistar": task.pistar, "t_probe": fx.probe_t,
                   "history_text": "\n".join(parts), "x_values": x_values}
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


# -----------------------------------------------------------------------------
# Q2: fixed-MDP ICL bundle (base model; CIs across CRN context draws)
# -----------------------------------------------------------------------------

def build_q2_eval_bundle(cfg: ExperimentConfig, out_dir: str) -> int:
    task = Task(task_id="fixed", tier="q2_fixed", pistar=cfg.fixed.pistar)
    e, q2 = cfg.eval, cfg.q2
    k_max = max(e.ktest_sweep)
    sub = os.path.join(out_dir, "q2")
    os.makedirs(sub, exist_ok=True)
    n = 0
    with open(os.path.join(sub, "eval_bundle.jsonl"), "w") as f:
        for d in range(q2.n_context_draws):
            demos = [gen_noisy_demo(cfg, task, cfg.data.sigma_label, "q2", d, j)
                     for j in range(k_max)]                 # nested in j
            for k in e.ktest_sweep:
                f.write(json.dumps(_bundle_row(cfg, "q2", task, k, d, demos)) + "\n")
                n += 1
    return n


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def build_all(cfg: ExperimentConfig, out_dir: str, exps: Sequence[str]) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cfg.to_json(os.path.join(out_dir, "config.json"))
    report = {}
    if "q4" in exps:
        report["q4_train"] = build_q4_training(cfg, out_dir)
        report["q4_eval_rows"] = build_q4_eval_bundle(cfg, out_dir)
    if "q1" in exps:
        report["q1_train"] = build_q1_training(cfg, out_dir)
        report["q1_eval_rows"] = build_q1_eval_bundle(cfg, out_dir)
        report["q1_probe_rows"] = build_q1_probe_bundle(cfg, out_dir)
    if "q2" in exps:
        report["q2_eval_rows"] = build_q2_eval_bundle(cfg, out_dir)
    return report
