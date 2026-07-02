#!/usr/bin/env python3
"""Figures from eval outputs.

  python scripts/make_plots.py --data data_main --out figs \
      --results results/q4_sft_n3200 results/q2_icl results/q1_sft \
      --ntask "100=results/q4_sft_n0100" "400=results/q4_sft_n0400" \
              "1600=results/q4_sft_n1600" "3200=results/q4_sft_n3200"
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from merton_meta.config import ExperimentConfig  # noqa: E402
from merton_meta.metrics import load_jsonl, loglog_slope  # noqa: E402


def _load_results(d):
    return load_jsonl(os.path.join(d, "results.jsonl"))


def plot_mse_vs_ktest(results_by_arm, cfg, out):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    plotted = False
    for arm, rows in results_by_arm.items():
        rows = [r for r in rows if r["tier"] in ("tier1", "q2_fixed") and r["ktest"] > 0]
        if not rows:
            continue
        by_k = defaultdict(list)
        for r in rows:
            by_k[r["ktest"]].append((r["pi_hat"] - r["pistar"]) ** 2)
        ks = np.array(sorted(by_k))
        mses = np.array([np.mean(by_k[k]) for k in ks])
        sl = loglog_slope(ks, mses)["slope"]
        ax.plot(ks, mses, "o-", label=f"{arm} (slope {sl:.2f})")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ks = np.array(sorted({k for rows in results_by_arm.values()
                          for r in rows if r["ktest"] > 0
                          for k in [r["ktest"]]}))
    if ks.size:
        bayes = cfg.data.sigma_label ** 2 / (ks * cfg.market.n_steps)
        ax.plot(ks, bayes, "k--", label="Bayes floor $\\sigma^2/(KT)$ (slope -1)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$K_{test}$ support trajectories"); ax.set_ylabel("MSE$(\\hat\\pi, \\pi^*)$")
    ax.set_title("In-context estimation error vs $K_{test}$")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out, "mse_vs_ktest.png"), dpi=160); plt.close(fig)


def plot_shrinkage(results_by_arm, out):
    for arm, rows in results_by_arm.items():
        rows = [r for r in rows if r["tier"].startswith("tier")]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(6, 4.6))
        colors = {"tier1": "tab:blue", "tier2_low": "tab:orange",
                  "tier2_high": "tab:red", "tier3": "tab:purple"}
        for tier, c in colors.items():
            rs = [r for r in rows if r["tier"] == tier]
            if not rs:
                continue
            p = [r["pistar"] for r in rs]; h = [r["pi_hat"] for r in rs]
            ax.scatter(p, h, s=12, alpha=0.7, color=c, label=f"{tier} (model)")
            b = [r["bayes_mean"] for r in rs if r.get("bayes_mean") is not None]
            if b:
                ax.scatter(p[: len(b)], b, s=8, alpha=0.5, color=c, marker="x")
        lim = ax.get_xlim()
        ax.plot(lim, lim, "k-", lw=0.8, label="identity ($\\pi^*$)")
        ax.axvspan(0.30, 0.80, color="grey", alpha=0.12, label="training prior support")
        ax.set_xlabel("$\\pi^*$"); ax.set_ylabel("$\\hat\\pi$")
        ax.set_title(f"Shrinkage / extrapolation: {arm}  (x = Bayes oracle)")
        ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(os.path.join(out, f"shrinkage_{arm}.png"), dpi=160); plt.close(fig)


def plot_ntask(ntask_pairs, out):
    if not ntask_pairs:
        return
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for tier in ("tier1",):
        ns, rm, lo, hi = [], [], [], []
        for n, d in sorted(ntask_pairs, key=lambda p: p[0]):
            with open(os.path.join(d, "summary.json")) as f:
                s = json.load(f)
            gs = [g for k, g in s["groups"].items()
                  if g["tier"] == tier and g["ktest"] == 2]
            if not gs:
                continue
            g = gs[0]["rmse_true"]
            ns.append(n); rm.append(g["rmse"])
            lo.append(g.get("rmse_ci_lo", g["rmse"]))
            hi.append(g.get("rmse_ci_hi", g["rmse"]))
        if ns:
            ax.errorbar(ns, rm, yerr=[np.array(rm) - np.array(lo),
                                      np.array(hi) - np.array(rm)],
                        fmt="o-", capsize=3, label=f"{tier}, $K_{{test}}=2$")
    ax.set_xscale("log")
    ax.set_xlabel("number of training tasks $N_{task}$")
    ax.set_ylabel("RMSE$(\\hat\\pi,\\pi^*)$")
    ax.set_title("Replication axis: performance vs task count")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out, "rmse_vs_ntask.png"), dpi=160); plt.close(fig)


def plot_oracle_tracking(results_by_arm, out):
    for arm, rows in results_by_arm.items():
        rows = [r for r in rows if r.get("bayes_mean") is not None]
        if not rows:
            continue
        tiers = sorted({r["tier"] for r in rows})
        db, ds = [], []
        for tier in tiers:
            rs = [r for r in rows if r["tier"] == tier]
            db.append(np.sqrt(np.mean([(r["pi_hat"] - r["bayes_mean"]) ** 2 for r in rs])))
            ds.append(np.sqrt(np.mean([(r["pi_hat"] - r["sample_mean"]) ** 2 for r in rs])))
        x = np.arange(len(tiers)); w = 0.38
        fig, ax = plt.subplots(figsize=(6, 4.0))
        ax.bar(x - w / 2, db, w, label="RMSE to Bayes (prior)")
        ax.bar(x + w / 2, ds, w, label="RMSE to sample mean (algorithm)")
        ax.set_xticks(x); ax.set_xticklabels(tiers, fontsize=8)
        ax.set_ylabel("RMSE"); ax.set_title(f"Which oracle does it track? {arm}")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(os.path.join(out, f"oracle_tracking_{arm}.png"), dpi=160)
        plt.close(fig)


def plot_q1(results_by_arm, out):
    for arm, rows in results_by_arm.items():
        q1 = [r for r in rows if r["tier"] == "q1_fixed"]
        if not q1:
            continue
        fig, ax = plt.subplots(figsize=(6, 4.0))
        ax.hist([r["pi_hat"] for r in q1], bins=20)
        ax.axvline(q1[0]["pistar"], color="k", ls="--", label="$\\pi^*$")
        ax.set_xlabel("$\\hat\\pi$ (per CRN path)"); ax.set_title(f"Q1 fixed MDP: {arm}")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(out, f"q1_pihat_{arm}.png"), dpi=160); plt.close(fig)
        pf = os.path.join([d for d, a in RESULT_DIRS if a == arm][0],
                          "probe_results.jsonl") if RESULT_DIRS else None
        if pf and os.path.exists(pf):
            pr = load_jsonl(pf)
            fig, ax = plt.subplots(figsize=(6, 4.0))
            for r in pr:
                ax.plot(r["x_values"], r["actions_by_x"], "o-", alpha=0.7)
            ax.axhline(pr[0]["pistar"], color="k", ls="--", label="$\\pi^*$")
            ax.set_xlabel("counterfactual log-wealth $x$")
            ax.set_ylabel("chosen action")
            ax.set_title(f"State-independence probe: {arm}")
            ax.legend(); fig.tight_layout()
            fig.savefig(os.path.join(out, f"q1_probe_{arm}.png"), dpi=160)
            plt.close(fig)


RESULT_DIRS = []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", nargs="*", default=[],
                    help="results dirs (arm read from rows)")
    ap.add_argument("--ntask", nargs="*", default=[],
                    help="pairs N=results_dir for the N_task curve")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = ExperimentConfig.from_json(os.path.join(args.data, "config.json"))

    results_by_arm = {}
    for d in args.results:
        rows = _load_results(d)
        if rows:
            arm = rows[0].get("arm", os.path.basename(d))
            results_by_arm[arm] = rows
            RESULT_DIRS.append((d, arm))
    ntask_pairs = []
    for spec in args.ntask:
        n, d = spec.split("=", 1)
        ntask_pairs.append((int(n), d))

    plot_mse_vs_ktest(results_by_arm, cfg, args.out)
    plot_shrinkage(results_by_arm, args.out)
    plot_oracle_tracking(results_by_arm, args.out)
    plot_ntask(ntask_pairs, args.out)
    plot_q1(results_by_arm, args.out)
    print(f"[make_plots] figures written to {args.out}/")


if __name__ == "__main__":
    main()
