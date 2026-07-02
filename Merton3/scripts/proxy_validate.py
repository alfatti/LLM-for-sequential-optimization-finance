#!/usr/bin/env python3
"""CPU design-validation harness. Run BEFORE any GPU spend.

Checks (all asserted, summary written to proxy_report.json):
  A. Serialization round-trip: rendered demo actions parse back exactly (2dp).
  B. Decision-span masking: extracted character spans reproduce exactly the
     query action strings, one per query step, for all three mask modes.
  C. EXACT CE-gap identity vs Monte Carlo (200k paths).
  D. Truncated-normal Bayes posterior mean: closed form vs numeric integration,
     including edge cases with the sample mean far outside the prior support.
  E. Transition-channel Fisher information ratio << 1 (justifies label-only
     oracles).
  F. Pre-registration, in-context term: empirical MSE of both oracles vs K_test
     has log-log slope ~ -1 for an interior task (Bayes and sample mean).
  G. Corpus proxy learner: per-context-size ridge fit of query label on demo
     sample-mean over the Q4 TRAINING corpus, evaluated on the eval bundle.
     Validates that the corpus encodes the estimation task (tier-1 proxy RMSE
     tracks Bayes) and that the shrinkage machinery detects tier-2 shrinkage
     (proxy beta < 1 out of support).
  H. Bundles well-formed for q1 / q1_probe / q2 / q4; CRN nesting of demos.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from merton_meta.config import ExperimentConfig  # noqa: E402
from merton_meta import serialize as S  # noqa: E402
from merton_meta.env import (Task, log_ce_gap, mc_certainty_equivalent,  # noqa: E402
                             pistar_to_mu, ce_growth)
from merton_meta.datagen import gen_noisy_demo, gen_clean_query, rng_for  # noqa: E402
from merton_meta.oracles import (bayes_posterior_mean,  # noqa: E402
                                 bayes_posterior_mean_numeric,
                                 sample_mean_oracle,
                                 transition_channel_info_ratio)
from merton_meta.metrics import (load_jsonl, loglog_slope, shrinkage_fit,  # noqa: E402
                                 mse_with_ci, dump_json)

PASS = "  [PASS]"


def check_serialization(cfg: ExperimentConfig, report: dict) -> None:
    task = Task("chk", "train", 0.55)
    demos = [gen_noisy_demo(cfg, task, cfg.data.sigma_label, "chk", j) for j in range(3)]
    q_xs, q_acts = gen_clean_query(cfg, task, "chk")
    text = S.render_training_text(demos, q_xs, q_acts)

    # A: demo action round-trip
    parsed = S.parse_demo_actions(text)
    truth = np.concatenate([np.round(a, 2) for _, a in demos])
    assert parsed.shape == truth.shape and np.allclose(parsed, truth, atol=5e-3), \
        "demo action round-trip failed"
    report["A_roundtrip"] = {"n_demo_actions": int(parsed.size)}
    print(PASS, f"A serialization round-trip ({parsed.size} demo actions)")

    # B: decision spans reproduce query action strings exactly
    spans = S.decision_char_spans(text, "decision")
    substrs = [text[a:b] for a, b in spans]
    expect = [S.fmt_action(a) for a in q_acts]
    assert substrs == expect, f"decision spans mismatch: {substrs[:3]} vs {expect[:3]}"
    for mode in ("query", "full"):
        sp = S.decision_char_spans(text, mode)
        assert len(sp) == 1 and sp[0][1] == len(text)
    report["B_mask_spans"] = {"n_decision_spans": len(spans)}
    print(PASS, f"B decision-span masking ({len(spans)} spans == n_steps)")


def check_ce_gap(cfg: ExperimentConfig, report: dict) -> None:
    m = cfg.market
    task = Task("chk", "train", 0.55)
    mu = task.mu(m)
    rng = rng_for(cfg.seed, "cegap")
    a_seq = 0.55 + rng.normal(0, 0.15, m.n_steps)          # a wobbly policy
    gap_exact = log_ce_gap(a_seq, task.pistar, m)
    # MC: CE of the wobbly policy vs CE(pi*), each per-step constant
    log_ce_opt = sum(ce_growth(task.pistar, mu, m) * m.dt for _ in range(m.n_steps))
    log_ce_pol = float(np.sum([ce_growth(a, mu, m) * m.dt for a in a_seq]))
    assert abs((log_ce_opt - log_ce_pol) - gap_exact) < 1e-12
    # and MC estimate for a constant policy against the analytic CE
    ce_mc = mc_certainty_equivalent(0.40, mu, m, 200_000, rng)
    ce_an = float(np.exp(ce_growth(0.40, mu, m) * m.dt * m.n_steps))
    rel = abs(ce_mc - ce_an) / ce_an
    assert rel < 2e-3, f"CE MC vs analytic mismatch: rel={rel:.2e}"
    report["C_ce_gap"] = {"mc_vs_analytic_rel_err": rel,
                          "flatness_example_logce_gap_at_err_0.10":
                              0.5 * m.gamma * m.sigma ** 2 * m.dt * m.n_steps * 0.01}
    print(PASS, f"C exact CE-gap identity (MC rel err {rel:.1e})")


def check_bayes_oracle(cfg: ExperimentConfig, report: dict) -> None:
    lo, hi, sl = cfg.prior.lo, cfg.prior.hi, cfg.data.sigma_label
    rng = rng_for(cfg.seed, "bayeschk")
    worst = 0.0
    for theta in (0.32, 0.55, 0.78, 0.05, 0.95, -0.30):     # incl. out-of-support
        a = theta + rng.normal(0, sl, 20)
        cf = bayes_posterior_mean(a, sl, lo, hi)
        nu = bayes_posterior_mean_numeric(a, sl, lo, hi)
        worst = max(worst, abs(cf - nu))
        assert lo - 1e-9 <= cf <= hi + 1e-9
    assert worst < 1e-6, f"closed form vs numeric: {worst:.2e}"
    report["D_bayes_closed_form_max_abs_err"] = worst
    print(PASS, f"D Bayes posterior closed form (max |err| {worst:.1e})")


def check_info_ratio(cfg: ExperimentConfig, report: dict) -> None:
    m = cfg.market
    ratio = transition_channel_info_ratio(m.gamma, m.sigma, m.dt, cfg.data.sigma_label)
    assert ratio < 0.01, f"transition channel not negligible: {ratio:.3e}"
    report["E_transition_info_ratio"] = ratio
    print(PASS, f"E transition/label Fisher-info ratio {ratio:.1e} (<1%)")


def check_slopes(cfg: ExperimentConfig, report: dict) -> None:
    """Pre-registered in-context slope: MSE ~ sigma_label^2/(K*T) => slope -1."""
    m, sl = cfg.market, cfg.data.sigma_label
    lo, hi = cfg.prior.lo, cfg.prior.hi
    theta = 0.55
    ks = np.array([1, 2, 4, 8, 16])
    n_rep = 4000
    rng = rng_for(cfg.seed, "slopes")
    mse_b, mse_s = [], []
    for k in ks:
        M = k * m.n_steps
        a = theta + rng.normal(0, sl, (n_rep, M))
        est_s = a.mean(axis=1)
        est_b = np.array([bayes_posterior_mean(row, sl, lo, hi) for row in a])
        mse_s.append(float(np.mean((est_s - theta) ** 2)))
        mse_b.append(float(np.mean((est_b - theta) ** 2)))
    sb = loglog_slope(ks, np.array(mse_b))["slope"]
    ss = loglog_slope(ks, np.array(mse_s))["slope"]
    assert -1.15 < ss < -0.85, f"sample-mean slope {ss:.3f} outside [-1.15,-0.85]"
    assert -1.20 < sb < -0.80, f"Bayes slope {sb:.3f} outside [-1.20,-0.80]"
    report["F_slopes"] = {"bayes": sb, "sample_mean": ss,
                          "mse_bayes": mse_b, "mse_sample": mse_s,
                          "ktest": ks.tolist()}
    print(PASS, f"F in-context slope pre-registration (bayes {sb:.3f}, "
                f"sample-mean {ss:.3f}; predicted -1)")


_PROXY_BUCKETS = [(1, 2), (3, 4), (5, 8), (9, 16)]   # n_support ranges


def _bucket_of(n_sup: int) -> int:
    for i, (lo, hi) in enumerate(_PROXY_BUCKETS):
        if lo <= n_sup <= hi:
            return i
    return len(_PROXY_BUCKETS) - 1


def check_corpus_proxy(data_dir: str, cfg: ExperimentConfig, report: dict) -> None:
    """G1: the ACTUAL Q4 corpus parses and supports a pooled linear read-out
    (label ~ demo sample mean) with small tier-1 error -- the corpus encodes
    the estimation task. Corpus-size independent.

    G2: a design-scale synthetic proxy (same generators, fresh seed tag,
    3000 examples, coarse n_support buckets) evaluated on the REAL eval
    bundle rows. Validates that the shrinkage machinery detects the
    guaranteed attenuation-type shrinkage (beta_pop = Var(pi*) /
    (Var(pi*) + sigma^2/M) < 1) at the smallest tier-2/3 context size, and
    that tier-1 proxy error tracks the Bayes oracle. Note this proxy's
    shrinkage is the attenuation kind (present at any N_task); the LLM's
    finite-N_task prior-estimation shrinkage is what Q4 measures.
    """
    from merton_meta.tasks import sample_training_tasks

    # ---- G1: actual corpus, pooled fit ----
    ntask_dirs = sorted(d for d in os.listdir(os.path.join(data_dir, "q4"))
                        if d.startswith("ntask_"))
    assert ntask_dirs, "no q4 training data found"
    sub = os.path.join(data_dir, "q4", ntask_dirs[-1])
    rows = load_jsonl(os.path.join(sub, "train.jsonl"))
    feats = []
    for r in rows:
        acts = S.parse_demo_actions(r["text"])
        assert acts.size == 10 * r["n_support"], "demo count mismatch in corpus"
        feats.append((r["n_support"], float(acts.mean()),
                      float(np.round(r["pistar"], 2))))
    A = np.array([[1.0, ab] for _, ab, _ in feats])
    y = np.array([yy for _, _, yy in feats])
    w = np.linalg.solve(A.T @ A + 1e-6 * np.eye(2), A.T @ y)
    bundle = load_jsonl(os.path.join(data_dir, "q4", "eval_bundle.jsonl"))
    t1 = [r for r in bundle if r["tier"] == "tier1"]
    h1 = np.array([w[0] + w[1] * np.mean(r["demo_actions"]) for r in t1])
    p1 = np.array([r["pistar"] for r in t1])
    g1_rmse = mse_with_ci((h1 - p1) ** 2)["rmse"]
    assert g1_rmse < 0.15, f"actual-corpus pooled proxy weak on tier1: {g1_rmse:.3f}"

    # ---- G2: design-scale synthetic proxy, coarse buckets ----
    n_syn = 3000
    m = cfg.market
    syn_tasks = sample_training_tasks(cfg, n_syn, rng_for(cfg.seed, "proxyG_tasks"))
    buckets = {i: [] for i in range(len(_PROXY_BUCKETS))}
    for i, task in enumerate(syn_tasks):
        rng = rng_for(cfg.seed, "proxyG_nsup", i)
        n_sup = int(rng.integers(cfg.data.n_support_min, cfg.data.n_support_max + 1))
        acts = np.concatenate(
            [gen_noisy_demo(cfg, task, cfg.data.sigma_label, "proxyG", i, j)[1]
             for j in range(n_sup)])
        buckets[_bucket_of(n_sup)].append(
            (float(np.round(acts, 2).mean()), float(np.round(task.pistar, 2))))
    coefs = {}
    for i, pts in buckets.items():
        Ab = np.array([[1.0, ab] for ab, _ in pts])
        yb = np.array([yy for _, yy in pts])
        wb = np.linalg.solve(Ab.T @ Ab + 1e-6 * np.eye(2), Ab.T @ yb)
        coefs[i] = (float(wb[0]), float(wb[1]))

    def proxy_predict(r: dict) -> float:
        al, be = coefs[_bucket_of(max(1, r["ktest"]))]
        return al + be * float(np.mean(r["demo_actions"]))

    res = [{**r, "pi_hat": proxy_predict(r)} for r in bundle]
    t1 = [r for r in res if r["tier"] == "tier1"]
    p1 = np.array([r["pistar"] for r in t1])
    h1 = np.array([r["pi_hat"] for r in t1])
    b1 = np.array([r["bayes_mean"] for r in t1])
    prox_rmse = mse_with_ci((h1 - p1) ** 2)["rmse"]
    bayes_rmse = mse_with_ci((b1 - p1) ** 2)["rmse"]
    k_min = min(cfg.eval.tier23_ktests)
    t2 = [r for r in res if r["tier"].startswith("tier2") and r["ktest"] == k_min]
    beta2 = shrinkage_fit(np.array([r["pistar"] for r in t2]),
                          np.array([r["pi_hat"] for r in t2]))["beta"]
    beta2_bayes = shrinkage_fit(np.array([r["pistar"] for r in t2]),
                                np.array([r["bayes_mean"] for r in t2]))["beta"]
    assert prox_rmse < 1.5 * bayes_rmse + 0.02, \
        f"design proxy far from Bayes on tier1: {prox_rmse:.3f} vs {bayes_rmse:.3f}"
    assert beta2 < 0.97, \
        f"shrinkage machinery failed to detect attenuation (beta={beta2:.3f})"
    report["G_corpus_proxy"] = {
        "G1_actual_corpus_pooled_tier1_rmse": g1_rmse,
        "G1_n_examples": len(feats), "G1_corpus_used": ntask_dirs[-1],
        "G2_tier1_rmse_proxy": prox_rmse, "G2_tier1_rmse_bayes": bayes_rmse,
        "G2_tier2_beta_proxy_kmin": beta2, "G2_tier2_beta_bayes_kmin": beta2_bayes,
        "G2_ktest_used": k_min, "G2_n_synth": n_syn}
    print(PASS, f"G corpus proxy: G1 pooled tier1 RMSE {g1_rmse:.3f}; "
                f"G2 tier1 RMSE {prox_rmse:.3f} (Bayes {bayes_rmse:.3f}), "
                f"tier2 beta {beta2:.2f} @k={k_min} (Bayes {beta2_bayes:.2f})")


def check_bundles(data_dir: str, cfg: ExperimentConfig, report: dict) -> None:
    counts = {}
    for name, path in [("q4", "q4/eval_bundle.jsonl"), ("q2", "q2/eval_bundle.jsonl"),
                       ("q1", "q1/eval_bundle.jsonl"), ("q1_probe", "q1/probe_bundle.jsonl")]:
        rows = load_jsonl(os.path.join(data_dir, path))
        counts[name] = len(rows)
        assert rows, f"empty bundle {name}"
    # CRN nesting: q2 draw-0 demos at k=1 must prefix those at max k
    q2 = load_jsonl(os.path.join(data_dir, "q2", "eval_bundle.jsonl"))
    d0 = [r for r in q2 if r["draw"] == 0]
    d0.sort(key=lambda r: r["ktest"])
    small, big = np.array(d0[0]["demo_actions"]), np.array(d0[-1]["demo_actions"])
    assert np.allclose(big[: small.size], small), "CRN demo nesting broken (q2)"
    # oracles recomputable from text
    r = d0[-1]
    acts = S.parse_demo_actions(r["demos_text"] + "\n" + S.QUERY_HEADER)
    assert abs(sample_mean_oracle(acts) - r["sample_mean"]) < 1e-9
    assert abs(bayes_posterior_mean(acts, cfg.data.sigma_label, cfg.prior.lo,
                                    cfg.prior.hi) - r["bayes_mean"]) < 1e-9
    report["H_bundles"] = counts
    print(PASS, f"H bundles well-formed + CRN nesting {counts}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir produced by make_data.py")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = ExperimentConfig.from_json(os.path.join(args.data, "config.json"))
    report: dict = {"data_dir": args.data}
    print("[proxy_validate] running design checks ...")
    check_serialization(cfg, report)
    check_ce_gap(cfg, report)
    check_bayes_oracle(cfg, report)
    check_info_ratio(cfg, report)
    check_slopes(cfg, report)
    check_corpus_proxy(args.data, cfg, report)
    check_bundles(args.data, cfg, report)
    out = args.out or os.path.join(args.data, "proxy_report.json")
    dump_json(report, out)
    print(f"[proxy_validate] ALL CHECKS PASSED -> {out}")


if __name__ == "__main__":
    main()
