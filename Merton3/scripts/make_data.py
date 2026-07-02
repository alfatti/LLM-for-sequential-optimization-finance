#!/usr/bin/env python3
"""Generate training corpora and evaluation bundles.

Examples:
    python scripts/make_data.py --preset smoke --exp all --out data_smoke
    python scripts/make_data.py --preset main  --exp all --out data_main
    python scripts/make_data.py --preset main  --exp q1 --out data_q1_reveal --reveal-params
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from merton_meta.config import ExperimentConfig, main_config, smoke_config  # noqa: E402
from merton_meta.datagen import build_all  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["smoke", "main"], default="main")
    ap.add_argument("--config", default=None, help="JSON config overriding preset")
    ap.add_argument("--exp", default="all", help="comma list from {q1,q2,q4} or 'all'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reveal-params", action="store_true",
                    help="Q1 contamination arm: print task parameters in prompts")
    ap.add_argument("--q1-sigma-label", type=float, default=None,
                    help="override Q1 behavior noise (0.0 = noiseless control arm)")
    args = ap.parse_args()

    cfg: ExperimentConfig = (ExperimentConfig.from_json(args.config) if args.config
                             else (smoke_config() if args.preset == "smoke"
                                   else main_config()))
    if args.reveal_params:
        cfg.fixed.reveal_params = True
    if args.q1_sigma_label is not None:
        cfg.fixed.sigma_label_q1 = args.q1_sigma_label

    exps = ["q1", "q2", "q4"] if args.exp == "all" else args.exp.split(",")
    report = build_all(cfg, args.out, exps)
    print(json.dumps(report, indent=2))
    print(f"[make_data] wrote datasets under {args.out}/")


if __name__ == "__main__":
    main()
