# configs/

Declarative JSON equivalents of the presets in `merton_meta/config.py`,
consumed via `scripts/make_data.py --config <file>`.

    main.json               full experiment (all knobs at locked values)
    smoke.json              tiny preset for CPU validation / GPU smoke tests
    q1_noiseless.json       Q1 ablation arm: sigma_label_q1 = 0.0
                            (trivial-memorization control)
    q1_reveal_params.json   Q1 ablation arm: task parameters printed in
                            prompts (contamination-measurement arm)

Precedence: `--config` overrides `--preset`; explicit CLI flags
(`--reveal-params`, `--q1-sigma-label`) override both. The config actually
used is always frozen into `<out>/config.json` by make_data, and every
downstream script (proxy_validate, eval_llm, make_plots) reads THAT copy —
so a data directory is self-describing and the JSONs here are provenance,
not a runtime dependency.

To create a new ablation arm, copy main.json, edit, and pass it to
`make_data.py --config` with a fresh `--out` directory.
