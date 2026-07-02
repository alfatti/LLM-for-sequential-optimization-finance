#!/usr/bin/env bash
# End-to-end run order for Q1 / Q2 / Q4. Run from the repo root on the GPU box.
# CPU stages (0-1) should already have passed locally before any GPU spend.
set -euo pipefail

DATA=data_main
RUNS=runs
RES=results

# ---- 0. Data (CPU, deterministic) -------------------------------------------
python scripts/make_data.py --preset main --exp all --out $DATA

# ---- 1. Design validation (CPU; must PASS before training) ------------------
python scripts/proxy_validate.py --data $DATA

# ---- 2. GPU smoke tests (minutes) -------------------------------------------
python scripts/make_data.py --preset smoke --exp all --out data_smoke
python scripts/proxy_validate.py --data data_smoke
python scripts/train_qlora.py --data data_smoke/q4/ntask_00008 \
    --out $RUNS/smoke_q4 --max-len 4096 --smoke
python scripts/eval_llm.py --data data_smoke --bundle q4/eval_bundle.jsonl \
    --adapter $RUNS/smoke_q4/adapter_final --arm smoke --out $RES/smoke_q4 --limit 4

# ---- 3. Q4 training sweep (paper replication axis) ---------------------------
for N in 00100 00400 01600 03200; do
  python scripts/train_qlora.py --data $DATA/q4/ntask_$N \
      --out $RUNS/q4_n$N --max-len 4096 --mask-mode decision
done

# ---- 4. Q1 training (fixed MDP behavioral cloning) --------------------------
python scripts/train_qlora.py --data $DATA/q1 --out $RUNS/q1 \
    --max-len 640 --mask-mode decision

# ---- 5. Evaluation arms ------------------------------------------------------
# Q4: SFT checkpoints (K_test sweep + tiers are inside the bundle)
for N in 00100 00400 01600 03200; do
  python scripts/eval_llm.py --data $DATA --bundle q4/eval_bundle.jsonl \
      --adapter $RUNS/q4_n$N/adapter_final --arm sft_n$N --out $RES/q4_sft_n$N
done
# Q4: ICL-only baseline (base model, same bundle)
python scripts/eval_llm.py --data $DATA --bundle q4/eval_bundle.jsonl \
    --arm icl_base_q4 --out $RES/q4_icl_base
# Q2: fixed-MDP ICL (base model)
python scripts/eval_llm.py --data $DATA --bundle q2/eval_bundle.jsonl \
    --arm q2_icl_base --out $RES/q2_icl_base
# Q1: SFT arm + probe, and base-model zero-shot reference on the same bundle
python scripts/eval_llm.py --data $DATA --bundle q1/eval_bundle.jsonl \
    --adapter $RUNS/q1/adapter_final --arm q1_sft --out $RES/q1_sft \
    --probe-bundle q1/probe_bundle.jsonl
python scripts/eval_llm.py --data $DATA --bundle q1/eval_bundle.jsonl \
    --arm q1_base_zeroshot --out $RES/q1_base

# ---- 6. Figures ---------------------------------------------------------------
python scripts/make_plots.py --data $DATA --out figs \
    --results $RES/q4_sft_n03200 $RES/q4_icl_base $RES/q2_icl_base \
              $RES/q1_sft $RES/q1_base \
    --ntask 100=$RES/q4_sft_n00100 400=$RES/q4_sft_n00400 \
            1600=$RES/q4_sft_n01600 3200=$RES/q4_sft_n03200
