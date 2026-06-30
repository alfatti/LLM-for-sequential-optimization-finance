# American Put — SFT exercise-policy experiment

Single fixed MDP (one option), testing Paper 1's CORE claim only:
SFT on oracle-labeled rollouts beats ICL-only and random.
NOT an Eq. 6 replication (no task distribution), NOT an OOD test.

Instance: S0=K=40, T=1, sigma=0.4, r=0.06, risk-neutral. v* = 5.30278 (50-step CRR).

## Files
- oracle.py        CRR oracle + boundary extraction. Validated: price converges
                   ~5.318, LS cross-check 5.291+/-0.013. `python oracle.py`
- rollouts.py      On-lattice Q-paths, exact oracle labels, serialized (version A:
                   `t=k S=S -> STOP|GO`). Self-validates by reproducing v*.
                   `python rollouts.py` regenerates the .jsonl dumps.
- proxy_sft.py     CPU design-validation harness. Logistic stop-go classifier in
                   the policy slot; full ICL/SFT/random eval through the three
                   metrics (value gap / boundary error / stopping-time error).
                   `python proxy_sft.py`
- train_qlora.py   REAL training script for GPU (H200). Llama-2-7B, QLoRA/NF4,
                   Paper 1 App. C config. Core run = full-sequence loss;
                   ablation = masked-to-decision. NOT runnable on CPU.

## Data
- rollouts_optimal.jsonl   raw labeled trajectories (one JSON/line)
- sft_text.jsonl           serialized training strings (one {"text":...}/line)
- meta.json                instance, lattice params, v*, boundary S*(t_k)

## Key validated findings (from proxy_sft.py)
- The three metrics DECOUPLE hard: value gap is nearly blind (never-exercising
  costs only ~5.2%; multi-dollar boundary errors cost ~0.5%), while boundary
  error and stopping-time error discriminate. This is smooth-pasting flatness,
  the structural result the single-option testbed delivers.
- Boundary is learnable on its UPPER EDGE (where Q-paths graze it), unconstrained
  in the deep interior; smooth-pasting makes those misses costless in value.
- ~41% of optimal-policy trajectories never exercise -> heavy GO/STOP imbalance;
  classification accuracy is a POOR proxy for policy quality. Read the LLM run
  via boundary + value gap, not label accuracy.

## GPU run (H200)
    pip install "transformers>=4.40" "peft>=0.10" trl accelerate bitsandbytes datasets
    # smoke test first (both loss paths):
    head -50 sft_text.jsonl > sft_smoke.jsonl
    python train_qlora.py --data sft_smoke.jsonl --out smoke_full   --loss full   --epochs 1 --batch_size 4 --grad_accum 1 --max_len 512
    python train_qlora.py --data sft_smoke.jsonl --out smoke_masked --loss masked --epochs 1 --batch_size 4 --grad_accum 1 --max_len 512
    # check: trainable params ~30-80M (not 0, not 7B); masked% sane + decodes to STOP/GO;
    #        loss finite & decreasing; [len] truncated = 0; adapter writes to out/.
    # then the real core run:
    python train_qlora.py --loss full --out adapter_put_sft_full
    python train_qlora.py --loss masked --out adapter_put_sft_masked   # ablation

## Still ahead
- LLM eval harness: proxy_sft.py structure with Llama in the policy slot, loading
  the trained adapter, ICL-only / SFT / random through the three metrics on the
  same fresh lattice paths.
