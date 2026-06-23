"""SFT data handling for the BG market-making imitation task.

Two responsibilities:
  1. Window long episodes into bounded training sequences (episodes can be ~20k tokens;
     we train on fixed windows that always start at an <OBS> boundary so context is
     coherent). This keeps `horizon_days` a free data knob while training stays bounded.
  2. Mask the loss to ACTION tokens only -- the model is scored solely on reproducing the
     oracle's action given history, exactly as the SFT paper does (loss on a_t | history,
     not on the observation/reward tokens which are environment-supplied context).

The schema is line-structured:
    <OBS> ...        (context)
    <ACT> <digit>    (the supervised target -- the digit token)
    <REW> ...        (context: outcome feedback)
We locate the single digit token following each '<ACT>' and unmask only that position.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import torch


# Special tokens added to the tokenizer so the structure is single-token and stable.
SPECIAL_TOKENS = ["<EPISODE>", "<OBS>", "<ACT>", "<REW>", "<END>"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def build_windows(text, tokenizer, max_len, stride_frac=0.5):
    """Tokenize an episode and split into windows that begin at an <OBS> boundary.

    Returns a list of (input_ids, action_label_mask) where action_label_mask[i]=True iff
    position i is the digit token immediately following an '<ACT>' token (the target).
    Windows overlap by (1-stride_frac) so actions near a window edge still get full
    left-context in some window.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    act_id = tokenizer.convert_tokens_to_ids("<ACT>")
    obs_id = tokenizer.convert_tokens_to_ids("<OBS>")

    # mark action-target positions: the token right after each <ACT>
    is_target = [False] * len(ids)
    for i, tok in enumerate(ids):
        if tok == act_id and i + 1 < len(ids):
            is_target[i + 1] = True
    # candidate window starts = <OBS> positions (coherent context boundaries)
    obs_positions = [i for i, tok in enumerate(ids) if tok == obs_id]
    if not obs_positions:
        obs_positions = [0]

    windows = []
    stride = max(int(max_len * stride_frac), 1)
    start_idx = 0
    used_starts = set()
    # walk obs boundaries, packing as many as fit per window
    p = 0
    while p < len(obs_positions):
        s = obs_positions[p]
        if s in used_starts:
            p += 1
            continue
        e = min(s + max_len, len(ids))
        win_ids = ids[s:e]
        win_mask = is_target[s:e]
        if any(win_mask):                      # skip windows with no action target
            windows.append((win_ids, win_mask))
        used_starts.add(s)
        # advance start by ~stride tokens, snapped to the next obs boundary
        target_pos = s + stride
        while p < len(obs_positions) and obs_positions[p] < target_pos:
            p += 1
    return windows


@dataclass
class WindowedSFTDataset(torch.utils.data.Dataset):
    """Flattens all episodes into action-masked training windows."""
    records: list
    tokenizer: object
    max_len: int = 2048
    stride_frac: float = 0.5

    def __post_init__(self):
        self.examples = []
        for r in self.records:
            for win_ids, win_mask in build_windows(
                r["text"], self.tokenizer, self.max_len, self.stride_frac):
                self.examples.append((win_ids, win_mask))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ids, mask = self.examples[i]
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        m = torch.tensor(mask, dtype=torch.bool)
        labels[~m] = -100          # CrossEntropy ignore_index -> loss only on actions
        return {"input_ids": input_ids, "labels": labels,
                "attention_mask": torch.ones_like(input_ids)}


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for b in batch:
            n = maxlen - len(b["input_ids"])
            out["input_ids"].append(
                torch.cat([b["input_ids"], torch.full((n,), self.pad_token_id)]))
            out["labels"].append(
                torch.cat([b["labels"], torch.full((n,), -100)]))
            out["attention_mask"].append(
                torch.cat([b["attention_mask"], torch.zeros(n, dtype=torch.long)]))
        return {k: torch.stack(v) for k, v in out.items()}
