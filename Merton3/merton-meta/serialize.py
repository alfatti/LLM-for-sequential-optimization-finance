"""Serialization schema (D2, D3) + decision-span extraction for loss masking.

Schema (plain, task-name-free -- the word "Merton" and all task parameters are
deliberately absent from prompts as a contamination guard):

    ### DEMO 1
    t=01 x=+0.000 -> a=+0.37
    t=02 x=+0.031 -> a=+0.55
    ...
    ### DEMO k
    ...
    ### QUERY
    t=01 x=+0.000 -> a=+0.55
    ...

* x = log(W_t / W_0), formatted %+.3f  (normalized log-wealth stabilizes the
  state distribution across tasks; see D1 / fixed-Lambda discussion).
* a = risky fraction, formatted %+.2f  (2 decimals == 0.01 action grid).
* Support-demo actions are the EXECUTED noisy actions a_t = pi* + eps_t
  (unclipped -- demos are text and need not lie on the scoring grid).
* Training query actions are the CLEAN pi* rounded to the grid (DPT-faithful).

Loss masking (pre-registered ablation): character spans of the action strings
after each "-> a=" delimiter inside the QUERY section are located here and
converted to token masks in the trainer via tokenizer offset mappings. This
character-offset approach is robust to subword tokenization.

Oracles read the SAME text the model reads: `parse_demo_actions` recovers demo
actions from the rendered prompt (so 2-decimal rounding is inside the oracle's
information set, by construction).
"""
from __future__ import annotations

import re
from typing import List, Sequence, Tuple

import numpy as np

QUERY_HEADER = "### QUERY"
DEMO_HEADER_FMT = "### DEMO {i}"
STEP_FMT = "t={t:02d} x={x} -> a={a}"
ACTION_RE = re.compile(r"-> a=([+-]\d+\.\d{2})")
DECISION_PREFIX = "-> a="


def fmt_action(a: float) -> str:
    return f"{a:+.2f}"


def fmt_state(x: float) -> str:
    return f"{x:+.3f}"


def render_traj_lines(xs: Sequence[float], actions: Sequence[float]) -> List[str]:
    """xs has length T+1 (x_0..x_T); actions length T. One line per decision."""
    assert len(xs) == len(actions) + 1
    return [STEP_FMT.format(t=t + 1, x=fmt_state(xs[t]), a=fmt_action(actions[t]))
            for t in range(len(actions))]


def render_demos_block(demos: Sequence[Tuple[Sequence[float], Sequence[float]]]) -> str:
    """demos: list of (xs, actions)."""
    parts: List[str] = []
    for i, (xs, acts) in enumerate(demos, start=1):
        parts.append(DEMO_HEADER_FMT.format(i=i))
        parts.extend(render_traj_lines(xs, acts))
    return "\n".join(parts)


def params_header(mu: float, sigma: float, r: float, gamma: float) -> str:
    """Q1 contamination arm ONLY (reveal_params=True): task parameters in plain
    text so the model could in principle retrieve the textbook formula."""
    return f"# params mu={mu:.4f} sigma={sigma:.2f} r={r:.2f} gamma={gamma:.1f}"


def render_training_text(demos, query_xs, query_actions, header: str = "") -> str:
    """Full training example: [optional header] + [optional demos] + query."""
    parts: List[str] = []
    if header:
        parts.append(header)
    if demos:
        parts.append(render_demos_block(demos))
    parts.append(QUERY_HEADER)
    parts.extend(render_traj_lines(query_xs, query_actions))
    return "\n".join(parts) + "\n"


def render_decision_prompt(demos_text: str, query_xs: Sequence[float],
                           query_actions: Sequence[float], next_x: float,
                           next_t: int, header: str = "") -> str:
    """Prompt for the decision at step `next_t` (1-indexed): demos + query
    history so far + the open line ending exactly at 'a=' (candidate follows)."""
    lines = render_traj_lines(list(query_xs) + [next_x], list(query_actions))
    open_line = f"t={next_t:02d} x={fmt_state(next_x)} {DECISION_PREFIX}"
    parts: List[str] = []
    if header:
        parts.append(header)
    if demos_text:
        parts.append(demos_text)
    parts.append(QUERY_HEADER)
    parts.extend(lines)
    parts.append(open_line)
    return "\n".join(parts)


def parse_demo_actions(text: str) -> np.ndarray:
    """All demo actions (2dp floats), in order, from the demos block of a prompt.
    Only matches BEFORE the QUERY header."""
    qpos = text.find(QUERY_HEADER)
    region = text if qpos < 0 else text[:qpos]
    return np.array([float(m.group(1)) for m in ACTION_RE.finditer(region)])


def parse_all_actions(text: str) -> np.ndarray:
    return np.array([float(m.group(1)) for m in ACTION_RE.finditer(text)])


def decision_char_spans(text: str, mask_mode: str = "decision") -> List[Tuple[int, int]]:
    """Character spans that should receive loss.

    mask_mode:
      'decision'  -> only the action strings after '-> a=' inside QUERY (default)
      'query'     -> the whole QUERY section (states + actions)
      'full'      -> the whole sequence
    """
    if mask_mode == "full":
        return [(0, len(text))]
    qpos = text.find(QUERY_HEADER)
    if qpos < 0:
        raise ValueError("QUERY header not found in training text")
    if mask_mode == "query":
        start = qpos + len(QUERY_HEADER)
        return [(start, len(text))]
    if mask_mode == "decision":
        spans = []
        for m in ACTION_RE.finditer(text, pos=qpos):
            spans.append((m.start(1), m.end(1)))
        if not spans:
            raise ValueError("no decision spans found in QUERY section")
        return spans
    raise ValueError(f"unknown mask_mode: {mask_mode}")
