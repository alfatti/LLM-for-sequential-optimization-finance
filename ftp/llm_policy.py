"""LLM policies for evaluation: drop into the rollout harness with the standard
policy(step, q, t) -> action_index signature.

  LLMPolicy      : the fine-tuned (SFT) model. Builds the running episode prefix exactly
                   as in training, appends the current <OBS> line + '<ACT> ', and reads
                   the argmax over the 9 action-digit tokens (restricted decoding, as the
                   SFT paper does).
  ICLPolicy      : same base model WITHOUT fine-tuning, but the prompt is prefixed with a
                   few-shot support trajectory (k oracle episodes = prior regime
                   realizations on the same bond). This is the paper's ICL baseline.

Both maintain the running serialized history across the episode so the model conditions on
the same format it trained on. The rollout calls the policy once per RFQ; the policy
appends the realized <REW> after the environment returns (via .observe_outcome).
"""
from __future__ import annotations

import numpy as np
import torch

from .serialize import (ACTION_BINS, N_ACTIONS, encode_action, encode_observation,
                        encode_reward)


class _BaseLLMPolicy:
    def __init__(self, model, tokenizer, z=1.0, header=None, prefix=""):
        self.model = model
        self.tok = tokenizer
        self.z = z
        self.prefix = prefix                 # few-shot support context (ICL) or ""
        self.header = header or "<EPISODE>"
        self.lines = [self.header]
        # precompute the token ids for the 9 action digits '0'..'8'
        self.act_token_ids = [self.tok.convert_tokens_to_ids(str(a))
                              for a in range(N_ACTIONS)]
        self._pending = None                 # (q_next placeholder) bookkeeping
        self.device = next(model.parameters()).device

    def reset(self):
        self.lines = [self.header]
        self._pending = None

    def __call__(self, step, q, t):
        obs = encode_observation(step, q, self.z)
        self.lines.append(obs)
        prompt = self.prefix + "\n".join(self.lines) + "\n<ACT> "
        ids = self.tok(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**ids).logits[0, -1]      # next-token logits
        act_logits = torch.tensor([logits[i] for i in self.act_token_ids])
        a = int(torch.argmax(act_logits).item())
        self.lines.append(encode_action(a))
        self._pending = True
        return a

    def observe_outcome(self, filled, pnl, q_next):
        """Append the realized reward line so history matches the training schema."""
        self.lines.append(encode_reward(filled, pnl, q_next, self.z))
        self._pending = None


class LLMPolicy(_BaseLLMPolicy):
    """Fine-tuned SFT model policy (no few-shot prefix needed)."""
    pass


class ICLPolicy(_BaseLLMPolicy):
    """Base (un-fine-tuned) model with a few-shot support-trajectory prefix."""
    def __init__(self, model, tokenizer, support_text, **kw):
        # support_text = concatenation of k full oracle episodes (the demonstrations)
        super().__init__(model, tokenizer, prefix=support_text + "\n", **kw)


def build_support_prefix(support_records, k=2):
    """Concatenate k oracle episodes into a few-shot demonstration prefix (ICL)."""
    chosen = support_records[:k]
    return "\n".join(r["text"] for r in chosen) + "\n"


def load_sft_model(adapter_path, base_path=None):
    """Load the base model + LoRA adapter for evaluation."""
    import os
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    base_path = base_path or os.environ.get("LLAMA3_8B_PATH", "./models/llama-3-8b")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok
