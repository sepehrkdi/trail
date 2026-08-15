"""LLM adapter — design-only stub with the NORMATIVE outputs payload spec.

The LLM adapter ships as interface plus payload
specification; every method raises NotImplementedError. The documented
activation path is BACKEND ADOPTION: an adapter translates TRAIL
requests into OpenUnlearning evaluator configurations, runs its evaluators
(TOFU forget quality M28, TOFU utility M29, ROUGE-L M30, plus M16), and
re-emits scores as MetricResults with the backend version in provenance.
A placeholder with a precise contract is more useful — and more honest —
than a hasty implementation that freezes wrong loss semantics into the
protocol.

NORMATIVE OUTPUTS PAYLOAD SPECIFICATION (task="llm", outputs-in)
================================================================

An LLM outputs payload submitted per split MUST contain:

- ``losses``: float array [N] — per-example loss reduced from **per-token
  NLL with answer-span masking** under the four pinned loss-semantics
  choices below.
- ``targets``: int array [N] — example ids within the versioned evaluation
  set (no class semantics), aligned with ``losses``.

``logits`` MUST be absent or None (vocabulary-sized logits dumps are
rejected; per-token information enters only through the pinned reduction).
``features`` are optional pooled final-hidden-state vectors [N, D].

THE FOUR PINNED LOSS-SEMANTICS CHOICES
-----------------------------------------------------
Two differing choices on any axis yield incomparable numbers that look
identical in a report; submissions are therefore commensurable or rejected
at validation:

1. **Tokenizer identity** — the exact tokenizer is hashed into the manifest
   (``tokenizer_sha``); payloads computed under a different tokenizer hash
   are rejected, not silently compared.
2. **Normalization** — per-example loss is the MEAN per-token NLL over
   unmasked (answer-span) tokens, NOT the per-sequence sum; sequence-length
   confounds are removed by definition, not by footnote.
3. **Padding/truncation policy** — right padding, truncation to the
   declared ``max_length``; padding tokens never contribute to the loss.
   The policy string is part of ``mask_policy``.
4. **Answer-span masking** — NLL is computed ONLY over answer-span tokens
   (prompt/question tokens are masked out); the masking rule is declared in
   ``mask_policy``.

REQUIRED MANIFEST FIELDS (verified at validation and stamped into
``provenance.preprocessing``):

- ``tokenizer_sha`` — hash of the tokenizer identity (vocab + merges +
  special-token map); token-level loss semantics differ silently across
  tokenizers.
- ``mask_policy``   — declaration of choices 2–4 above (normalization,
  padding/truncation, answer-span rule), e.g.
  ``"answer_span;per_token_mean;right_pad;trunc=512"``.

Gold retrains are not user-feasible artifacts at LLM scale and for
web-scale pretraining are not even well-defined; gold-tier metrics
auto-skip with ``missing_role:gold``. Where a benchmark ships
retained references (TOFU), the adapter consumes them as supplied gold.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from trail.adapters.base import ModalityAdapter
from trail.core.types import SplitOutputs

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

logger = logging.getLogger("trail.adapters.llm")

_NOT_IMPLEMENTED = "LLM via backend adoption"


class LLMAdapter(ModalityAdapter):
    """Design-only stub (E1): contract pinned, activation via
    OpenUnlearning backend adoption."""

    name = "llm"

    def load_checkpoint(self, path: str, device: "torch.device") -> "nn.Module":
        """Backend adoption: checkpoint handling delegates to the wrapped
        evaluator's loading path."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def forward_stats(self, model: "nn.Module", loader: "DataLoader", *,
                      seed: int, device: "torch.device") -> SplitOutputs:
        """Backend adoption: per-token NLL with answer-span masking under the
        four pinned loss-semantics choices (module docstring)."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def canonical_eval_view(self, loader: "DataLoader", *, seed: int,
                            num_workers: int = 0) -> tuple["DataLoader", dict]:
        """Backend adoption: tokenization/truncation/masking is the hidden
        preprocessing here; manifest emits tokenizer_sha/mask_policy."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def validate_outputs_payload(self, payload: dict) -> SplitOutputs:
        """Validate against the normative payload spec in the module
        docstring (losses+targets required, logits forbidden,
        tokenizer_sha/mask_policy verified)."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def default_metrics(self) -> list[str]:
        """Cataloged backend rows: M28 TOFU forget quality, M29 TOFU utility,
        M30 ROUGE-L on forget completions, M16 probing."""
        raise NotImplementedError(_NOT_IMPLEMENTED)
