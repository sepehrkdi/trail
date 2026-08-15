"""Shared value types — the leaf module every other trail module imports.

Keep this file dependency-light: numpy + stdlib only (torch under TYPE_CHECKING).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

Role = Literal["unlearned", "original", "gold"]
Split = Literal["forget", "retain", "test", "forget_test", "retain_test"]

ROLES: tuple[str, ...] = ("unlearned", "original", "gold")
SPLITS: tuple[str, ...] = ("forget", "retain", "test", "forget_test", "retain_test")

# bump when forward_stats semantics change (participates in L1 keys).
# "1" -> "2": multi-layer feature export — the probe now captures one or
# more named penultimate/intermediate features (SplitOutputs.features_multi) via
# ARCH_FEATURE_RESOLVERS, and probe_layers joins the L1 cache key. The accuracy/
# CE-loss path is unchanged, so the G10 accuracy panel is unaffected.
PROBE_VERSION = "2"


@dataclass(frozen=True)
class SplitOutputs:
    """Per-example model outputs on one data split — the L1 cache value.

    ``losses`` semantics are modality-defined: per-example cross-entropy for
    classification; per-token NLL with answer-span masking (LLM, by specification).
    Non-finite losses are NOT rejected here; they surface as ``runtime_error``
    skips at the consuming metric.

    ``features`` is the single penultimate feature (the activation_distance
    adjunct, kept for back-compat). ``features_multi`` is the named
    per-layer feature map ``{layer_name: [N, D]}`` produced by the multi-hook
    probe; ``features`` mirrors ``features_multi['penultimate']`` when present.
    """

    losses: np.ndarray              # [N] float32
    targets: np.ndarray             # [N] int64
    logits: np.ndarray | None       # [N, C] classification only
    features: np.ndarray | None     # [N, D] penultimate features; model-in only
    n: int
    features_multi: dict[str, np.ndarray] | None = None  # F4: per-layer [N, D]

    def __post_init__(self) -> None:
        if self.losses.ndim != 1 or self.targets.ndim != 1:
            raise ValueError("losses/targets must be 1-D per-example arrays")
        if len(self.losses) != self.n or len(self.targets) != self.n:
            raise ValueError(f"n={self.n} inconsistent with array lengths "
                             f"({len(self.losses)}, {len(self.targets)})")
        if self.logits is not None and len(self.logits) != self.n:
            raise ValueError("logits length inconsistent with n")
        if self.features is not None and len(self.features) != self.n:
            raise ValueError("features length inconsistent with n")
        # F4 drop-all-or-nothing: every captured layer is full-length or the
        # probe must omit it entirely; a ragged layer here is a hard error.
        if self.features_multi is not None:
            for name, arr in self.features_multi.items():
                if arr is None or len(arr) != self.n:
                    raise ValueError(
                        f"features_multi[{name!r}] length "
                        f"{None if arr is None else len(arr)} inconsistent with "
                        f"n={self.n}")


@dataclass(frozen=True)
class Reference:
    """Handle for a resolved reference model (gold; shadow handles are ShadowStats)."""

    kind: Literal["gold", "shadow"]
    path: str
    sha256: str
    source: Literal["supplied", "built"]
    cache_key: str


@dataclass(frozen=True)
class ShadowStats:
    """L3 cache value for likelihood-ratio MIA (M9 opt-in tier).

    The audit pool is the concatenation ``[forget examples ; test examples]``
    in canonical (aug-stripped) loader order — the SAME order the target model
    is scored in, so per-example alignment with ``ctx.outputs`` is positional.
    For audit example ``i`` and shadow ``s``:

    * ``shadow_phi[s, i]`` is the LiRA confidence signal phi (logit-transformed
      true-class softmax probability; attacks/lira.py:confidence_logit) of
      example ``i`` under shadow model ``s``.
    * ``member_mask[s, i]`` is True iff example ``i`` was IN shadow ``s``'s
      training set. Per shadow, ~``in_fraction`` of the audit pool is IN; the
      retain pool is always-in filler (not part of the audit pool / mask).

    ``audit_labels[i]`` is 1 for forget (member-candidate of the original
    training set) and 0 for test (true non-member) — the ground truth the LiRA
    AUC distinguishes. Method-INDEPENDENT (no checkpoint hash in the key):
    the ensemble is amortized across every method evaluated on the same data.
    """

    shadow_phi: np.ndarray           # [n_shadow, N] float64  (LiRA confidence signal)
    member_mask: np.ndarray          # [n_shadow, N] bool     (IN/OUT per shadow)
    audit_labels: np.ndarray         # [N] int                (1=forget, 0=test)
    n_shadow: int
    cache_key: str


@dataclass(frozen=True)
class EnsembleMargins:
    """L3 cache value for the KLoM metric (M68).

    Per-split KLoM logit-margins of a checkpoint ENSEMBLE — the oracle golds
    (``kind="gold"``: retrained-from-scratch models; method-INDEPENDENT, so the
    L3 key is content-addressed by the sorted gold checkpoint SHAs and the
    ensemble is built once and amortized across every method on the same data,
    exactly like the LiRA shadow ensemble) or the method's own unlearned seeds
    (``kind="unlearned"``: method-specific).

    ``margins[split]`` is ``[n_models, N_split]`` float64 in canonical
    (augmentation-stripped) loader order, so every ensemble member — and the
    gold vs. unlearned ensembles — align positionally per example. The margin is
    ``phi(x;theta) = f_y(x) - log sum_{k!=y} exp(f_k(x))`` clipped to the KLoM
    range (references/gold_ensemble.klom_margin).
    """

    kind: str                              # "gold" | "unlearned"
    margins: dict[str, np.ndarray]         # split -> [n_models, N] float64
    n_models: int
    cache_key: str


LeafProbe = Callable[["nn.Module", "DataLoader", int], SplitOutputs]
