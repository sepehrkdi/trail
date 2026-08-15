"""Shared accuracy primitives for the metric layer.

Pure numpy helpers over :class:`~trail.core.types.SplitOutputs`. All
accuracies in trail are reported on the 0-100 scale.
"""
from __future__ import annotations

import logging

import numpy as np

from trail.core.errors import MetricError
from trail.core.types import SplitOutputs

logger = logging.getLogger("trail.metrics._accuracy_core")


def correctness(out: SplitOutputs) -> np.ndarray:
    """Per-example correctness mask: ``argmax(logits) == targets``.

    Args:
        out: split outputs; ``logits`` must be present (classification).

    Returns:
        Boolean array of shape ``[N]``.

    Raises:
        MetricError: if ``out.logits`` is ``None`` (non-classification
            payload, or an outputs-in payload that omitted logits).
    """
    if out.logits is None:
        raise MetricError(
            "correctness requires logits; SplitOutputs.logits is None "
            "(non-classification payload, or logits omitted from outputs-in)")
    preds = np.argmax(out.logits, axis=1)
    return preds == out.targets


def acc100(mask: np.ndarray) -> float:
    """Accuracy on the 0-100 scale from a per-example correctness mask."""
    arr = np.asarray(mask, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(100.0 * arr.mean())


def accuracy_pct(logits: np.ndarray, targets: np.ndarray) -> float:
    """Top-1 multiclass accuracy on the 0-100 scale — via TorchMetrics.

    This is the metric layer's accuracy PRIMITIVE and the ONLY place trail
    uses TorchMetrics: the headline accuracy point estimate flows through
    ``torchmetrics.functional.classification.multiclass_accuracy`` (micro
    averaging) for a standard, reputable-library computation. It is
    numerically identical to ``acc100(correctness(out))`` (a count/total
    ratio) — the per-example correctness mask stays numpy and remains the
    basis for the bootstrap CIs (G5); TorchMetrics is not used for the CI, the
    MIA/relearning logic, or anywhere in the framework layer.

    Args:
        logits: ``[N, C]`` scores; ``argmax`` over axis 1 is the prediction.
        targets: ``[N]`` integer labels.

    Returns:
        Accuracy in ``[0, 100]``; ``nan`` for an empty split.
    """
    logits = np.asarray(logits)
    targets = np.asarray(targets)
    if targets.size == 0:
        return float("nan")
    import torch
    from torchmetrics.functional.classification import multiclass_accuracy
    num_classes = int(logits.shape[1])
    acc = multiclass_accuracy(
        torch.as_tensor(logits),
        torch.as_tensor(targets, dtype=torch.long),
        num_classes=num_classes, average="micro")
    return float(acc.item()) * 100.0
