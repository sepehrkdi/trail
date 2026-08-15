"""Pure-numpy signal transforms shared by the MIA stack.

Ported from the original evaluation pipeline (the Unlearn-Sparse-faithful MIA
helpers); trail works on numpy logits from SplitOutputs rather than torch
models, so the softmax is recomputed here instead of collected in a forward
pass. No torch, no RNG, no I/O — these are deterministic array functions.
"""
# Portions of this file (the per-class threshold search and modified-entropy signal) are adapted from
# OPTML-Group/Unlearn-Sparse, Copyright (c) 2023 OPTML Group, MIT License.
# See the third-party notice in LICENSE.

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("trail.metrics._signals")


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax over ``[N, C]`` logits.

    Numpy equivalent of the ``F.softmax(model(inputs), dim=-1)`` collection in
    the original evaluation pipeline (``_collect_softmax_probs``), for use
    on cached/outputs-in logits.
    """
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def entropy(probs: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    """Per-example Shannon entropy of softmax rows.

    Port of the original evaluation pipeline (``_entropy``).
    """
    probs = np.asarray(probs)
    return np.sum(probs * (-np.log(np.maximum(probs, eps))), axis=1)


def modified_entropy(probs: np.ndarray, labels: np.ndarray,
                     eps: float = 1e-30) -> np.ndarray:
    """Modified entropy (Song & Mittal): swap ``p_true <-> (1 - p_true)`` to
    sharpen the membership signal.

    Port of the original evaluation pipeline (``_modified_entropy``),
    itself matching Unlearn-Sparse ``m_entropy``.
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels, dtype=np.int64)
    log_probs = -np.log(np.maximum(probs, eps))
    reverse_probs = 1.0 - probs
    log_reverse_probs = -np.log(np.maximum(reverse_probs, eps))
    idx = np.arange(len(labels))
    modified_probs = np.copy(probs)
    modified_probs[idx, labels] = reverse_probs[idx, labels]
    modified_log_probs = np.copy(log_reverse_probs)
    modified_log_probs[idx, labels] = log_probs[idx, labels]
    return np.sum(modified_probs * modified_log_probs, axis=1)


def per_class_threshold(tr_vals: np.ndarray, te_vals: np.ndarray) -> float:
    """Threshold maximizing balanced accuracy ``0.5 * (TPR + TNR)`` over the
    grid of all candidate values from both shadow arrays.

    Port of the original evaluation pipeline (``_per_class_threshold``,
    Unlearn-Sparse ``_thre_setting``). ``tr_vals`` are shadow member values,
    ``te_vals`` shadow non-member values; members are predicted by
    ``value >= threshold``.

    Guard added at port time: either array empty returns 0.0 (the source
    would divide by zero); callers should not feed empty shadow classes.
    """
    tr_vals = np.asarray(tr_vals)
    te_vals = np.asarray(te_vals)
    if len(tr_vals) == 0 or len(te_vals) == 0:
        logger.warning("per_class_threshold called with an empty shadow array")
        return 0.0
    candidates = np.concatenate([tr_vals, te_vals])
    best_thre, best_acc = 0.0, 0.0
    for value in candidates:
        tr_ratio = np.sum(tr_vals >= value) / len(tr_vals)
        te_ratio = np.sum(te_vals < value) / len(te_vals)
        acc = 0.5 * (tr_ratio + te_ratio)
        if acc > best_acc:
            best_thre, best_acc = float(value), acc
    return best_thre
