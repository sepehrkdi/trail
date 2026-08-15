"""Representation-space metrics (Tier-1, M33-M36) — off the default panel
(``external=True``).

Computed over the cached penultimate features (``SplitOutputs.features``, F4) on
``retain_test``, mirroring the ``activation_distance`` template (structural.py):
the same ``_AD_MAX_SAMPLES`` deterministic first-N cap, model-in only, gold/
original references where applicable.

The math helpers are pure, deterministic, and bit-exact (NO SGD, NO sklearn at
runtime): linear CKA via the O(N·D) feature-space gram trick (== the N×N
doubly-centered gram form), nearest-class-mean accuracy, and a closed-form ridge
linear probe (== ``sklearn.Ridge(fit_intercept=False)``). CKA CIs use the
frozen-estimator group/row bootstrap (the statistic is RECOMPUTED on each
resample, not averaged per-example; G5).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from trail.core.bootstrap import bootstrap_ci, bootstrap_ci_groups
from trail.core.errors import MetricError, MetricSkip
from trail.core.registry import register_metric
from trail.core.report import MetricResult
from trail.core.seeding import numpy_rng

if TYPE_CHECKING:  # pragma: no cover
    from trail.core.context import EvalContext

logger = logging.getLogger("trail.metrics.representation")

_CLS: set[str] = {"classification"}

#: deterministic first-N feature cap, shared with activation_distance.
_AD_MAX_SAMPLES = 4096

#: pinned ridge regularization for the linear probe (frozen protocol knob).
_RIDGE_LAMBDA = 1.0


# ---------------------------------------------------------------------------
# Pure math (deterministic, bit-exact)
# ---------------------------------------------------------------------------

def _center(M: np.ndarray) -> np.ndarray:
    """Column-center (subtract the per-feature mean)."""
    return M - M.mean(axis=0, keepdims=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al. 2019) via the feature-space gram trick.

    ``CKA = ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F · ||Yc^T Yc||_F)`` over
    column-centered ``Xc``/``Yc`` — identical to the N×N doubly-centered gram
    form but O(N·D) instead of O(N^2). Invariant to orthogonal transforms
    (permutation/rotation) and isotropic scaling; 1.0 for identical inputs.
    """
    Xc = _center(np.asarray(X, dtype=np.float64))
    Yc = _center(np.asarray(Y, dtype=np.float64))
    xty = Xc.T @ Yc
    hsic_xy = float(np.sum(xty * xty))                  # ||Xc^T Yc||_F^2
    xtx = Xc.T @ Xc
    yty = Yc.T @ Yc
    denom = float(np.sqrt(np.sum(xtx * xtx)) * np.sqrt(np.sum(yty * yty)))
    if denom <= 0.0:
        return float("nan")
    return hsic_xy / denom


def _ncc_predict(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Nearest-class-mean predictions (frozen class means)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)
    means = np.stack([X[y == c].mean(axis=0) for c in classes])  # [K, D]
    # ||x-m||^2 = ||x||^2 - 2 x·m + ||m||^2 (avoids the [N,K,D] tensor)
    d2 = ((X * X).sum(1)[:, None]
          - 2.0 * (X @ means.T)
          + (means * means).sum(1)[None, :])
    return classes[np.argmin(d2, axis=1)]


def ncc_accuracy(X: np.ndarray, y: np.ndarray) -> float:
    """Nearest-class-mean accuracy (0-100). 100 for separable classes."""
    return float(100.0 * np.mean(_ncc_predict(X, y) == np.asarray(y)))


def ridge_weights(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge weights ``W = (X^T X + λI)^-1 X^T Y`` (shape [D, K]).

    Equals ``sklearn.linear_model.Ridge(alpha=λ, fit_intercept=False).coef_.T``;
    no SGD, no sklearn import — bit-exact and reproducible."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    d = X.shape[1]
    a = X.T @ X + float(lam) * np.eye(d)
    return np.linalg.solve(a, X.T @ Y)


def _remap_labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map labels to contiguous 0..K-1; return (indices, classes)."""
    classes = np.unique(y)
    lookup = {int(c): i for i, c in enumerate(classes)}
    return np.array([lookup[int(v)] for v in y]), classes


def _ridge_predict(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Predicted class INDICES (0..K-1) from the closed-form ridge probe."""
    yi, classes = _remap_labels(np.asarray(y))
    one_hot = np.eye(len(classes))[yi]
    scores = np.asarray(X, dtype=np.float64) @ ridge_weights(X, one_hot, lam)
    return np.argmax(scores, axis=1)


def ridge_probe_accuracy(X: np.ndarray, y: np.ndarray, *, lam: float) -> float:
    """Closed-form ridge linear-probe accuracy (0-100) recovering ``y`` from
    frozen features. 100 for linearly separable classes."""
    yi, _ = _remap_labels(np.asarray(y))
    return float(100.0 * np.mean(_ridge_predict(X, y, lam) == yi))


# ---------------------------------------------------------------------------
# Registered metrics (external=True — off the default panel)
# ---------------------------------------------------------------------------

def _features(ctx: "EvalContext", role: str, name: str) -> tuple[np.ndarray, "object"]:
    out = ctx.outputs(role, "retain_test")
    if out.features is None:
        raise MetricError(
            f"{name} requires penultimate features for role {role!r}; features "
            "are only captured by model-in probes (see ARCH_FEATURE_RESOLVERS)")
    return np.asarray(out.features, dtype=np.float64).reshape(len(out.features), -1), out


def _cka_metric(ctx: "EvalContext", ref_role: str, name: str) -> MetricResult:
    xu, _ = _features(ctx, "unlearned", name)
    xr, _ = _features(ctx, ref_role, name)
    xu, xr = xu[:_AD_MAX_SAMPLES], xr[:_AD_MAX_SAMPLES]
    if xu.shape[0] != xr.shape[0]:
        raise MetricError(f"{name}: row count mismatch {xu.shape[0]} vs {xr.shape[0]}")
    value = linear_cka(xu, xr)
    n = xu.shape[0]
    rng = numpy_rng(ctx.seed, f"{name}:bootstrap")

    def _stat(resampled: dict) -> float:
        idx = resampled["rows"]
        return linear_cka(xu[idx], xr[idx])  # estimator recomputed, pairing kept

    ci = bootstrap_ci_groups({"rows": np.arange(n)}, _stat,
                             n=ctx.hp.bootstrap.n, alpha=ctx.hp.bootstrap.alpha,
                             rng=rng)
    ctx.stamp(f"structural.{name}.reference", ref_role)
    return MetricResult(value=float(value), ci=ci, n=n)


@register_metric(name="cka_to_gold", table_id="M33", category="structural",
                 modalities=_CLS, input_modes={"model"}, needs_gold=True,
                 external=True, cost="moderate")
def cka_to_gold(ctx: "EvalContext") -> MetricResult:
    """Linear CKA between unlearned and gold penultimate features on retain_test
    (1 = identical representation geometry). M33."""
    return _cka_metric(ctx, "gold", "cka_to_gold")


@register_metric(name="cka_to_original", table_id="M34", category="structural",
                 modalities=_CLS, input_modes={"model"}, needs_original=True,
                 external=True, cost="moderate")
def cka_to_original(ctx: "EvalContext") -> MetricResult:
    """Linear CKA between unlearned and ORIGINAL (pre-unlearning) features —
    drift-from-parent. M34."""
    return _cka_metric(ctx, "original", "cka_to_original")


def _representation_correctness_ci(ctx: "EvalContext", correct: np.ndarray,
                                   name: str) -> tuple[float, float]:
    rng = numpy_rng(ctx.seed, f"{name}:bootstrap")
    return bootstrap_ci(correct, n=ctx.hp.bootstrap.n,
                        alpha=ctx.hp.bootstrap.alpha, rng=rng)


@register_metric(name="ncc_feature_acc", table_id="M35", category="structural",
                 modalities=_CLS, input_modes={"model"}, external=True,
                 cost="moderate")
def ncc_feature_acc(ctx: "EvalContext") -> MetricResult:
    """Nearest-class-mean accuracy on unlearned penultimate features (retain_test):
    does class geometry survive in the representation? M35."""
    x, out = _features(ctx, "unlearned", "ncc_feature_acc")
    x = x[:_AD_MAX_SAMPLES]
    y = np.asarray(out.targets)[:_AD_MAX_SAMPLES]
    if np.unique(y).size < 2:
        raise MetricSkip("not_applicable_mode",
                         "ncc_feature_acc needs >=2 classes in retain_test")
    preds = _ncc_predict(x, y)
    correct = (preds == y).astype(np.float64) * 100.0
    return MetricResult(value=float(np.mean(correct)),
                        ci=_representation_correctness_ci(ctx, correct, "ncc_feature_acc"),
                        n=len(y))


@register_metric(name="linear_probe_recoverability", table_id="M36",
                 category="structural", modalities=_CLS, input_modes={"model"},
                 external=True, cost="moderate")
def linear_probe_recoverability(ctx: "EvalContext") -> MetricResult:
    """Closed-form ridge linear-probe accuracy recovering labels from frozen
    unlearned features (the classification 'coreset effect'). M36."""
    x, out = _features(ctx, "unlearned", "linear_probe_recoverability")
    x = x[:_AD_MAX_SAMPLES]
    y = np.asarray(out.targets)[:_AD_MAX_SAMPLES]
    if np.unique(y).size < 2:
        raise MetricSkip("not_applicable_mode",
                         "linear_probe_recoverability needs >=2 classes in retain_test")
    yi, _ = _remap_labels(y)
    correct = (_ridge_predict(x, y, _RIDGE_LAMBDA) == yi).astype(np.float64) * 100.0
    return MetricResult(value=float(np.mean(correct)),
                        ci=_representation_correctness_ci(ctx, correct,
                                                          "linear_probe_recoverability"),
                        n=len(y))
