"""Distributional / two-sample metrics (Tier-2, M46-M51) — off the default panel
(``external=True``).

Two-sample statistics over the per-example loss distributions already computed
for the loss-MIA (M8 family): forget (member-candidate) vs test (non-member)
losses, and unlearned-vs-gold forget losses. ``scipy`` is lazy-imported. CIs use
the frozen-estimator group bootstrap (each sample resampled independently, the
statistic RECOMPUTED, not averaged per-example; G5).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from trail.core.bootstrap import bootstrap_ci_groups
from trail.core.errors import MetricError
from trail.core.registry import register_metric
from trail.core.report import MetricResult
from trail.core.seeding import numpy_rng

if TYPE_CHECKING:  # pragma: no cover
    from trail.core.context import EvalContext

logger = logging.getLogger("trail.metrics.distributional")

_CLS: set[str] = {"classification"}

#: pinned JS histogram bins — fixed range/count so loss_js is comparable across
#: methods (losses are CE nats, typically 0..~10; out-of-range mass is clipped).
_JS_BINS = 64
_JS_RANGE = (0.0, 10.0)

#: thresholds for forget_loss_polarization (low = still-confident, high = forgotten).
_POLAR_LOW = 0.5
_POLAR_HIGH = 2.0


# ---------------------------------------------------------------------------
# Pure two-sample helpers (lazy scipy)
# ---------------------------------------------------------------------------

def loss_ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    """Kolmogorov-Smirnov 2-sample statistic (0 = indistinguishable, 1 = disjoint)."""
    from scipy import stats  # lazy
    return float(stats.ks_2samp(np.asarray(a, dtype=np.float64),
                                np.asarray(b, dtype=np.float64)).statistic)


def loss_wasserstein_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Wasserstein-1 (earth-mover) distance between two 1-D samples."""
    from scipy import stats  # lazy
    return float(stats.wasserstein_distance(np.asarray(a, dtype=np.float64),
                                            np.asarray(b, dtype=np.float64)))


def loss_js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """Jensen-Shannon DIVERGENCE (bits, [0,1]) between loss histograms on the
    pinned ``_JS_BINS``/``_JS_RANGE`` grid. 0 for identical, ~1 for disjoint."""
    from scipy.spatial.distance import jensenshannon  # lazy
    edges = np.linspace(_JS_RANGE[0], _JS_RANGE[1], _JS_BINS + 1)
    pa, _ = np.histogram(np.clip(np.asarray(a, dtype=np.float64), *_JS_RANGE), bins=edges)
    pb, _ = np.histogram(np.clip(np.asarray(b, dtype=np.float64), *_JS_RANGE), bins=edges)
    if pa.sum() == 0 or pb.sum() == 0:
        return float("nan")
    dist = jensenshannon(pa.astype(np.float64), pb.astype(np.float64), base=2)
    return float(dist * dist) if np.isfinite(dist) else float("nan")  # distance^2 = divergence


def state_dict_l2(sd_a: dict, sd_b: dict) -> float:
    """L2 distance over parameters shared by two state dicts: ``√Σ‖a_k − b_k‖²``."""
    total = 0.0
    for key, va in sd_a.items():
        vb = sd_b.get(key)
        if vb is None or not hasattr(va, "shape") or tuple(va.shape) != tuple(vb.shape):
            continue
        diff = np.asarray(va, dtype=np.float64) - np.asarray(vb, dtype=np.float64)
        total += float((diff * diff).sum())
    return float(np.sqrt(total))


# ---------------------------------------------------------------------------
# Registered metrics (external=True)
# ---------------------------------------------------------------------------

def _losses(ctx: "EvalContext", role: str, split: str, name: str) -> np.ndarray:
    out = ctx.outputs(role, split)
    if out.losses is None:
        raise MetricError(f"{name}: no per-example losses for {role}/{split}")
    return np.asarray(out.losses, dtype=np.float64)


def _two_sample_ci(ctx: "EvalContext", a: np.ndarray, b: np.ndarray,
                   stat, name: str) -> tuple[float, float]:
    """Frozen-estimator group bootstrap: resample a and b independently, recompute."""
    rng = numpy_rng(ctx.seed, f"{name}:bootstrap")

    def _stat(res: dict) -> float:
        return stat(res["a"], res["b"])

    return bootstrap_ci_groups({"a": a, "b": b}, _stat,
                               n=ctx.hp.bootstrap.n, alpha=ctx.hp.bootstrap.alpha,
                               rng=rng)


@register_metric(name="loss_ks", table_id="M46", category="distributional",
                 modalities=_CLS, input_modes={"outputs", "model"},
                 external=True, cost="cheap")
def loss_ks(ctx: "EvalContext") -> MetricResult:
    """KS 2-sample between forget and test per-example losses (M46)."""
    a = _losses(ctx, "unlearned", "forget", "loss_ks")
    b = _losses(ctx, "unlearned", "test", "loss_ks")
    value = loss_ks_stat(a, b)
    ci = _two_sample_ci(ctx, a, b, loss_ks_stat, "loss_ks")
    return MetricResult(value=value, ci=ci, n=len(a) + len(b))


@register_metric(name="loss_wasserstein", table_id="M47", category="distributional",
                 modalities=_CLS, input_modes={"outputs", "model"},
                 external=True, cost="cheap")
def loss_wasserstein(ctx: "EvalContext") -> MetricResult:
    """Wasserstein-1 between forget and test losses (M47)."""
    a = _losses(ctx, "unlearned", "forget", "loss_wasserstein")
    b = _losses(ctx, "unlearned", "test", "loss_wasserstein")
    value = loss_wasserstein_dist(a, b)
    ci = _two_sample_ci(ctx, a, b, loss_wasserstein_dist, "loss_wasserstein")
    return MetricResult(value=value, ci=ci, n=len(a) + len(b))


@register_metric(name="loss_js", table_id="M48", category="distributional",
                 modalities=_CLS, input_modes={"outputs", "model"},
                 external=True, cost="cheap")
def loss_js(ctx: "EvalContext") -> MetricResult:
    """Jensen-Shannon divergence (pinned bins) between forget and test losses (M48)."""
    a = _losses(ctx, "unlearned", "forget", "loss_js")
    b = _losses(ctx, "unlearned", "test", "loss_js")
    value = loss_js_divergence(a, b)
    ci = _two_sample_ci(ctx, a, b, loss_js_divergence, "loss_js")
    return MetricResult(value=value, ci=ci, n=len(a) + len(b))


@register_metric(name="loss_dist_to_gold", table_id="M49", category="distributional",
                 modalities=_CLS, input_modes={"outputs", "model"},
                 needs_gold=True, external=True, cost="cheap")
def loss_dist_to_gold(ctx: "EvalContext") -> MetricResult:
    """Wasserstein-1 between unlearned and GOLD forget-loss distributions (M49).
    How far is the unlearned forget-loss law from the oracle's?"""
    a = _losses(ctx, "unlearned", "forget", "loss_dist_to_gold")
    b = _losses(ctx, "gold", "forget", "loss_dist_to_gold")
    value = loss_wasserstein_dist(a, b)
    ci = _two_sample_ci(ctx, a, b, loss_wasserstein_dist, "loss_dist_to_gold")
    return MetricResult(value=value, ci=ci, n=len(a) + len(b))


@register_metric(name="forget_loss_polarization", table_id="M50",
                 category="distributional", modalities=_CLS,
                 input_modes={"outputs", "model"}, external=True, cost="cheap")
def forget_loss_polarization(ctx: "EvalContext") -> MetricResult:
    """Mass of forget losses in the low+high tails — bimodality of memorization
    (M50). High = polarized (some still-memorized, some fully-forgotten)."""
    a = _losses(ctx, "unlearned", "forget", "forget_loss_polarization")
    tail = ((a < _POLAR_LOW) | (a > _POLAR_HIGH)).astype(np.float64)
    rng = numpy_rng(ctx.seed, "forget_loss_polarization:bootstrap")
    from trail.core.bootstrap import bootstrap_ci
    ci = bootstrap_ci(tail, n=ctx.hp.bootstrap.n, alpha=ctx.hp.bootstrap.alpha, rng=rng)
    return MetricResult(value=float(np.mean(tail)), ci=ci, n=len(a),
                        components={"frac_low": float(np.mean(a < _POLAR_LOW)),
                                    "frac_high": float(np.mean(a > _POLAR_HIGH))})


@register_metric(name="weight_l2_to_gold", table_id="M51", category="structural",
                 modalities=_CLS, input_modes={"model"}, needs_gold=True,
                 external=True, cost="cheap")
def weight_l2_to_gold(ctx: "EvalContext") -> MetricResult:
    """Raw parameter-space drift ‖θ(A_u) − θ(A_r)‖₂ over shared params (M51).
    Model-in only (reads weights, not outputs); no CI (a single deterministic
    scalar — point estimate)."""
    import torch  # local

    u = ctx.model("unlearned").state_dict()
    g = ctx.model("gold").state_dict()
    sd_u = {k: v.detach().cpu().numpy() for k, v in u.items()
            if isinstance(v, torch.Tensor) and v.is_floating_point()}
    sd_g = {k: v.detach().cpu().numpy() for k, v in g.items()
            if isinstance(v, torch.Tensor) and v.is_floating_point()}
    value = state_dict_l2(sd_u, sd_g)
    return MetricResult(value=value, ci=(value, value), n=len(sd_u))
