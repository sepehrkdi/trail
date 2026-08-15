"""White-box MIA metrics (Tier-2 attack family) — ``external=True`` (off the
default panel).

The optimal balanced-accuracy of a single-threshold attack on a scalar signal is
``0.5·(1 + max_t|TPR(t) − FPR(t)|)`` and ``max_t|TPR − FPR|`` is exactly the
two-sample Kolmogorov-Smirnov statistic — so the MIA *advantage* of any scalar
signal is its KS statistic (forget = member-candidate vs test = non-member),
computed exactly in O(N log N). Signals:

* ``mia_whitebox_gradient`` (M52): per-example ∂CE/∂θ norm (members fit better →
  smaller gradients), via the shared grad scaffold (deepcopy-before-grad).
* ``mia_whitebox_activation`` (M53): cached penultimate-feature norm (no grad).
* ``mia_worst_case`` (M54): the MAX advantage over the INSTALLED non-shadow
  attacks (loss / gradient / activation); the winning attack name is stamped
  (never a metric component). Never forces a shadow (LiRA) build.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from trail.core.errors import MetricError
from trail.core.registry import register_metric
from trail.core.report import MetricResult
from trail.core.seeding import numpy_rng

if TYPE_CHECKING:  # pragma: no cover
    from trail.core.context import EvalContext

logger = logging.getLogger("trail.metrics.whitebox")

_CLS: set[str] = {"classification"}
_AD_MAX_SAMPLES = 4096


def balanced_threshold_advantage(member: np.ndarray, nonmember: np.ndarray) -> float:
    """MIA advantage of a scalar signal = the 2-sample KS statistic (== the best
    single-threshold ``max|TPR−FPR|``, direction-free). 0 = indistinguishable,
    1 = perfectly separable."""
    from scipy import stats  # lazy
    a = np.asarray(member, dtype=np.float64)
    b = np.asarray(nonmember, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(stats.ks_2samp(a, b).statistic)


def _advantage_to_balacc(adv: float) -> float:
    return 0.5 * (1.0 + adv)


def _gather_inputs(ctx: "EvalContext", split: str, cap: int):
    """Capped (X, y) tensors from a split's canonical loader (deterministic order)."""
    import torch

    xs, ys, got = [], [], 0
    for xb, yb in ctx.loader(split):
        xs.append(xb)
        ys.append(yb)
        got += len(xb)
        if got >= cap:
            break
    return torch.cat(xs)[:cap], torch.cat(ys)[:cap]


def _grad_norms(ctx: "EvalContext", split: str) -> np.ndarray:
    from trail.attacks.whitebox import append_cuda_caveat, per_example_grad_norms

    append_cuda_caveat(ctx)
    x, y = _gather_inputs(ctx, split, _AD_MAX_SAMPLES)
    model = ctx.model("unlearned")
    return per_example_grad_norms(model, x, y, device=ctx.device,
                                  layer=ctx.hp.whitebox.grad_layer)


def _feature_signal(ctx: "EvalContext", split: str) -> np.ndarray:
    out = ctx.outputs("unlearned", split)
    if out.features is None:
        raise MetricError("mia_whitebox_activation requires penultimate features "
                          "(model-in probes only)")
    f = np.asarray(out.features, dtype=np.float64).reshape(len(out.features), -1)
    return np.linalg.norm(f[:_AD_MAX_SAMPLES], axis=1)


@register_metric(name="mia_whitebox_gradient", table_id="M52", category="privacy",
                 modalities=_CLS, input_modes={"model"}, external=True,
                 cost="expensive")
def mia_whitebox_gradient(ctx: "EvalContext") -> MetricResult:
    """Gradient-norm MIA: balanced attack accuracy distinguishing forget
    (member-candidate) from test (non-member) per-example gradient norms (M52)."""
    gm = _grad_norms(ctx, "forget")
    gn = _grad_norms(ctx, "test")
    adv = balanced_threshold_advantage(gm, gn)
    return MetricResult(value=_advantage_to_balacc(adv),
                        ci=(_advantage_to_balacc(adv), _advantage_to_balacc(adv)),
                        n=len(gm) + len(gn), components={"advantage": adv})


@register_metric(name="mia_whitebox_activation", table_id="M53", category="privacy",
                 modalities=_CLS, input_modes={"model"}, external=True,
                 cost="moderate")
def mia_whitebox_activation(ctx: "EvalContext") -> MetricResult:
    """Activation MIA: balanced attack accuracy on cached penultimate-feature
    norms, forget vs test (M53). No gradients."""
    sm = _feature_signal(ctx, "forget")
    sn = _feature_signal(ctx, "test")
    adv = balanced_threshold_advantage(sm, sn)
    return MetricResult(value=_advantage_to_balacc(adv),
                        ci=(_advantage_to_balacc(adv), _advantage_to_balacc(adv)),
                        n=len(sm) + len(sn), components={"advantage": adv})


@register_metric(name="mia_worst_case", table_id="M54", category="privacy",
                 modalities=_CLS, input_modes={"model"}, external=True,
                 cost="expensive")
def mia_worst_case(ctx: "EvalContext") -> MetricResult:
    """Worst-case (max) MIA advantage over the INSTALLED non-shadow attacks —
    loss-threshold, gradient, activation (M54). The winning attack name is
    stamped (never a component). NEVER forces a shadow (LiRA) build."""
    advantages: dict[str, float] = {}

    fm = np.asarray(ctx.outputs("unlearned", "forget").losses, dtype=np.float64)
    fn = np.asarray(ctx.outputs("unlearned", "test").losses, dtype=np.float64)
    advantages["loss"] = balanced_threshold_advantage(fm, fn)

    try:
        advantages["gradient"] = balanced_threshold_advantage(
            _grad_norms(ctx, "forget"), _grad_norms(ctx, "test"))
    except Exception as e:  # an unavailable attack is skipped, never fatal
        logger.info("mia_worst_case: gradient attack unavailable (%s)", e)
    try:
        advantages["activation"] = balanced_threshold_advantage(
            _feature_signal(ctx, "forget"), _feature_signal(ctx, "test"))
    except Exception as e:
        logger.info("mia_worst_case: activation attack unavailable (%s)", e)

    finite = {k: v for k, v in advantages.items() if np.isfinite(v)}
    if not finite:
        raise MetricError("mia_worst_case: no installed attack produced a signal")
    winner = max(finite, key=finite.get)
    adv = finite[winner]
    ctx.stamp("privacy.mia_worst_case.argmax_attack", winner)  # name via stamp, NOT components
    return MetricResult(value=_advantage_to_balacc(adv),
                        ci=(_advantage_to_balacc(adv), _advantage_to_balacc(adv)),
                        n=len(fm) + len(fn), components={"advantage": adv})
