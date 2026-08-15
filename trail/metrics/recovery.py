"""Recovery metrics (Tier-2) — ``external=True``, off the default panel.

``posthoc_output_recovery`` (M59) is computable from outputs alone: it Yeo-
Johnson-normalizes the forget-set confidences, Otsu-thresholds the result to
isolate a residual still-memorized cluster, and reports the recovery-GAIN over an
OOD (test-set) baseline. ``indiscriminate_recovery`` (M58) measures recovery from
indiscriminate poisoning and is blocked on an externally-produced poisoned
baseline artifact (``missing_artifact``-skips without one; UNVALIDATED).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from trail.core.errors import MetricError, MetricSkip
from trail.core.registry import register_metric
from trail.core.report import MetricResult

if TYPE_CHECKING:  # pragma: no cover
    from trail.core.context import EvalContext

logger = logging.getLogger("trail.metrics.recovery")

_CLS: set[str] = {"classification"}


# ----------------------------------------------------- synthetic-testable helpers


def _maxprob(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return p.max(axis=1)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold (maximizes between-class variance of a 1-D histogram)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan")
    hist, edges = np.histogram(v, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = hist.astype(np.float64)
    total = w.sum()
    if total == 0:
        return float(np.median(v))
    wB = np.cumsum(w)
    wF = total - wB
    sum_b = np.cumsum(w * centers)
    sum_total = (w * centers).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        mB = np.where(wB > 0, sum_b / wB, 0.0)
        mF = np.where(wF > 0, (sum_total - sum_b) / wF, 0.0)
    between = wB * wF * (mB - mF) ** 2
    return float(centers[int(np.argmax(between))])


def _yeojohnson_pair(forget: np.ndarray, ood: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Yeo-Johnson-normalize ``forget`` (fit λ on it) and apply the SAME λ to
    ``ood`` so both live on one scale (lazy scipy)."""
    from scipy import stats
    yj_f, lmbda = stats.yeojohnson(np.asarray(forget, dtype=np.float64))
    yj_o = stats.yeojohnson(np.asarray(ood, dtype=np.float64), lmbda=lmbda)
    return yj_f, yj_o


# ------------------------------------------------------------------- metrics


@register_metric(name="posthoc_output_recovery", table_id="M59", category="recovery",
                 modalities=_CLS, input_modes={"outputs", "model"},
                 external=True, cost="cheap")
def posthoc_output_recovery(ctx: "EvalContext") -> MetricResult:
    """Post-hoc output-recovery gain (M59): excess high-confidence forget mass
    over an OOD (test) baseline after Yeo-Johnson + Otsu (0-100, higher = more
    residual memorization recoverable from the outputs alone)."""
    fo = ctx.outputs("unlearned", "forget")
    to = ctx.outputs("unlearned", "test")
    if fo.logits is None or to.logits is None:
        raise MetricError("posthoc_output_recovery requires logits")
    fp = _maxprob(fo.logits)
    tp = _maxprob(to.logits)
    if np.std(fp) < 1e-9:  # degenerate (Yeo-Johnson undefined) -> no recoverable signal
        return MetricResult(value=0.0, ci=(0.0, 0.0), n=len(fp))
    yj_f, yj_t = _yeojohnson_pair(fp, tp)
    thr = otsu_threshold(yj_f)
    gain = 100.0 * (float(np.mean(yj_f > thr)) - float(np.mean(yj_t > thr)))
    return MetricResult(value=gain, ci=(gain, gain), n=len(fp),
                        components={"otsu_threshold": thr})


@register_metric(name="indiscriminate_recovery", table_id="M58", category="recovery",
                 modalities=_CLS, input_modes={"model"}, external=True,
                 cost="moderate")
def indiscriminate_recovery(ctx: "EvalContext") -> MetricResult:
    """Recovery from indiscriminate poisoning (M58). UNVALIDATED — blocked on an
    externally-produced poisoned-baseline artifact; ``missing_artifact``-skips
    when ``hp.metric_overrides['indiscriminate_recovery']['artifact']`` is absent."""
    from trail.core.artifacts import load_input_artifact

    path = (ctx.hp.metric_overrides.get("indiscriminate_recovery", {}) or {}).get("artifact")
    if not path:
        raise MetricSkip(
            "missing_artifact",
            "indiscriminate_recovery needs an externally-produced poisoned-"
            "baseline artifact; set hp.metric_overrides['indiscriminate_recovery']"
            "['artifact'] to a sha-stamped .npy/.npz of the baseline accuracies")
    try:
        baseline, sha = load_input_artifact(path)
    except FileNotFoundError:
        raise MetricSkip("missing_artifact",
                         f"indiscriminate_recovery: artifact not found at {path}")
    ctx.stamp("recovery.indiscriminate_recovery.artifact_sha256", sha)
    # recovery = current retain-test accuracy minus the poisoned baseline.
    from trail.metrics._accuracy_core import accuracy_pct
    out = ctx.outputs("unlearned", "retain_test")
    cur = accuracy_pct(out.logits, out.targets) if out.logits is not None else float("nan")
    value = float(cur - float(np.asarray(baseline).reshape(-1)[0]))
    return MetricResult(value=value, ci=(value, value), n=out.n)
