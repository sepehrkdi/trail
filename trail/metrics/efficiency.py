"""Efficiency metrics M31-M32: self-reported unlearning cost.

Structural limitation, stated plainly: trail evaluates checkpoints after
the fact and cannot measure the cost of an unlearning run it did not execute.
FDE and wall-clock are therefore SUPPLIED metadata — passed in the request,
stamped ``efficiency.self_reported = true``, and passed through with a
degenerate CI. Evaluation-side costs (per-metric cost_s, peak memory) are
measured directly by the runner wrapper, not here.

Skip-code note: when no self-reported cost metadata is supplied, these
metrics raise ``MetricSkip("missing_metadata", ...)`` — a dedicated SkipInfo
code (report.SKIP_CODES) so consumers filtering skips can distinguish
"user did not supply cost metadata" from mode-structural skips
(``not_applicable_mode``) and runtime failures.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np  # noqa: F401  (kept for parity with sibling metric modules)

from trail.core.errors import MetricSkip
from trail.core.registry import register_metric
from trail.core.report import MetricResult

if TYPE_CHECKING:  # pragma: no cover
    from trail.core.context import EvalContext

logger = logging.getLogger("trail.metrics.efficiency")

_ALL_MODALITIES: set[str] = {"classification", "llm"}
_BOTH_MODES: set[str] = {"outputs", "model"}


def _require_self_reported(ctx: "EvalContext", key: str) -> dict:
    """Fetch the self-reported cost dict, skipping if absent or missing
    ``key`` (see module docstring for the skip-code rationale)."""
    sr = ctx.self_reported()
    if not sr or key not in sr or sr[key] is None:
        raise MetricSkip(
            "missing_metadata",
            f"self_reported_cost not supplied in request (need {key!r})")
    ctx.stamp("efficiency.self_reported", True)
    return sr


@register_metric(name="fde", table_id="M31", category="efficiency",
                 modalities=_ALL_MODALITIES, input_modes=_BOTH_MODES,
                 cost="cheap")
def fde(ctx: "EvalContext") -> MetricResult:
    """M31 — full-data-equivalent epochs: FDE = Σᵢ epochsᵢ·|D_usedᵢ|/|D|.

    The hardware-independent unlearning cost unit. Self-reported passthrough
    (n=1, degenerate CI); computed by the method runner, not by trail.
    """
    sr = _require_self_reported(ctx, "fde")
    value = float(sr["fde"])
    return MetricResult(value=value, ci=(value, value), n=1, components=None)


@register_metric(name="wall_clock", table_id="M32", category="efficiency",
                 modalities=_ALL_MODALITIES, input_modes=_BOTH_MODES,
                 cost="cheap")
def wall_clock(ctx: "EvalContext") -> MetricResult:
    """M32 — unlearning wall-clock seconds (GPU model lives in provenance).

    Self-reported passthrough (n=1, degenerate CI). Optional supplied fields
    ``flops`` and ``peak_mem_mb`` are reporting fields attached here as
    components, not separate metrics.
    """
    sr = _require_self_reported(ctx, "wall_clock_s")
    value = float(sr["wall_clock_s"])
    components = {k: float(sr[k]) for k in ("flops", "peak_mem_mb")
                  if sr.get(k) is not None}
    return MetricResult(value=value, ci=(value, value), n=1,
                        components=components or None)


# ---------------------------------------------------------------------------
# M61 — metric_stability (Tier-2; external; eval-robustness sweep).
# ---------------------------------------------------------------------------

def stability_from_correct(correct: "np.ndarray", *, rng, n_boot: int,
                           alpha: float) -> float:
    """Eval-robustness score (0-100) = ``100 − bootstrap-CI-width`` (pp) of the
    accuracy implied by a per-example correctness array. 100 = perfectly stable
    (tight under eval-set resampling); lower = the metric swings with the draw."""
    from trail.core.bootstrap import bootstrap_ci

    c = np.asarray(correct, dtype=np.float64) * 100.0
    lo, hi = bootstrap_ci(c, n=n_boot, alpha=alpha, rng=rng)
    return float(max(0.0, 100.0 - (hi - lo)))


@register_metric(name="metric_stability", table_id="M61", category="efficiency",
                 modalities={"classification"}, input_modes={"outputs", "model"},
                 external=True, cost="cheap")
def metric_stability(ctx: "EvalContext") -> MetricResult:
    """M61 — eval-robustness (external, PROVISIONAL): tightness of the forget
    accuracy under eval-set bootstrap resampling (``100 − CI width`` in pp;
    100 = perfectly stable). A meta-metric on the headline forget signal."""
    from trail.core.errors import MetricError
    from trail.core.seeding import numpy_rng

    out = ctx.outputs("unlearned", "forget")
    if out.logits is None:
        raise MetricError("metric_stability requires logits to form the forget "
                          "accuracy signal")
    correct = (np.argmax(out.logits, axis=1) == np.asarray(out.targets)).astype(np.float64)
    rng = numpy_rng(ctx.seed, "metric_stability:bootstrap")
    value = stability_from_correct(correct, rng=rng, n_boot=ctx.hp.bootstrap.n,
                                   alpha=ctx.hp.bootstrap.alpha)
    return MetricResult(value=value, ci=(value, value), n=int(out.n))
