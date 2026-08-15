"""Generative metrics M28-M30 — registered interfaces, LLM-backend bodies.

The LLM generative rows (M28-M30) are registered now so the compatibility
matrix, default panels, and skip accounting are complete from v0.1: under
classification the feasibility pass emits ``not_applicable_modality`` before
any body runs. The bodies raise MetricError until the LLM backend lands
(design-only) — reaching a body means an llm adapter declared a default panel
containing a metric this version cannot compute, which must fail loudly (G4
fail-soft converts it to a ``runtime_error`` skip), never silently pass.

(Diffusion generative metrics are out of scope for v0.1.)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from trail.core.errors import MetricError
from trail.core.registry import register_metric
from trail.core.report import MetricResult

if TYPE_CHECKING:  # pragma: no cover
    from trail.core.context import EvalContext

logger = logging.getLogger("trail.metrics.generative")

_LLM: set[str] = {"llm"}
# Generation is required for every metric in this category -> model-in only.
_MODEL_IN: set[str] = {"model"}


def _unimplemented(name: str) -> MetricResult:
    raise MetricError(f"{name} is not implemented in trail v0.1 "
                      "(the LLM generative backend is design-only)")


@register_metric(name="tofu_forget_quality", table_id="M28",
                 category="generative", modalities=_LLM,
                 input_modes=_MODEL_IN, needs_gold=True, cost="expensive")
def tofu_forget_quality(ctx: "EvalContext") -> MetricResult:
    """M28 — TOFU forget quality: distributional test against a
    retained-reference (gold) model. Cataloged for the LLM backend."""
    return _unimplemented("tofu_forget_quality")


@register_metric(name="tofu_utility", table_id="M29", category="generative",
                 modalities=_LLM, input_modes=_MODEL_IN, cost="expensive")
def tofu_utility(ctx: "EvalContext") -> MetricResult:
    """M29 — TOFU model utility aggregate. Cataloged for the LLM backend
."""
    return _unimplemented("tofu_utility")


@register_metric(name="rouge_forget", table_id="M30", category="generative",
                 modalities=_LLM, input_modes=_MODEL_IN, cost="expensive")
def rouge_forget(ctx: "EvalContext") -> MetricResult:
    """M30 — ROUGE-L on forget completions. Cataloged for the LLM backend
."""
    return _unimplemented("rouge_forget")
