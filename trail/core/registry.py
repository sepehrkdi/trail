"""Metric registry: metrics are data (MetricSpec) plus a function.

Registration is static and import-time: importing
``trail.metrics`` populates METRIC_REGISTRY deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:  # pragma: no cover
    from trail.core.context import EvalContext
    from trail.core.report import MetricResult

# CONVENTION (F7 process gate — see aggregate.compute_composites):
#   * No metric without a registered M-ID. Before ``register_metric`` runs for
#     a new metric (or ``aggregate.COMPOSITE_SPECS`` gains a composite), reserve
#   * The ``Category`` Literal and the ``CATEGORIES`` tuple below are ONE COUPLED
#     EDIT: they must list the same names in the same order. Half-applying the
#     pair (a name in one but not the other) hard-fails import via the
#     ``category not in CATEGORIES`` guard in ``register_metric`` — adding a new
#     category therefore always touches BOTH lines plus a metrics/__init__
#     import-smoke test. Keep them in lockstep.
Category = Literal["accuracy", "privacy", "relearning",
                   "efficiency", "structural", "generative", "distributional",
                   "poisoning", "recovery", "robustness"]

CATEGORIES: tuple[str, ...] = ("accuracy", "privacy", "relearning",
                               "efficiency", "structural", "generative",
                               "distributional", "poisoning", "recovery",
                               "robustness")

COST_RANK: dict[str, int] = {"cheap": 0, "moderate": 1, "expensive": 2}


@dataclass
class MetricSpec:
    name: str
    table_id: str                      # metric-table row id, e.g. "M9" ("guard" for validity guards)
    category: Category
    modalities: set[str]
    input_modes: set[str]              # subset of {"outputs", "model"}
    needs_gold: bool = False
    needs_shadow: bool = False
    needs_original: bool = False
    cost: Literal["cheap", "moderate", "expensive"] = "cheap"
    fn: Callable[["EvalContext"], "MetricResult"] | None = None
    external: bool = False             # plugin-registered; excluded from default panel
    modes: set[str] = field(default_factory=set)  # restrict to forgetting modes; empty = all


METRIC_REGISTRY: dict[str, MetricSpec] = {}


def register_metric(*, name: str, table_id: str, category: Category,
                    modalities: set[str], input_modes: set[str],
                    needs_gold: bool = False, needs_shadow: bool = False,
                    needs_original: bool = False,
                    cost: str = "cheap", external: bool = False,
                    modes: set[str] | None = None) -> Callable:
    """Decorator registering ``fn(ctx) -> MetricResult`` under ``name``."""
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}")
    if cost not in COST_RANK:
        raise ValueError(f"unknown cost tier {cost!r}")

    def _register(fn: Callable) -> Callable:
        if name in METRIC_REGISTRY:
            raise ValueError(f"duplicate metric registration: {name!r}")
        METRIC_REGISTRY[name] = MetricSpec(
            name=name, table_id=table_id, category=category,
            modalities=set(modalities), input_modes=set(input_modes),
            needs_gold=needs_gold, needs_shadow=needs_shadow,
            needs_original=needs_original,
            cost=cost, fn=fn, external=external, modes=set(modes or ()),
        )
        return fn

    return _register


def resolve(metrics: list[str] | str, registry: dict[str, MetricSpec],
            task: str, adapter) -> list[MetricSpec]:
    """Resolve a metric selection. Modality filtering does NOT happen here —
    the feasibility pass emits ``not_applicable_modality`` skips instead of
    silently dropping (G4)."""
    if metrics == "default":
        names = [n for n in adapter.default_metrics()
                 if n in registry and not registry[n].external]
    else:
        unknown = [n for n in metrics if n not in registry]
        if unknown:
            from trail.core.errors import RequestError
            raise RequestError(
                f"unknown metrics {unknown}; available: {sorted(registry)}")
        names = list(metrics)
    return [registry[n] for n in names]
