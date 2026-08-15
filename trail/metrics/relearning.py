"""Relearning-attack metrics M11/M12.

- ``relearn_forget`` (M11): attacker fine-tunes on N deleted samples
  (``forget_only`` source; Pawelczyk et al. 2024).
- ``relearn_retain_mix`` (M12): attacker fine-tunes on the retain pool plus N
  deleted samples (``retain_mix`` source); its ``n=0`` row trains on the
  ``retain_fraction`` slice of the retain pool only (benign-relearning
  baseline, Hu et al. 2025 proxy; identical to the ``retain_only`` source
  only when ``retain_fraction == 1.0``).

Headline value = post-attack forget-recovery accuracy (0-100) at the budget-100
cell (largest integer budget if 100 is not on the grid). The CI does NOT
re-run the attack: it bootstraps the per-example correctness of the relearned
model's final eval at the headline budget. Full per-epoch curves are stamped
into the report (curves, not endpoints).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from trail.attacks.relearn import (
    anamnesis,
    gold_relearn_curve,
    run_d2d_relearn_attack,
    run_relearn_attack,
    ttr_auc,
    ttr_epoch,
)
from trail.core.bootstrap import bootstrap_ci
from trail.core.errors import MissingReference
from trail.core.registry import register_metric
from trail.core.report import MetricResult
from trail.core.seeding import numpy_rng

if TYPE_CHECKING:  # pragma: no cover
    from trail.core.context import EvalContext

logger = logging.getLogger("trail.metrics.relearning")

# Attack knobs a metric override may touch. ``source`` is deliberately NOT
# overridable: it is the identity of M11 vs M12, not a hyperparameter.
_OVERRIDABLE: frozenset[str] = frozenset(
    {"budgets", "epochs", "lr", "momentum", "weight_decay", "batch_size",
     "retain_fraction"})


def _attack_params(ctx: "EvalContext", name: str) -> dict[str, Any]:
    """Frozen-protocol attack hyperparameters from ``hp.relearn``, merged with
    ``hp.metric_overrides[name]`` (unknown keys and ``source`` are ignored
    with a log line — they never reach the attack)."""
    rl = ctx.hp.relearn
    params: dict[str, Any] = {
        "budgets": list(rl.budgets),
        "epochs": rl.epochs,
        "lr": rl.lr,
        "momentum": rl.momentum,
        "weight_decay": rl.weight_decay,
        "batch_size": rl.batch_size,
        "retain_fraction": rl.retain_fraction,
    }
    overrides = dict(ctx.hp.metric_overrides.get(name, {}) or {})
    for k, v in overrides.items():
        if k in _OVERRIDABLE:
            params[k] = v
        else:
            logger.warning("%s: ignoring non-overridable/unknown override "
                           "%r=%r", name, k, v)
    return params


def _headline_label(budgets: list[Any]) -> str:
    """Headline cell: budget 100 if on the grid; else the largest integer
    budget; else (integer-free grid) the 'full' cell."""
    ints = sorted(b for b in budgets
                  if isinstance(b, int) and not isinstance(b, bool))
    if 100 in ints:
        return "100"
    if ints:
        return str(ints[-1])
    return "full"


def _relearn_metric(ctx: "EvalContext", *, name: str, source: str,
                    ) -> "MetricResult":
    """Shared body for M11/M12: run the grid, assemble components, bootstrap
    the headline cell's per-example correctness, stamp the curves."""
    params = _attack_params(ctx, name)
    budgets = list(params["budgets"])

    results = run_relearn_attack(ctx, role="unlearned", source=source, **{
        k: params[k] for k in ("budgets", "epochs", "lr", "momentum",
                               "weight_decay", "batch_size", "retain_fraction")
    })

    headline = _headline_label(budgets)
    if headline not in results:
        # Defensive: grid produced no headline cell (e.g. empty budgets).
        headline = next(iter(results))
        logger.warning("%s: headline budget missing from results; "
                       "falling back to %r", name, headline)
    head = results[headline]
    value = float(head["post"])

    # Components: per-budget endpoints + curve scalars (on the full-set curve;
    # falls back to the headline curve if "full" is not on the grid).
    components: dict[str, float] = {}
    for b in budgets:
        if isinstance(b, int) and not isinstance(b, bool) and str(b) in results:
            components[f"n{b}"] = float(results[str(b)]["post"])
    if "full" in results:
        components["full"] = float(results["full"]["post"])
    curve_label = "full" if "full" in results else headline
    history = list(results[curve_label]["history"])
    t = ttr_epoch(history)
    if t is not None:
        components["ttr_epoch"] = float(t)
    auc_u = float(ttr_auc(history))
    components["ttr_auc"] = auc_u

    # Anamnesis index: gold-tier; omitted when gold is unavailable.
    try:
        gold_results = gold_relearn_curve(ctx, source=source, **{
            k: params[k] for k in ("budgets", "epochs", "lr", "momentum",
                                   "weight_decay", "batch_size",
                                   "retain_fraction")
        })
        if curve_label in gold_results:
            auc_g = float(ttr_auc(gold_results[curve_label]["history"]))
            components["ain"] = anamnesis(auc_u, auc_g)
    except MissingReference as e:
        logger.info("%s: gold unavailable (%s); omitting anamnesis index",
                    name, e.skip_code)

    # Stamp the full per-epoch curves and the collateral retain endpoints.
    ctx.stamp(f"relearning.{name}.curves",
              {label: list(r["history"]) for label, r in results.items()})
    ctx.stamp(f"relearning.{name}.retain_post",
              {label: float(r["retain_post"]) for label, r in results.items()})
    ctx.stamp(f"relearning.{name}.headline_budget", headline)

    # CI over the relearned model's per-example correctness at the headline
    # budget — NOT over re-running the attack.
    per_example = np.asarray(head["post_correct"], dtype=np.float64)
    rng = numpy_rng(ctx.seed, f"{name}:bootstrap")
    ci = bootstrap_ci(per_example, lambda a: float(np.mean(a) * 100.0),
                      n=ctx.hp.bootstrap.n, alpha=ctx.hp.bootstrap.alpha,
                      rng=rng)
    return MetricResult(value=value, ci=ci, n=int(per_example.size),
                        components=components)


@register_metric(name="relearn_forget", table_id="M11", category="relearning",
                 modalities={"classification"}, input_modes={"model"},
                 cost="expensive")
def relearn_forget(ctx: "EvalContext") -> "MetricResult":
    """M11 — forget-only relearning attack (Pawelczyk et al. 2024).

    Fine-tunes the unlearned model on N deleted samples per the frozen-protocol
    grid; value = post-attack forget-recovery accuracy (0-100) at the headline
    budget. Components: per-budget endpoints, time-to-recovery epoch,
    recovery AUC, and (gold-tier) the anamnesis index.
    """
    return _relearn_metric(ctx, name="relearn_forget", source="forget_only")


@register_metric(name="relearn_retain_mix", table_id="M12",
                 category="relearning", modalities={"classification"},
                 input_modes={"model"}, cost="expensive")
def relearn_retain_mix(ctx: "EvalContext") -> "MetricResult":
    """M12 — retain-mix relearning attack.

    Fine-tunes the unlearned model on the retain pool plus N deleted samples;
    the ``n=0`` row trains on the ``retain_fraction`` slice of the retain
    pool only — the benign-relearning baseline (Hu et al. 2025 proxy; equal
    to the ``retain_only`` source only at ``retain_fraction == 1.0``).
    Value/components as in M11.
    """
    return _relearn_metric(ctx, name="relearn_retain_mix", source="retain_mix")


# Knobs a metric override may touch for the D2D variant.
_D2D_OVERRIDABLE: frozenset[str] = frozenset(
    {"d2d_relearn_fraction", "epochs", "lr", "momentum", "weight_decay",
     "batch_size"})


def _d2d_params(ctx: "EvalContext") -> dict[str, Any]:
    """Frozen D2D hyperparameters from ``hp.relearn`` + the
    ``metric_overrides['relearn_d2d']`` subset."""
    rl = ctx.hp.relearn
    params: dict[str, Any] = {
        "d2d_relearn_fraction": rl.d2d_relearn_fraction,
        "epochs": rl.epochs, "lr": rl.lr, "momentum": rl.momentum,
        "weight_decay": rl.weight_decay, "batch_size": rl.batch_size,
    }
    for k, v in (ctx.hp.metric_overrides.get("relearn_d2d", {}) or {}).items():
        if k in _D2D_OVERRIDABLE:
            params[k] = v
        else:
            logger.warning("relearn_d2d: ignoring non-overridable/unknown "
                           "override %r=%r", k, v)
    return params


@register_metric(name="relearn_d2d", table_id="M13", category="relearning",
                 modalities={"classification"}, input_modes={"model"},
                 cost="expensive")
def relearn_d2d(ctx: "EvalContext") -> "MetricResult":
    """M13 — D2D sharpness-aware relearning attack (Fan et al. 2025, NPO-SAM).

    Carves the forget set into a disjoint ``d2d_relearn_fraction`` attack slice
    (the "1%" of a 10% forget set) and the held-out remainder eval slice (the
    "9%"); fine-tunes the unlearned model on the attack slice with the frozen
    recipe and measures forget recovery on the disjoint eval slice. Value =
    post-attack recovery accuracy (0-100) on the eval slice. Components: the
    pre-attack baseline, the recovery delta, the slice sizes, time-to-recovery
    epoch / AUC, and (gold-tier) the anamnesis index. The CI bootstraps the
    relearned model's per-example correctness on the eval slice (it does not
    re-run the attack).
    """
    params = _d2d_params(ctx)
    res = run_d2d_relearn_attack(ctx, role="unlearned", **params)

    value = float(res["post"])
    history = list(res["history"])
    components: dict[str, float] = {
        "pre": float(res["pre"]),
        "recovery_delta": value - float(res["pre"]),
        "n_relearn": float(res["n_relearn"]),
        "n_eval": float(res["n_eval"]),
        "ttr_auc": float(ttr_auc(history)),
    }
    t = ttr_epoch(history)
    if t is not None:
        components["ttr_epoch"] = float(t)

    # Anamnesis vs gold (gold-tier; omitted when gold is unavailable).
    try:
        ctx.gold()  # raises MissingReference with the right skip code
        gold_res = run_d2d_relearn_attack(ctx, role="gold", **params)
        auc_g = float(ttr_auc(list(gold_res["history"])))
        components["ain"] = anamnesis(components["ttr_auc"], auc_g)
    except MissingReference as e:
        logger.info("relearn_d2d: gold unavailable (%s); omitting anamnesis",
                    e.skip_code)

    ctx.stamp("relearning.relearn_d2d.curve", history)
    ctx.stamp("relearning.relearn_d2d.retain_post", float(res["retain_post"]))

    per_example = np.asarray(res["post_correct"], dtype=np.float64)
    rng = numpy_rng(ctx.seed, "relearn_d2d:bootstrap")
    ci = bootstrap_ci(per_example, lambda a: float(np.mean(a) * 100.0),
                      n=ctx.hp.bootstrap.n, alpha=ctx.hp.bootstrap.alpha,
                      rng=rng)
    return MetricResult(value=value, ci=ci, n=int(per_example.size),
                        components=components)


# ---------------------------------------------------------------------------
# M60 — efficacy vs compute (Tier-2; external; reuses the M11 relearn grid).
# ---------------------------------------------------------------------------

def early_recovery_efficiency(budget_to_post: "dict[str, float]") -> float:
    """Fraction (0-100) of the full-budget forget-recovery accuracy already
    achieved at the SMALLEST non-zero compute budget. High = forgetting unravels
    cheaply (weak unlearning). ``budget_to_post`` maps ``run_relearn_attack``
    labels ("0"/"10"/"100"/"full") to post-attack forget accuracy.
    """
    numeric = {int(k): float(v) for k, v in budget_to_post.items()
               if k.lstrip("-").isdigit()}
    ref = budget_to_post.get("full")
    ref = float(ref) if ref is not None else (
        max(numeric.values()) if numeric else float("nan"))
    nz = sorted(b for b in numeric if b > 0)
    if not nz or not np.isfinite(ref) or ref <= 0:
        return float("nan")
    return float(100.0 * numeric[nz[0]] / ref)


@register_metric(name="efficacy_vs_compute", table_id="M60", category="relearning",
                 modalities={"classification"}, input_modes={"model"},
                 external=True, cost="expensive")
def efficacy_vs_compute(ctx: "EvalContext") -> "MetricResult":
    """M60 — early relearning efficiency (external, PROVISIONAL): the fraction of
    the full-budget forget recovery achieved at the smallest non-zero compute
    budget, over the frozen M11 forget-only grid. A relearning attack — its
    recipe is disclosed via the ``relearn`` family (it is in ``_ATTACK_METRICS``).
    No CI (a single derived scalar)."""
    params = _attack_params(ctx, "efficacy_vs_compute")
    results = run_relearn_attack(ctx, role="unlearned", source="forget_only", **{
        k: params[k] for k in ("budgets", "epochs", "lr", "momentum",
                               "weight_decay", "batch_size", "retain_fraction")})
    posts = {label: float(r["post"]) for label, r in results.items()}
    value = early_recovery_efficiency(posts)
    ctx.stamp("relearning.efficacy_vs_compute.posts", posts)
    n = next((len(np.asarray(r["post_correct"])) for r in results.values()
              if "post_correct" in r), len(posts))
    return MetricResult(value=value, ci=(value, value), n=int(n))
