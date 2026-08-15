"""Accuracy / utility metrics M1-M7 + M21.

Pure summarizers over SplitOutputs — outputs-in sufficient, cheap, one
L1-cached forward pass per (role, split). All values on the 0-100 scale.

Headline semantics ported from the original evaluation pipeline:
``unlearning_acc`` (143-152), ``forget_gap_to_gold`` (155-164),
``headline_test_acc`` (188-216).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from trail.core.bootstrap import bootstrap_ci, bootstrap_ci_groups
from trail.core.errors import MetricError, SplitNotAvailable
from trail.core.registry import register_metric
from trail.core.report import MetricResult
from trail.core.seeding import numpy_rng
from trail.metrics._accuracy_core import acc100, accuracy_pct, correctness

if TYPE_CHECKING:  # pragma: no cover
    from trail.core.context import EvalContext

logger = logging.getLogger("trail.metrics.accuracy")

_CLS: set[str] = {"classification"}
_BOTH_MODES: set[str] = {"outputs", "model"}


def _acc_stat(arr: np.ndarray) -> float:
    """Bootstrap statistic: mean correctness on the 0-100 scale."""
    return float(100.0 * np.mean(arr))


def _split_accuracy(ctx: "EvalContext", *, name: str, split: str) -> MetricResult:
    """Accuracy of the unlearned model on one split, with a per-example
    bootstrap CI over the correctness mask."""
    out = ctx.outputs("unlearned", split)
    corr = correctness(out).astype(np.float64)  # per-example mask -> CI (G5)
    rng = numpy_rng(ctx.seed, f"{name}:bootstrap")
    ci = bootstrap_ci(corr, _acc_stat, n=ctx.hp.bootstrap.n,
                      alpha=ctx.hp.bootstrap.alpha, rng=rng)
    # Point estimate via the TorchMetrics primitive (identical to acc100(corr)).
    value = accuracy_pct(out.logits, out.targets)
    return MetricResult(value=value, ci=ci, n=out.n, components=None)


def _ta_basis(ctx: "EvalContext") -> str:
    """Mode-aware test-accuracy basis split.

    Port of the original evaluation pipeline (``headline_test_acc``):
    single_class -> test set restricted to retained classes (``retain_test``);
    random / sub_class_atypical (and any future non-class mode) -> full
    ``test`` split.
    """
    return "retain_test" if ctx.mode == "single_class" else "test"


def _paired_correctness(ctx: "EvalContext", split: str) -> np.ndarray:
    """Stacked paired rows ``[N, 2]`` of (unlearned, gold) correctness on the
    same split — paired bootstrap resamples the shared example indices."""
    u = correctness(ctx.outputs("unlearned", split)).astype(np.float64)
    g = correctness(ctx.outputs("gold", split)).astype(np.float64)
    if len(u) != len(g):
        raise MetricError(
            f"paired comparison on split {split!r}: unlearned has {len(u)} "
            f"examples but gold has {len(g)} — canonical views diverged")
    return np.stack([u, g], axis=1)


def _abs_delta(pairs: np.ndarray) -> float:
    """|acc(unlearned) - acc(gold)| in points from one paired-rows array."""
    return abs(100.0 * float(np.mean(pairs[:, 0])) -
               100.0 * float(np.mean(pairs[:, 1])))


@register_metric(name="ra_train", table_id="M1", category="accuracy",
                 modalities=_CLS, input_modes=_BOTH_MODES, cost="cheap")
def ra_train(ctx: "EvalContext") -> MetricResult:
    """M1 — retain-set (train-side) accuracy of the unlearned model."""
    return _split_accuracy(ctx, name="ra_train", split="retain")


@register_metric(name="ra_test", table_id="M2", category="accuracy",
                 modalities=_CLS, input_modes=_BOTH_MODES, cost="cheap")
def ra_test(ctx: "EvalContext") -> MetricResult:
    """M2 — retained-classes test accuracy of the unlearned model."""
    return _split_accuracy(ctx, name="ra_test", split="retain_test")


@register_metric(name="fa_train", table_id="M3", category="accuracy",
                 modalities=_CLS, input_modes=_BOTH_MODES, cost="cheap")
def fa_train(ctx: "EvalContext") -> MetricResult:
    """M3 — train-side forget accuracy (the SCRUB/L1-Sparse/SalUn basis)."""
    return _split_accuracy(ctx, name="fa_train", split="forget")


@register_metric(name="fa_test", table_id="M4", category="accuracy",
                 modalities=_CLS, input_modes=_BOTH_MODES, cost="cheap")
def fa_test(ctx: "EvalContext") -> MetricResult:
    """M4 — test-side forget accuracy (class/concept modes only).

    In modes where ``forget_test`` is empty by design (e.g. random),
    ``ctx.outputs`` raises SplitNotAvailable and the runner records the
    ``not_applicable_mode`` skip — no extra guard here.
    """
    return _split_accuracy(ctx, name="fa_test", split="forget_test")


@register_metric(name="ua", table_id="M5", category="accuracy",
                 modalities=_CLS, input_modes=_BOTH_MODES, cost="cheap")
def ua(ctx: "EvalContext") -> MetricResult:
    """M5 — unlearning accuracy, ``100 - fa_train``.

    Literature convention (Kurmanji et al. 2023 §3.1; Jia et al. 2023 Eq. 1);
    port of the original evaluation pipeline (``unlearning_acc``).
    Headline for class forgetting; for random forgetting the honest headline
    is M7/M21 (gap to gold), not this absolute number.
    """
    out = ctx.outputs("unlearned", "forget")
    corr = correctness(out).astype(np.float64)  # per-example mask -> CI (G5)
    rng = numpy_rng(ctx.seed, "ua:bootstrap")
    ci = bootstrap_ci(corr, lambda a: 100.0 - 100.0 * float(np.mean(a)),
                      n=ctx.hp.bootstrap.n, alpha=ctx.hp.bootstrap.alpha,
                      rng=rng)
    fa = accuracy_pct(out.logits, out.targets)  # TorchMetrics primitive
    return MetricResult(value=100.0 - fa, ci=ci, n=out.n,
                        components={"fa_train": fa})


@register_metric(name="ta", table_id="M6", category="accuracy",
                 modalities=_CLS, input_modes=_BOTH_MODES, cost="cheap")
def ta(ctx: "EvalContext") -> MetricResult:
    """M6 — mode-aware headline test accuracy.

    Port of the original evaluation pipeline (``headline_test_acc``):
    single_class evaluates on the retained-classes test partition; other
    modes on the full test set. The chosen basis is stamped into the report.

    Fallback: when the class-mode basis (``retain_test``) cannot be
    materialized — a raw user bundle whose test partition the adapter cannot
    derive raises SplitNotAvailable — ta falls back to FULL-test accuracy
    with a recorded warning and stamps the actually-used basis. Never silent,
    never a skip: the mode is applicable; only the partition is underivable.
    """
    basis = _ta_basis(ctx)
    try:
        result = _split_accuracy(ctx, name="ta", split=basis)
    except SplitNotAvailable:
        if basis == "test":
            raise  # nothing to fall back to
        msg = (f"ta: basis split {basis!r} unavailable in mode {ctx.mode!r} "
               "(test partition not derivable from the supplied bundle); "
               "falling back to full-test accuracy")
        logger.warning(msg)
        ctx.warnings.append(msg)
        basis = "test"
        result = _split_accuracy(ctx, name="ta", split=basis)
    ctx.stamp("accuracy.ta.basis", basis)
    return result


@register_metric(name="forget_gap_to_gold", table_id="M7", category="accuracy",
                 modalities=_CLS, input_modes=_BOTH_MODES, needs_gold=True,
                 cost="cheap")
def forget_gap_to_gold(ctx: "EvalContext") -> MetricResult:
    """M7 — ``|fa_train(unlearned) - fa_train(gold)|``, the random-mode headline.

    Semantics ported from the original evaluation pipeline
    (``forget_gap_to_gold``), upgraded from the dense-model proxy to the true
    gold reference. CI is a PAIRED bootstrap: both models are evaluated on
    the same forget examples, so resampling draws shared example indices
    (one group of stacked [N, 2] correctness rows).
    """
    pairs = _paired_correctness(ctx, "forget")
    rng = numpy_rng(ctx.seed, "forget_gap_to_gold:bootstrap")
    ci = bootstrap_ci_groups({"forget": pairs},
                             lambda groups: _abs_delta(groups["forget"]),
                             n=ctx.hp.bootstrap.n, alpha=ctx.hp.bootstrap.alpha,
                             rng=rng)
    fa_u = acc100(pairs[:, 0])
    fa_g = acc100(pairs[:, 1])
    return MetricResult(value=abs(fa_u - fa_g), ci=ci, n=len(pairs),
                        components={"fa_train_unlearned": fa_u,
                                    "fa_train_gold": fa_g})


@register_metric(name="sum_delta_to_gold", table_id="M21", category="accuracy",
                 modalities=_CLS, input_modes=_BOTH_MODES, needs_gold=True,
                 cost="cheap")
def sum_delta_to_gold(ctx: "EvalContext") -> MetricResult:
    """M21 — one-number gold-relative aggregate: Σ|Δ| over {UA, RA_train,
    RA_test, TA} between the unlearned and gold models.

    The TA component uses the mode-aware basis of M6 (in single_class mode it
    coincides with RA_test by construction; both are still reported).
    Components carry the SIGNED deltas (unlearned - gold; d_ua in UA space,
    so d_ua = -(fa_u - fa_g)). CI bootstraps the four splits as independent
    groups of paired (unlearned, gold) correctness rows.
    """
    basis = _ta_basis(ctx)
    ctx.stamp("accuracy.sum_delta_to_gold.ta_basis", basis)
    split_for = {"ua": "forget", "ra_train": "retain",
                 "ra_test": "retain_test", "ta": basis}
    groups: dict[str, np.ndarray] = {}
    deltas: dict[str, float] = {}
    n_total = 0
    for key, split in split_for.items():
        pairs = _paired_correctness(ctx, split)
        groups[key] = pairs
        signed = (100.0 * float(np.mean(pairs[:, 0])) -
                  100.0 * float(np.mean(pairs[:, 1])))
        # UA = 100 - fa_train, so the UA-space delta flips sign vs the
        # forget-accuracy delta; |d_ua| is unchanged.
        deltas[f"d_{key}"] = -signed if key == "ua" else signed
        n_total += len(pairs)

    def _stat(resampled: dict[str, np.ndarray]) -> float:
        return float(sum(_abs_delta(arr) for arr in resampled.values()))

    rng = numpy_rng(ctx.seed, "sum_delta_to_gold:bootstrap")
    ci = bootstrap_ci_groups(groups, _stat, n=ctx.hp.bootstrap.n,
                             alpha=ctx.hp.bootstrap.alpha, rng=rng)
    value = float(sum(abs(v) for v in deltas.values()))
    return MetricResult(value=value, ci=ci, n=n_total, components=deltas)
