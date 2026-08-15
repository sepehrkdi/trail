"""Adversarial-robustness metrics (Tier-3) — NEW category ``robustness``,
``external=True`` (off the default panel). Forget accuracy under FGSM/PGD
perturbation: a high value means the forgetting survives an adversarial nudge.
Off the default panel; the perturbation recipe is disclosed via the ``adv``
family. GPU backward kernels are tolerance-only (caveat appended).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from trail.core.errors import MetricError
from trail.core.registry import register_metric
from trail.core.report import MetricResult

if TYPE_CHECKING:  # pragma: no cover
    from trail.core.context import EvalContext

logger = logging.getLogger("trail.metrics.robustness")

_CLS: set[str] = {"classification"}
_ADV_MAX_SAMPLES = 1024   # adversarial passes are gradient-heavy; cap the cohort


def _gather(ctx: "EvalContext", split: str, cap: int):
    import torch

    xs, ys, got = [], [], 0
    for xb, yb in ctx.loader(split):
        xs.append(xb)
        ys.append(yb)
        got += len(xb)
        if got >= cap:
            break
    if not xs:
        raise MetricError(f"empty split: {split}")
    return torch.cat(xs)[:cap], torch.cat(ys)[:cap]


def _adv_metric(ctx: "EvalContext", *, name: str, perturb_kind: str) -> MetricResult:
    from trail.attacks.adversarial import (
        adversarial_accuracy,
        append_cuda_caveat,
        fgsm_perturb,
        pgd_perturb,
    )

    append_cuda_caveat(ctx)  # GPU tolerance-only
    x, y = _gather(ctx, "forget", _ADV_MAX_SAMPLES)
    model = ctx.model("unlearned")
    adv = ctx.hp.adv
    if perturb_kind == "fgsm":
        def _p(xb, yb):
            return fgsm_perturb(model, xb, yb, eps=adv.eps, device=ctx.device)
        comp = {"eps": float(adv.eps)}
    else:
        def _p(xb, yb):
            return pgd_perturb(model, xb, yb, eps=adv.eps, alpha=adv.alpha,
                               steps=adv.steps, device=ctx.device)
        comp = {"eps": float(adv.eps), "alpha": float(adv.alpha),
                "steps": float(adv.steps)}
    acc = adversarial_accuracy(ctx.adapter, model, x, y, device=ctx.device, perturb=_p)
    return MetricResult(value=acc, ci=(acc, acc), n=int(len(y)), components=comp)


@register_metric(name="adv_robustness_fgsm", table_id="M66", category="robustness",
                 modalities=_CLS, input_modes={"model"}, external=True,
                 cost="expensive")
def adv_robustness_fgsm(ctx: "EvalContext") -> MetricResult:
    """M66 — forget accuracy under an FGSM perturbation (``hp.adv.eps``). High =
    the forgetting is adversarially robust. ``eps=0`` recovers clean forget acc."""
    return _adv_metric(ctx, name="adv_robustness_fgsm", perturb_kind="fgsm")


@register_metric(name="adv_robustness_pgd", table_id="M67", category="robustness",
                 modalities=_CLS, input_modes={"model"}, external=True,
                 cost="expensive")
def adv_robustness_pgd(ctx: "EvalContext") -> MetricResult:
    """M67 — forget accuracy under a PGD perturbation (``hp.adv``). Stronger than
    FGSM; ``eps=0`` recovers clean forget acc."""
    return _adv_metric(ctx, name="adv_robustness_pgd", perturb_kind="pgd")
