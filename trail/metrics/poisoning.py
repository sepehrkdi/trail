"""Poisoning / backdoor attack metrics (Tier-2) — ``external=True``, off the
default panel.

SCOPE FIREWALL: the repo never trains, and backdoor/clean-label poisoning need
a checkpoint poisoned at train time. So ``backdoor_trigger_asr`` (M56) and
``witchbrew_asr`` (M57) are **blocked on externally-produced poisoned
checkpoints + trigger/poison artifacts**: they consume a sha256-stamped INPUT
artifact (``allow_pickle=False``) and ``missing_artifact``-skip when it is
absent — which it always is without external data. Their ASR logic is unit-
tested on synthetic tensors and **marked unvalidated** until real poisoned
checkpoints arrive. ``gus_gaussian`` (M55) is computable (Gaussian input
robustness of the forgetting) and runs without external data.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from trail.core.errors import MetricError, MetricSkip
from trail.core.registry import register_metric
from trail.core.report import MetricResult
from trail.core.seeding import numpy_rng

if TYPE_CHECKING:  # pragma: no cover
    from trail.core.context import EvalContext

logger = logging.getLogger("trail.metrics.poisoning")

_CLS: set[str] = {"classification"}
_AD_MAX_SAMPLES = 4096


# ----------------------------------------------------- synthetic-testable helpers


def attack_success_rate(preds: np.ndarray, target_class: int) -> float:
    """Fraction of predictions equal to the attacker's target class (0-100)."""
    preds = np.asarray(preds)
    if preds.size == 0:
        return float("nan")
    return float(100.0 * np.mean(preds == int(target_class)))


def _artifact_path(ctx: "EvalContext", name: str) -> str | None:
    """Per-metric input-artifact path from ``hp.metric_overrides[name]['artifact']``."""
    return (ctx.hp.metric_overrides.get(name, {}) or {}).get("artifact")


def _load_trigger(ctx: "EvalContext", name: str) -> np.ndarray:
    """Load + sha-stamp the trigger/poison artifact, or ``missing_artifact``-skip."""
    from trail.core.artifacts import load_input_artifact

    path = _artifact_path(ctx, name)
    if not path:
        raise MetricSkip(
            "missing_artifact",
            f"{name} needs an externally-produced poison artifact; set "
            f"hp.metric_overrides[{name!r}]['artifact'] to a sha-stamped .npy/.npz")
    try:
        arr, sha = load_input_artifact(path)
    except FileNotFoundError:
        raise MetricSkip("missing_artifact", f"{name}: artifact not found at {path}")
    ctx.stamp(f"poisoning.{name}.artifact_sha256", sha)
    return arr


# ------------------------------------------------------------------- metrics


@register_metric(name="gus_gaussian", table_id="M55", category="poisoning",
                 modalities=_CLS, input_modes={"model"}, external=True,
                 cost="moderate")
def gus_gaussian(ctx: "EvalContext") -> MetricResult:
    """Gaussian Unlearning Score (M55): fraction of forget predictions that FLIP
    under seeded Gaussian input noise — instability of the forgetting boundary.
    Computable without external data (PROVISIONAL definition)."""
    import torch

    loader = ctx.loader("forget")
    model = ctx.model("unlearned")
    rng = numpy_rng(ctx.seed, "gus_gaussian:noise")
    flips, total = 0, 0
    with torch.inference_mode():
        for xb, _ in loader:
            xb = xb.to(ctx.device).float()
            base = model(xb).argmax(1)
            noise = torch.from_numpy(
                rng.standard_normal(tuple(xb.shape)).astype("float32")).to(ctx.device)
            pert = model(xb + ctx.hp.poison.trigger_value * 0.1 * noise).argmax(1)
            flips += int((base != pert).sum().item())
            total += len(xb)
            if total >= _AD_MAX_SAMPLES:
                break
    if total == 0:
        raise MetricError("gus_gaussian: empty forget split")
    value = 100.0 * flips / total
    return MetricResult(value=value, ci=(value, value), n=total)


@register_metric(name="backdoor_trigger_asr", table_id="M56", category="poisoning",
                 modalities=_CLS, input_modes={"model"}, external=True,
                 cost="moderate")
def backdoor_trigger_asr(ctx: "EvalContext") -> MetricResult:
    """Backdoor-trigger attack success rate (M56): apply the loaded trigger to
    the forget inputs and measure the rate of target-class predictions.
    UNVALIDATED — blocked on an externally-produced backdoored checkpoint +
    trigger artifact; ``missing_artifact``-skips without one."""
    import torch

    trigger = _load_trigger(ctx, "backdoor_trigger_asr")
    trig = torch.from_numpy(np.asarray(trigger, dtype="float32"))
    model = ctx.model("unlearned")
    preds = []
    with torch.inference_mode():
        for xb, _ in ctx.loader("forget"):
            xb = xb.to(ctx.device).float()
            t = trig.to(ctx.device)
            preds.append(model(xb + t).argmax(1).cpu().numpy())
            if sum(len(p) for p in preds) >= _AD_MAX_SAMPLES:
                break
    asr = attack_success_rate(np.concatenate(preds), ctx.hp.poison.target_class)
    return MetricResult(value=asr, ci=(asr, asr), n=int(sum(len(p) for p in preds)))


@register_metric(name="witchbrew_asr", table_id="M57", category="poisoning",
                 modalities=_CLS, input_modes={"model"}, external=True,
                 cost="moderate")
def witchbrew_asr(ctx: "EvalContext") -> MetricResult:
    """Witches'-Brew clean-label poisoning ASR (M57): rate at which the poisoned
    target sample(s) (loaded artifact) are predicted as the attacker class.
    UNVALIDATED — blocked on externally-produced poisoned data; skips without it."""
    import torch

    poison_x = _load_trigger(ctx, "witchbrew_asr")
    px = torch.from_numpy(np.asarray(poison_x, dtype="float32"))
    if px.ndim == 3:
        px = px.unsqueeze(0)
    model = ctx.model("unlearned")
    with torch.inference_mode():
        preds = model(px.to(ctx.device)).argmax(1).cpu().numpy()
    asr = attack_success_rate(preds, ctx.hp.poison.target_class)
    return MetricResult(value=asr, ci=(asr, asr), n=len(preds))
