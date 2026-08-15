"""Adversarial perturbations (Tier-3 robustness) — FGSM + PGD under
``torch.enable_grad`` (the forward_stats OOM retry runs under inference_mode and
is unreachable here, so this has its OWN CUDA-OOM chunk-halving). deepcopy-
before-grad (the ctx model is never mutated). ``eps == 0`` is the identity
perturbation (returns the inputs unchanged) — the cheapest invariant to test.
GPU backward kernels are tolerance-only (caller appends the caveat).
"""
from __future__ import annotations

import copy
import logging

import numpy as np

logger = logging.getLogger("trail.attacks.adversarial")

_CUDA_CAVEAT = ("adversarial (FGSM/PGD) attack: CUDA backward kernels are not "
                "determinism-pinned; robustness values are GPU-tolerance-only, "
                "not bit-wise (CPU stays deterministic)")


def append_cuda_caveat(ctx) -> None:
    """Record the GPU-tolerance caveat once on ctx.warnings (no-op on CPU)."""
    if getattr(ctx, "device", None) is not None and ctx.device.type == "cuda":
        if _CUDA_CAVEAT not in ctx.warnings:
            ctx.warnings.append(_CUDA_CAVEAT)


def _ce_grad_sign(model, x, y):
    import torch
    from torch import nn

    x = x.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        loss = nn.CrossEntropyLoss()(model(x), y)
        grad = torch.autograd.grad(loss, x)[0]
    return grad.sign().detach()


def fgsm_perturb(model, x, y, *, eps, device):
    """One-step FGSM: ``x + eps·sign(∂CE/∂x)``, clamped to [0,1]-free (operates in
    the model's input space). ``eps == 0`` returns ``x`` unchanged."""
    import torch

    if eps == 0:
        return x.clone().detach()
    work = copy.deepcopy(model).to(device).eval()
    x = x.to(device).float()
    y = y.to(device)
    return (x + float(eps) * _ce_grad_sign(work, x, y)).detach()


def pgd_perturb(model, x, y, *, eps, alpha, steps, device):
    """Iterative PGD projected to the ``eps`` L-inf ball. ``eps == 0`` returns
    ``x`` unchanged; ``steps`` FGSM-sized ``alpha`` updates otherwise."""
    import torch

    if eps == 0:
        return x.clone().detach()
    work = copy.deepcopy(model).to(device).eval()
    x0 = x.to(device).float()
    y = y.to(device)
    adv = x0.clone().detach()
    for _ in range(max(1, int(steps))):
        adv = adv + float(alpha) * _ce_grad_sign(work, adv, y)
        adv = torch.clamp(adv - x0, min=-float(eps), max=float(eps)) + x0  # project
        adv = adv.detach()
    return adv


def adversarial_accuracy(adapter, model, x, y, *, device, perturb) -> float:
    """Accuracy (0-100) of ``model`` on inputs perturbed by ``perturb(x, y)``,
    with its OWN CUDA-OOM chunk-halving (the grad path can't reuse the
    inference-mode forward retry)."""
    import torch

    x = x.to(device)
    y = y.to(device)
    n = len(x)
    correct = 0
    i, cur = 0, max(1, min(256, n))
    while i < n:
        end = min(i + cur, n)
        try:
            xa = perturb(x[i:end], y[i:end])
            with torch.inference_mode():
                preds = model(xa).argmax(1)
            correct += int((preds == y[i:end]).sum().item())
            i = end
        except torch.cuda.OutOfMemoryError:  # pragma: no cover - GPU only
            torch.cuda.empty_cache()
            if cur == 1:
                raise
            cur = max(1, cur // 2)
            logger.warning("adversarial_accuracy: CUDA OOM; halving chunk to %d", cur)
    return float(100.0 * correct / n) if n else float("nan")
