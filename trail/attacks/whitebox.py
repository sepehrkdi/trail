"""Shared white-box backward-pass scaffold (Tier-2 attack family).

A gradient path CANNOT reuse the adapter's ``forward_stats`` CUDA-OOM retry —
that pass runs under ``torch.inference_mode`` and never builds an autograd graph,
so its retry is unreachable here. This scaffold therefore carries its OWN
CUDA-OOM chunk-halving. It also:

* **deepcopies the model BEFORE taking gradients** — the ctx-memoized model is a
  shared instance and must never be mutated (model-mutation rule);
* defaults to the **last layer** (classifier head) parameters;
* leaves the GPU determinism caveat to the caller (``append_cuda_caveat``) —
  backward kernels are not determinism-pinned (tolerance, not bit-exact).
"""
from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch import nn

logger = logging.getLogger("trail.attacks.whitebox")

_CUDA_CAVEAT = ("white-box gradient attack: CUDA backward kernels are not "
                "determinism-pinned; gradient-MIA values are reproducible only "
                "to GPU kernel tolerance, not bit-wise (CPU stays deterministic)")


def append_cuda_caveat(ctx) -> None:
    """Record the GPU-tolerance caveat once on ctx.warnings (no-op on CPU)."""
    if getattr(ctx, "device", None) is not None and ctx.device.type == "cuda":
        if _CUDA_CAVEAT not in ctx.warnings:
            ctx.warnings.append(_CUDA_CAVEAT)


def last_linear(model: "nn.Module") -> "nn.Module | None":
    """The classifier head: ``model.fc`` if present, else the LAST ``nn.Linear``
    in module order (ViT/Swin/AllCNN differ, so never reach ``.fc`` directly).
    None when the model exposes no linear head."""
    from torch import nn

    fc = getattr(model, "fc", None)
    if isinstance(fc, nn.Linear):
        return fc
    last = None
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            last = mod
    return last


def _grad_params(model: "nn.Module", layer: str) -> "list[torch.Tensor]":
    """Parameters to differentiate w.r.t. ``layer`` ("last" = classifier head,
    else a named submodule). Falls back to all params if no head is found."""
    if layer == "last":
        head = last_linear(model)
        if head is not None:
            return [p for p in head.parameters() if p.requires_grad]
    else:
        sub = dict(model.named_modules()).get(layer)
        if sub is not None:
            return [p for p in sub.parameters() if p.requires_grad]
    return [p for p in model.parameters() if p.requires_grad]


def per_example_grad_norms(model: "nn.Module", X: "torch.Tensor", y: "torch.Tensor",
                           *, device: "torch.device", layer: str = "last",
                           chunk: int = 256) -> np.ndarray:
    """Per-example L2 norm of ∂CE/∂θ over the selected layer's params — the
    gradient-MIA signal (members tend to have SMALLER gradients).

    deepcopy-before-grad (the input model is never mutated); own CUDA-OOM
    chunk-halving (the inference-mode forward retry is unreachable here).
    Returns ``[N]`` float64. Deterministic on CPU; GPU is tolerance-only.
    """
    import torch
    from torch import nn

    work = copy.deepcopy(model).to(device)
    work.eval()
    params = _grad_params(work, layer)
    if not params:
        raise ValueError("per_example_grad_norms: model exposes no grad params")
    X = X.to(device).float()
    y = y.to(device)
    ce = nn.CrossEntropyLoss()
    n = len(X)
    norms = np.empty(n, dtype=np.float64)

    i, cur = 0, max(1, chunk)
    while i < n:
        end = min(i + cur, n)
        try:
            for j in range(i, end):
                grads = torch.autograd.grad(
                    ce(work(X[j:j + 1]), y[j:j + 1]), params)
                norms[j] = float(sum((g * g).sum().item() for g in grads)) ** 0.5
            i = end
        except torch.cuda.OutOfMemoryError:  # pragma: no cover - GPU only
            torch.cuda.empty_cache()
            if cur == 1:
                raise
            cur = max(1, cur // 2)
            logger.warning("per_example_grad_norms: CUDA OOM; halving chunk to %d", cur)
    return norms
