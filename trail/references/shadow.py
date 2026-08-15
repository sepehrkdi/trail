"""Shadow-ensemble builder for the opt-in LiRA tier (M9, L3 cache).

Trains ``hp.references.shadow`` fresh models and records, for every audit
example (the canonical forget set followed by the canonical test set), the
LiRA confidence signal phi under each shadow plus an IN/OUT membership mask.
The result is a :class:`ShadowStats` cached at L3 — method-INDEPENDENT (no
checkpoint hash in the key), so it is built once per (dataset, seed, ensemble
size, recipe, split identity) and amortized across every method evaluated on
the same data.

Training protocol (frozen ``hp.shadow`` recipe, fully in the L3 key):

* Each shadow trains on the FULL retain pool (always-in filler, gives the
  model a real classification task) plus an ``in_fraction`` random slice of the
  audit pool (forget+test). The complement of the audit pool is OUT for that
  shadow, so across the ensemble every audit example is IN of ~``in_fraction``
  of the shadows and OUT of the rest — the per-example IN/OUT coverage LiRA
  needs.
* Training and scoring both use the CANONICAL (augmentation-stripped) views, so
  the shadow phi is directly comparable to the target's ``ctx.outputs`` phi
  (which is also probed on the canonical loaders) and the test split — which
  has no train-time augmentation view — can be trained on uniformly.

Reproducibility (G2): every random draw is a named seeding substream. The one
deliberate global ``torch.manual_seed`` is the per-shadow weight init — fresh
``nn.Module`` construction reads the global generator, not a passed one — set
explicitly here from the substream value, not buried in a helper.

Methods are NOT trained here (scope firewall): shadow models are reference
artifacts, the same category as the gold retrained model.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

from trail.attacks.lira import confidence_logit
from trail.attacks.relearn import _logits_of, _split_identity
from trail.core.cache import l3_key
from trail.core.errors import MissingReference
from trail.core.hashing import config_hash
from trail.core.seeding import (
    make_worker_init_fn,
    numpy_rng,
    seed_for,
    torch_generator,
)
from trail.core.types import ShadowStats

if TYPE_CHECKING:  # pragma: no cover
    from trail.core.context import EvalContext

logger = logging.getLogger("trail.references.shadow")


def _recipe_dict(ctx: "EvalContext") -> dict[str, Any]:
    """The frozen shadow recipe, normalized for the cache key + provenance."""
    s = ctx.hp.shadow
    return {
        "epochs": int(s.epochs),
        "lr": float(s.lr),
        "momentum": float(s.momentum),
        "weight_decay": float(s.weight_decay),
        "batch_size": int(s.batch_size),
        "in_fraction": float(s.in_fraction),
    }


def _l3_blob_key(ctx: "EvalContext", n_shadow: int) -> str:
    """Content-address the ensemble: the reserved l3_key (dataset/seed/size)
    PLUS the recipe and the exact split identity, so a changed recipe or a
    different forget/retain/test partition can never serve a stale ensemble."""
    return config_hash({
        "l3": l3_key(ctx._dataset_id(), ctx.seed, n_shadow),
        "n_shadow": int(n_shadow),
        "recipe": _recipe_dict(ctx),
        "split_identity": _split_identity(ctx),
    })


def _audit_pool(ctx: "EvalContext"):
    """The audit pool as ``(concat_dataset, n_forget, n_test, audit_labels)``.

    Order is ``[canonical forget ; canonical test]`` — identical to the order
    the target model is scored in (``ctx.outputs("unlearned","forget")`` then
    ``"test"``), so ShadowStats columns align positionally with the target phi.
    ``audit_labels`` is 1 for forget (member-candidate), 0 for test.
    """
    forget_ds = ctx.loader("forget").dataset
    test_ds = ctx.loader("test").dataset
    n_forget, n_test = len(forget_ds), len(test_ds)  # type: ignore[arg-type]
    audit = ConcatDataset([forget_ds, test_ds])
    labels = np.concatenate([
        np.ones(n_forget, dtype=np.int64), np.zeros(n_test, dtype=np.int64)])
    return audit, n_forget, n_test, labels


def _train_one_shadow(ctx: "EvalContext", shadow_idx: int,
                      in_idx: np.ndarray, audit, retain_ds) -> "nn.Module":
    """Build + train one shadow on retain (always-in) + the IN audit slice."""
    device = ctx.device
    s = ctx.hp.shadow

    # The single deliberate global seed: fresh nn.Module init reads the global
    # generator. Routed through the named substream so the ensemble is
    # reproducible (G2); explicit here, not hidden in build_fresh_model.
    torch.manual_seed(seed_for(ctx.seed, f"shadow:{shadow_idx}:init"))
    if not hasattr(ctx.adapter, "build_fresh_model"):
        raise MissingReference(
            "not_applicable_modality",
            f"adapter {ctx.adapter.name!r} cannot build fresh models; the "
            "LiRA shadow tier requires a classification adapter")
    model = ctx.adapter.build_fresh_model(device)

    train_set = ConcatDataset([retain_ds, Subset(audit, in_idx.tolist())])
    loader = DataLoader(
        train_set, batch_size=min(s.batch_size, len(train_set)), shuffle=True,
        generator=torch_generator(ctx.seed, f"shadow:{shadow_idx}:loader"),
        num_workers=0,
        worker_init_fn=None)
    optimizer = torch.optim.SGD(model.parameters(), lr=s.lr, momentum=s.momentum,
                                weight_decay=s.weight_decay)
    criterion = nn.CrossEntropyLoss()
    model.train()
    n_epochs = s.epochs
    for epoch in range(n_epochs):
        run_loss, run_correct, run_total = 0.0, 0, 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = _logits_of(model(inputs))
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            bs = int(targets.size(0))
            run_loss += float(loss.detach()) * bs
            run_correct += int((logits.detach().argmax(1) == targets).sum())
            run_total += bs
        # Live curve (no-op when W&B is off): per-epoch loss/acc across all
        # shadows, monotonic global step so the curve is continuous.
        ctx.wandb_log({
            "shadow/idx": shadow_idx,
            "shadow/epoch": epoch + 1,
            "shadow/train_loss": run_loss / max(run_total, 1),
            "shadow/train_acc": 100.0 * run_correct / max(run_total, 1),
        }, step=shadow_idx * n_epochs + epoch)
    return model


def _shadow_phi(model: "nn.Module", audit, device, batch_size: int) -> np.ndarray:
    """Phi (logit-conf) of every audit example under one trained shadow,
    in canonical pool order."""
    model.eval()
    loader = DataLoader(audit, batch_size=batch_size, shuffle=False, num_workers=0)
    logit_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for inputs, targets in loader:
            logits = _logits_of(model(inputs.to(device)))
            logit_chunks.append(logits.detach().cpu().numpy())
            label_chunks.append(targets.numpy())
    logits = np.concatenate(logit_chunks)
    labels = np.concatenate(label_chunks)
    return confidence_logit(logits, labels)


def build_shadow_stats(ctx: "EvalContext") -> ShadowStats:
    """Build (or load from L3) the shadow ensemble for the LiRA tier.

    Returns a :class:`ShadowStats` over the audit pool. Raises
    :class:`MissingReference` (``reference_disabled:shadow``) when
    ``hp.references.shadow == 0`` — the metric records the skip and moves on.

    GPU-determinism caveat (disclosed via ``ctx.warnings``, as for the relearn
    attack): shadow training backprops with stock CUDA kernels, so the ensemble
    is bit-reproducible on CPU but only tolerance-reproducible on GPU.
    """
    n_shadow = int(ctx.hp.references.shadow)
    if n_shadow <= 0:
        raise MissingReference(
            "reference_disabled:shadow",
            "hp.references.shadow=0; set shadow=8 to build the LiRA ensemble")

    if ctx.device.type == "cuda":
        msg = ("LiRA shadow training: CUDA kernels are not determinism-pinned; "
               "M9 is reproducible only up to GPU kernel tolerance, not "
               "bit-wise (see references/shadow.py)")
        if msg not in ctx.warnings:
            ctx.warnings.append(msg)

    key = _l3_blob_key(ctx, n_shadow)
    cache = getattr(ctx, "cache", None)
    if cache is not None and hasattr(cache, "get_blob"):
        try:
            cached = cache.get_blob("L3", key)
        except Exception:  # a cache failure must never break the attack
            logger.warning("shadow L3 get_blob failed (key=%s)", key, exc_info=True)
            cached = None
        if isinstance(cached, ShadowStats):
            logger.info("shadow ensemble: L3 cache hit (key=%s, n=%d)",
                        key[:12], cached.n_shadow)
            return cached

    audit, n_forget, n_test, audit_labels = _audit_pool(ctx)
    n_audit = n_forget + n_test
    retain_ds = ctx.loader("retain").dataset
    in_fraction = float(ctx.hp.shadow.in_fraction)
    k_in = max(1, min(n_audit - 1, int(round(n_audit * in_fraction))))

    ctx.stamp("privacy.mia_lira.n_shadow", n_shadow)
    ctx.stamp("privacy.mia_lira.audit", {"n_forget": n_forget, "n_test": n_test})
    ctx.stamp("privacy.mia_lira.recipe", _recipe_dict(ctx))
    logger.info("building shadow ensemble: n_shadow=%d audit=%d (forget=%d "
                "test=%d) k_in=%d retain=%d epochs=%d", n_shadow, n_audit,
                n_forget, n_test, k_in, len(retain_ds),  # type: ignore[arg-type]
                ctx.hp.shadow.epochs)

    shadow_phi = np.empty((n_shadow, n_audit), dtype=np.float64)
    member_mask = np.zeros((n_shadow, n_audit), dtype=bool)
    for sdx in range(n_shadow):
        # Per-shadow IN slice of the audit pool (the rest is OUT). Independent
        # substream per shadow so adding shadows never perturbs earlier ones.
        perm = numpy_rng(ctx.seed, f"shadow:{sdx}:mask").permutation(n_audit)
        in_idx = perm[:k_in]
        member_mask[sdx, in_idx] = True
        model = _train_one_shadow(ctx, sdx, in_idx, audit, retain_ds)
        shadow_phi[sdx] = _shadow_phi(model, audit, ctx.device,
                                      ctx.hp.shadow.batch_size)
        logger.info("  shadow %d/%d trained (k_in=%d)", sdx + 1, n_shadow, k_in)

    stats = ShadowStats(shadow_phi=shadow_phi, member_mask=member_mask,
                        audit_labels=audit_labels, n_shadow=n_shadow,
                        cache_key=key)

    if cache is not None and hasattr(cache, "put_blob"):
        try:
            cache.put_blob("L3", key, stats)
        except Exception:
            logger.warning("shadow L3 put_blob failed (key=%s)", key, exc_info=True)
    return stats
