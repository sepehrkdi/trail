"""Checkpoint-ensemble margin builder for the KLoM metric (M68, L3 cache).

KLoM (Rinberg et al. 2026, "Data-Unlearn-Bench: Making Evaluating Data
Unlearning Easy", arXiv 2602.16400) compares the per-example margin DISTRIBUTION
of an ensemble of oracle (retrained-from-scratch) models against an ensemble of
unlearned models. This module builds those two ensembles' margins: for each
checkpoint in an ensemble, probe it over the canonical forget/retain/test
loaders and record the KLoM logit-margin per example. The result is an
:class:`EnsembleMargins` cached at L3.

The GOLD ensemble is method-INDEPENDENT — the L3 key is content-addressed by the
sorted gold checkpoint SHAs (no method/unlearned hash), so it is built once and
amortized across every method evaluated on the same data, exactly like the LiRA
shadow ensemble (references/shadow.py). The UNLEARNED ensemble is method-specific
(keyed by its own checkpoint SHAs).

Methods are NOT trained here (scope firewall): ensemble members are external
checkpoints supplied as reference artifacts via ``hp.references.gold_ensemble`` /
``hp.references.unlearned_ensemble``.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from trail.attacks.relearn import _split_identity
from trail.core.cache import l3_key
from trail.core.errors import MetricError, MissingReference, SplitNotAvailable
from trail.core.hashing import config_hash, sha256_file
from trail.core.types import EnsembleMargins

if TYPE_CHECKING:  # pragma: no cover
    from trail.core.context import EvalContext

logger = logging.getLogger("trail.references.gold_ensemble")

#: hp.references field holding each ensemble's checkpoint-path list.
_KIND_FIELD = {"gold": "gold_ensemble", "unlearned": "unlearned_ensemble"}


def klom_margin(logits: np.ndarray, labels: np.ndarray, *,
                clip: float = 100.0) -> np.ndarray:
    """KLoM per-example logit margin (Data-Unlearn-Bench, arXiv 2602.16400):

        phi(x) = f_y(x) - log sum_{k != y} exp(f_k(x)),

    clipped to ``[-clip, clip]``. Algebraically this equals ``logit(p_true)``
    (= ``attacks.lira.confidence_logit``); the difference is the clipping
    domain — KLoM clips the MARGIN to +/-100 (the published histogram range),
    whereas ``confidence_logit`` clips the PROBABILITY to [1e-6, 1-1e-6] (a
    tighter +/-13.8 margin bound). The +/-100 form is used here to stay faithful
    to the metric's histogram spec.

    Args:
        logits: ``[N, C]`` raw model logits.
        labels: ``[N]`` integer true-class indices.
        clip: symmetric clip magnitude on the margin.

    Returns:
        ``[N]`` float64 clipped margins.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64).ravel()
    n, c = logits.shape
    rows = np.arange(n)
    true = logits[rows, labels]
    if c == 1:  # degenerate single-class head: no "other" classes
        return np.clip(true, -clip, clip)
    masked = logits.copy()
    masked[rows, labels] = -np.inf                 # exclude the true class
    m = masked.max(axis=1)                          # stable log-sum-exp
    lse_others = m + np.log(np.exp(masked - m[:, None]).sum(axis=1))
    return np.clip(true - lse_others, -clip, clip)


def _ensemble_paths(ctx: "EvalContext", kind: str) -> list[str]:
    return list(getattr(ctx.hp.references, _KIND_FIELD[kind]))


def _l3_blob_key(ctx: "EvalContext", kind: str, paths: list[str],
                 splits: list[str], clip: float) -> str:
    """Content-address the ensemble margins: dataset/seed/size PLUS the sorted
    member SHAs (content, not paths — identical golds across methods share the
    entry), the scored splits, the margin clip, and the exact split identity so
    a different forget/retain/test partition can never serve a stale ensemble."""
    shas = sorted(sha256_file(p) for p in paths)
    return config_hash({
        "l3": l3_key(ctx._dataset_id(), ctx.seed, len(paths)),
        "purpose": "klom_ensemble",
        "kind": kind,
        "ckpt_shas": shas,
        "splits": sorted(splits),
        "margin_clip": float(clip),
        "split_identity": _split_identity(ctx),
    })


def build_ensemble_margins(ctx: "EvalContext", kind: str) -> EnsembleMargins:
    """Build (or load from L3) the KLoM margin ensemble for ``kind``.

    Raises :class:`MissingReference` (``reference_disabled:gold``) when the
    ``hp.references.<kind>_ensemble`` list is empty — the metric records the
    skip. ``not_applicable_mode`` when the forget split is unavailable.

    GPU-determinism caveat (as for the shadow ensemble and the relearn attack):
    the probes run stock CUDA kernels, so margins are bit-reproducible on CPU
    but only tolerance-reproducible on GPU — surfaced via ``ctx.warnings``.
    """
    if kind not in _KIND_FIELD:
        raise ValueError(f"unknown ensemble kind {kind!r}")
    if ctx.input_mode != "model":
        raise MissingReference(
            "requires_model_in",
            "KLoM ensembles load and probe checkpoints; supply model-in inputs")

    paths = _ensemble_paths(ctx, kind)
    if not paths:
        field = _KIND_FIELD[kind]
        raise MissingReference(
            "reference_disabled:gold",
            f"hp.references.{field} is empty; supply BOTH the gold and "
            "unlearned checkpoint ensembles to run KLoM (M68)")

    clip = float(ctx.hp.klom.clip_high)  # symmetric clip magnitude
    splits = list(ctx.hp.klom.splits)
    key = _l3_blob_key(ctx, kind, paths, splits, clip)

    cache = getattr(ctx, "cache", None)
    if cache is not None and hasattr(cache, "get_blob"):
        try:
            cached = cache.get_blob("L3", key)
        except Exception:  # a cache failure must never break the metric
            logger.warning("klom %s ensemble L3 get_blob failed (key=%s)",
                           kind, key, exc_info=True)
            cached = None
        if isinstance(cached, EnsembleMargins):
            logger.info("klom %s ensemble: L3 cache hit (key=%s, n=%d)",
                        kind, key[:12], cached.n_models)
            return cached

    if ctx.device.type == "cuda":
        msg = ("KLoM ensemble probing: CUDA kernels are not determinism-pinned; "
               "M68 margins are reproducible only up to GPU kernel tolerance "
               "(see references/gold_ensemble.py)")
        if msg not in ctx.warnings:
            ctx.warnings.append(msg)

    # Materialize the available splits (canonical view + fingerprint). A split
    # that is empty by design in this mode is skipped; forget is required.
    available: list[str] = []
    for split in splits:
        try:
            ctx.loader(split)
        except SplitNotAvailable:
            logger.info("klom %s ensemble: split %r unavailable; skipping",
                        kind, split)
            continue
        available.append(split)
    if "forget" not in available:
        raise MissingReference(
            "not_applicable_mode",
            "KLoM requires the forget split, absent in this mode")

    logger.info("building klom %s ensemble: n=%d splits=%s", kind,
                len(paths), available)
    per_split: dict[str, list[np.ndarray]] = {s: [] for s in available}
    for idx, path in enumerate(paths):
        model = ctx.adapter.load_checkpoint(path, ctx.device)
        for split in available:
            out = ctx.probe_model(model, split,
                                  seed_name=f"klom:{kind}:{idx}:{split}")
            if out.logits is None:
                raise MetricError(
                    f"klom {kind} ensemble: split {split!r} has no logits "
                    "(non-classification payload)")
            per_split[split].append(
                klom_margin(out.logits, out.targets, clip=clip))
        del model  # release before the next member (GPU memory hygiene)
        logger.info("  klom %s member %d/%d probed", kind, idx + 1, len(paths))

    margins = {s: np.stack(per_split[s], axis=0) for s in available}  # [n, N]
    stats = EnsembleMargins(kind=kind, margins=margins, n_models=len(paths),
                            cache_key=key)

    if cache is not None and hasattr(cache, "put_blob"):
        try:
            cache.put_blob("L3", key, stats)
        except Exception:
            logger.warning("klom %s ensemble L3 put_blob failed (key=%s)",
                           kind, key, exc_info=True)
    return stats
