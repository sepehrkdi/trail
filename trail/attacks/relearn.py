"""Standardized relearning attack.

Fine-tunes a copy of a checkpointed model on attacker-held data and records
the *per-epoch* forget-recovery curve (curves, not endpoints —
the legacy implementation only recorded endpoints). Three attacker source
models:

- ``forget_only``  — attacker holds N deleted samples (M11; Pawelczyk et al. 2024)
- ``retain_mix``   — attacker holds public retain data plus N deleted samples (M12)
- ``retain_only``  — attacker holds only public data: the FULL retain pool,
  ignoring budgets and ``retain_fraction`` (benign-relearning instantiation
  of Hu et al. 2025). NOT identical to M12's ``n=0`` row — that row trains
  on the ``retain_fraction`` slice of the retain pool (legacy
  the original evaluation pipeline); the two coincide only at ``retain_fraction=1.0``.

Ported and adapted from the original evaluation pipeline (cited per function);
no imports from the original research pipeline.

All accuracies on the 0-100 scale. All randomness flows through
``trail.core.seeding`` named substreams.
"""
from __future__ import annotations

import copy
import json
import logging
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from trail.core.errors import RequestError, SplitNotAvailable
from trail.core.hashing import config_hash
from trail.core.seeding import make_worker_init_fn, numpy_rng, torch_generator

if TYPE_CHECKING:  # pragma: no cover
    from trail.core.context import EvalContext

logger = logging.getLogger("trail.attacks.relearn")

Budget = int | str  # non-negative int, or the sentinel "full"

SOURCES: tuple[str, ...] = ("forget_only", "retain_mix", "retain_only")


# ──────────────────────────────────────────────────────────────────────────
# Curve scalars
# ──────────────────────────────────────────────────────────────────────────

def ttr_epoch(history: Sequence[float], threshold: float = 80.0) -> int | None:
    """First 1-indexed epoch where forget accuracy crosses ``threshold``.

    Returns ``None`` if the curve never crosses. Ported from
    the original evaluation pipeline (``time_to_recovery_epoch``).
    """
    for i, acc in enumerate(history, start=1):
        if acc >= threshold:
            return i
    return None


def ttr_auc(history: Sequence[float], baseline: float = 0.0) -> float:
    """Trapezoidal AUC of ``max(0, acc - baseline)`` over the recovery curve.

    Higher = forget accuracy regrew faster or farther. Robust to threshold
    choice. Ported from the original evaluation pipeline
    (``time_to_recovery_auc``).
    """
    if not history:
        return 0.0
    shifted = [max(0.0, x - baseline) for x in history]
    if len(shifted) == 1:
        return shifted[0]
    total = 0.0
    for a, b in zip(shifted, shifted[1:]):
        total += (a + b) / 2.0
    return total


def anamnesis(ttr_u: float, ttr_g: float, eps: float = 1e-8) -> float:
    """Anamnesis index: unlearned-model recovery AUC over gold's.

    Both inputs are :func:`ttr_auc` values. AIN ≈ 1 — regrows like gold;
    AIN > 1 — regrows faster than gold (residual hidden knowledge); AIN < 1 —
    slower. ``eps`` guards the degenerate gold-never-recovered case. Ported
    from the original evaluation pipeline (``anamnesis_index``).
    """
    return float(ttr_u / max(ttr_g, eps))


# ──────────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────────

def _logits_of(out: Any) -> "torch.Tensor":
    """Unwrap (logits, aux...) tuples some classification backbones return."""
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _eval_accuracy(model: "nn.Module", loader: "DataLoader",
                   device: Any) -> tuple[float, np.ndarray]:
    """Plain argmax-accuracy inference loop (no probe features, no caching).

    Returns ``(accuracy_0_100, per_example_correct float32 array)``. Order of
    ``per_example_correct`` follows the loader's (canonical loaders are
    deterministically ordered).
    """
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            preds = _logits_of(model(inputs)).argmax(dim=1)
            chunks.append((preds == targets).to(torch.float32).cpu().numpy())
    correct = (np.concatenate(chunks) if chunks
               else np.zeros(0, dtype=np.float32))
    acc = float(correct.mean() * 100.0) if correct.size else float("nan")
    return acc, correct


def _subsample(pool: "Dataset", n: int, rng: np.random.Generator) -> "Dataset":
    """Deterministic without-replacement subsample: a PREFIX of one full
    permutation of the pool (``n`` clamped to the pool size).

    Ported from the original evaluation pipeline
    (``_sample_forget_subset``). Nesting note: the legacy
    ``RandomState.choice(n, size=k, replace=False)`` is a permutation prefix,
    so budget grids were NESTED (the n=10 attacker subset is a subset of the
    n=100 subset). numpy ``Generator.choice(replace=False)`` does NOT have
    that property, so we draw ONE full permutation per (role, source) from
    the named substream — the substream is re-derived identically for every
    budget cell — and slice ``perm[:n]`` prefixes per budget, restoring the
    legacy nesting (monotone attacker information across the M11/M12 grid)
    while keeping the substream discipline.
    """
    size = min(n, len(pool))  # type: ignore[arg-type]
    perm = rng.permutation(len(pool))  # type: ignore[arg-type]
    return Subset(pool, perm[:size].tolist())


def _build_attack_set(ctx: "EvalContext", role: str, source: str,
                      budget: Budget, retain_fraction: float) -> "Dataset | None":
    """Assemble the attacker's training set for one (source, budget) cell.

    Ported from the original evaluation pipeline (``_build_relearn_loader``)
    with splits supplied by the context (train-transform views) instead of
    being re-derived from a raw trainset. Returns ``None`` for the
    no-training baseline cell (``forget_only`` at budget 0).
    """
    forget_pool = ctx.train_view("forget")

    if source == "retain_only":
        # Benign relearning (Hu et al. 2025 proxy): the FULL public retain
        # pool, regardless of the nominal budget and of retain_fraction.
        # Deliberately NOT the same cell as retain_mix at n=0 (which applies
        # the retain_fraction slice, the original evaluation pipeline).
        return ctx.train_view("retain")

    # Forget part.
    if budget == "full":
        forget_part: Dataset | None = forget_pool
    else:
        n = int(budget)
        if n == 0:
            forget_part = None
        else:
            rng = numpy_rng(ctx.seed, f"relearn:{role}:forget_subset")
            forget_part = _subsample(forget_pool, n, rng)

    if source == "forget_only":
        return forget_part  # None at budget 0 -> pre-attack baseline row

    # retain_mix (the original evaluation pipeline): retain slice + N forget samples.
    retain_pool = ctx.train_view("retain")
    if 0.0 < retain_fraction < 1.0:
        rng = numpy_rng(ctx.seed, f"relearn:{role}:retain_mix")
        n_retain = max(1, int(retain_fraction * len(retain_pool)))  # type: ignore[arg-type]
        retain_part: Dataset = _subsample(retain_pool, n_retain, rng)
    else:
        retain_part = retain_pool
    if forget_part is None:
        # n=0 row of M12: the retain_fraction slice only — equals the
        # retain_only source iff retain_fraction=1.0.
        return retain_part
    return ConcatDataset([retain_part, forget_part])


def _resolve_eval_loaders(ctx: "EvalContext") -> tuple["DataLoader", str,
                                                       "DataLoader", str]:
    """Pick the forget-recovery and retain eval loaders, with disclosed fallback.

    Prefers held-out ``forget_test`` / ``retain_test``; falls back to the
    train-side ``forget`` / ``retain`` splits when a test split is empty by
    design in this forgetting mode.
    """
    try:
        forget_eval, forget_basis = ctx.loader("forget_test"), "forget_test"
    except SplitNotAvailable:
        forget_eval, forget_basis = ctx.loader("forget"), "forget"
    try:
        retain_eval, retain_basis = ctx.loader("retain_test"), "retain_test"
    except SplitNotAvailable:
        retain_eval, retain_basis = ctx.loader("retain"), "retain"
    return forget_eval, forget_basis, retain_eval, retain_basis


def _validate_budgets(budgets: Sequence[Budget]) -> None:
    for b in budgets:
        if b == "full":
            continue
        if isinstance(b, bool) or not isinstance(b, int) or b < 0:
            raise RequestError(
                f"relearn budget must be a non-negative int or 'full', got {b!r}")


# ──────────────────────────────────────────────────────────────────────────
# The attack
# ──────────────────────────────────────────────────────────────────────────

def run_relearn_attack(
    ctx: "EvalContext",
    role: str = "unlearned",
    *,
    budgets: Sequence[Budget],
    source: str,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    batch_size: int,
    retain_fraction: float = 1.0,
    num_workers: int = 0,
) -> dict[str, dict[str, Any]]:
    """Run the standardized relearning attack grid against one checkpoint role.

    Ported from the original evaluation pipeline (``run_relearn_attack``)
    with two deliberate changes: (1) per-epoch recovery evaluation — the spec
 build item; the source only recorded endpoints — and (2) pristine
    starting weights per budget via ``copy.deepcopy`` of the memoized context
    model instead of an in-place state-dict snapshot/restore (the context
    model must never be mutated).

    GPU-determinism caveat (disclosed via ``ctx.warnings``): the fine-tuning
    loop backprops with stock CUDA kernels (no
    ``torch.use_deterministic_algorithms`` pinning — that requires
    process-level setup such as ``CUBLAS_WORKSPACE_CONFIG`` before CUDA
    init, which a metric body cannot impose). All RNG is substream-routed,
    so M11/M12 are bit-deterministic on CPU but only
    tolerance-reproducible on GPU (G1 holds up to small kernel
    nondeterminism for these metrics).

    Args:
        ctx: evaluation context (model access, split views, seeds, stamps).
        role: checkpoint role to attack ("unlearned", "original", "gold").
        budgets: grid of forget-sample budgets — non-negative ints and/or
            "full". Budget 0 with ``source="forget_only"`` is the no-training
            pre-attack baseline row.
        source: one of ``forget_only`` / ``retain_mix`` / ``retain_only``.
        epochs: attack fine-tuning epochs (frozen-protocol constant).
        lr, momentum, weight_decay, batch_size: frozen SGD hyperparameters.
        retain_fraction: fraction of the retain pool mixed in for
            ``retain_mix`` (1.0 = the whole pool).
        num_workers: DataLoader workers (seeded via ``make_worker_init_fn``).

    Returns:
        ``{str(budget): {"history": list[float] per-epoch forget recovery
        accuracy (0-100), "post": float final forget recovery accuracy,
        "retain_post": float final retain accuracy, "post_correct":
        np.ndarray per-example correctness (float32 0/1) of the final
        forget-recovery eval}}``.
    """
    if source not in SOURCES:
        raise RequestError(f"unknown relearn source {source!r}; one of {SOURCES}")
    _validate_budgets(budgets)

    # Memoize on the context: M60 (efficacy_vs_compute) "reuses M11 grid" —
    # identical (role, source, recipe) must return the same attack results,
    # not an independent run that diverges under GPU nondeterminism.
    cache = getattr(ctx, "_relearn_cache", None)
    if cache is not None:
        memo_key = config_hash({
            "role": role, "source": source,
            "budgets": [str(b) for b in budgets],
            "epochs": int(epochs), "lr": float(lr),
            "momentum": float(momentum),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "retain_fraction": float(retain_fraction),
        })
        cached = cache.get(memo_key)
        if cached is not None:
            logger.info("relearn[%s/%s] served from in-memory cache", role, source)
            return cached

    device = ctx.device
    if device.type == "cuda":
        msg = ("relearn attack: CUDA training kernels are not "
               "determinism-pinned; M11/M12 values are reproducible only up "
               "to GPU kernel tolerance, not bit-wise (see run_relearn_attack "
               "docstring)")
        if msg not in ctx.warnings:
            ctx.warnings.append(msg)
    base_model = ctx.model(role)  # memoized — never mutated; deepcopy per budget
    forget_eval, forget_basis, retain_eval, retain_basis = _resolve_eval_loaders(ctx)
    ctx.stamp(f"relearning.attack.{role}.{source}.forget_eval_basis", forget_basis)
    ctx.stamp(f"relearning.attack.{role}.{source}.retain_eval_basis", retain_basis)

    criterion = nn.CrossEntropyLoss()
    results: dict[str, dict[str, Any]] = {}

    for budget in budgets:
        label = str(budget)
        # Snapshot semantics of the original evaluation pipeline: every budget starts
        # from the pristine checkpoint.
        model = copy.deepcopy(base_model).to(device)

        attack_set = _build_attack_set(ctx, role, source, budget, retain_fraction)

        if attack_set is None:
            # forget_only @ budget 0: pre-attack baseline, no training.
            pre_acc, pre_correct = _eval_accuracy(model, forget_eval, device)
            retain_post, _ = _eval_accuracy(model, retain_eval, device)
            results[label] = {
                "history": [pre_acc],
                "post": pre_acc,
                "retain_post": retain_post,
                "post_correct": pre_correct,
            }
            logger.info("relearn[%s/%s] budget=%s pre-attack baseline: "
                        "forget=%.2f retain=%.2f", role, source, label,
                        pre_acc, retain_post)
            continue

        loader = DataLoader(
            attack_set,
            batch_size=min(batch_size, len(attack_set)),  # type: ignore[arg-type]
            shuffle=True,
            generator=torch_generator(ctx.seed, f"relearn:{role}:loader"),
            num_workers=num_workers,
            worker_init_fn=(make_worker_init_fn(ctx.seed, f"relearn:{role}:workers")
                            if num_workers > 0 else None),
        )
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum,
            weight_decay=weight_decay,
        )

        history: list[float] = []
        best_acc = float("-inf")
        best_correct = np.zeros(0, dtype=np.float32)
        for _ in range(epochs):
            model.train()
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = criterion(_logits_of(model(inputs)), targets)
                loss.backward()
                optimizer.step()
            # Per-epoch recovery eval (curves, not endpoints).
            acc, correct = _eval_accuracy(model, forget_eval, device)
            history.append(acc)
            if acc > best_acc:
                best_acc = acc
                best_correct = correct

        retain_post, _ = _eval_accuracy(model, retain_eval, device)
        results[label] = {
            "history": history,
            "post": best_acc if history else float("nan"),
            "retain_post": retain_post,
            "post_correct": best_correct,
        }
        logger.info("relearn[%s/%s] budget=%s post: forget=%.2f retain=%.2f",
                    role, source, label, results[label]["post"], retain_post)

    if cache is not None:
        cache[memo_key] = results
    return results


# ──────────────────────────────────────────────────────────────────────────
# D2D relearning (1%/9% disjoint split; Fan et al. 2025 NPO-SAM, M13)
# ──────────────────────────────────────────────────────────────────────────

def run_d2d_relearn_attack(
    ctx: "EvalContext",
    role: str = "unlearned",
    *,
    d2d_relearn_fraction: float,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    batch_size: int,
    num_workers: int = 0,
) -> dict[str, Any]:
    """D2D sharpness-aware relearning: fine-tune on a small disjoint slice of
    the forget set, measure recovery on the held-out remainder.

    Reproduces the legacy 1%/9% protocol (the original evaluation pipeline,
    ``_split_forget_for_d2d_attack``; cited, not imported): a deterministic
    permutation of the forget pool splits it into a ``d2d_relearn_fraction``
    attack subset (the "1%" when the forget set is 10% of a class) and the
    disjoint remainder eval subset (the "9%"). The model is fine-tuned on the
    *augmented* attack subset (``train_view("forget")``); forget recovery is
    evaluated per-epoch on the *canonical* (aug-stripped) eval subset, which is
    index-aligned with the train view. Disjointness makes the recovery signal
    generalization from the attack slice, not memorization of it. Mode-agnostic
    over the forget split (the canonical D2D citation is the class-wise /
    ``sub_class_atypical`` setting where forget ≈ 10% of a class).

    The split seed is the named substream ``relearn:{role}:d2d_split`` (G2), NOT
    the legacy standalone ``RandomState``.

    Returns:
        ``{"history": list[float] per-epoch recovery accuracy on the eval slice,
        "post": float best-epoch recovery accuracy (rational attacker),
        "pre": float pre-attack accuracy on the eval slice,
        "retain_post": float final retain accuracy,
        "post_correct": np.ndarray per-example correctness of the best epoch,
        "n_relearn": int, "n_eval": int}``.
    """
    device = ctx.device
    if device.type == "cuda":
        msg = ("relearn attack: CUDA training kernels are not "
               "determinism-pinned; M11/M12/M13 values are reproducible only "
               "up to GPU kernel tolerance, not bit-wise (see "
               "run_relearn_attack docstring)")
        if msg not in ctx.warnings:
            ctx.warnings.append(msg)

    forget_train = ctx.train_view("forget")            # augmented (fine-tune)
    canonical_forget = ctx.loader("forget").dataset    # aug-stripped, aligned
    n = len(forget_train)  # type: ignore[arg-type]
    if n < 2:
        raise RequestError(
            f"D2D relearning needs >=2 forget samples, got {n}")
    perm = numpy_rng(ctx.seed, f"relearn:{role}:d2d_split").permutation(n)
    k = max(1, min(n - 1, int(round(n * d2d_relearn_fraction))))
    relearn_idx, eval_idx = perm[:k], perm[k:]
    relearn_set = Subset(forget_train, relearn_idx.tolist())
    eval_set = Subset(canonical_forget, eval_idx.tolist())
    ctx.stamp(f"relearning.d2d.{role}.n_relearn", int(k))
    ctx.stamp(f"relearning.d2d.{role}.n_eval", int(len(eval_idx)))

    eval_loader = DataLoader(
        eval_set, batch_size=min(batch_size, len(eval_set)), shuffle=False,
        num_workers=num_workers,
        worker_init_fn=(make_worker_init_fn(ctx.seed, f"relearn:{role}:d2d_eval")
                        if num_workers > 0 else None))
    _, _, retain_eval, retain_basis = _resolve_eval_loaders(ctx)
    ctx.stamp(f"relearning.d2d.{role}.retain_eval_basis", retain_basis)

    model = copy.deepcopy(ctx.model(role)).to(device)
    pre_acc, _ = _eval_accuracy(model, eval_loader, device)

    loader = DataLoader(
        relearn_set, batch_size=min(batch_size, len(relearn_set)), shuffle=True,
        generator=torch_generator(ctx.seed, f"relearn:{role}:d2d_loader"),
        num_workers=num_workers,
        worker_init_fn=(make_worker_init_fn(ctx.seed, f"relearn:{role}:d2d_workers")
                        if num_workers > 0 else None))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history: list[float] = []
    best_acc = float("-inf")
    best_correct = np.zeros(0, dtype=np.float32)
    for _ in range(epochs):
        model.train()
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(_logits_of(model(inputs)), targets)
            loss.backward()
            optimizer.step()
        acc, correct = _eval_accuracy(model, eval_loader, device)
        history.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_correct = correct

    retain_post, _ = _eval_accuracy(model, retain_eval, device)
    post = best_acc if history else float("nan")
    logger.info("relearn_d2d[%s] 1pct/9pct (n_relearn=%d n_eval=%d): "
                "pre=%.2f post=%.2f retain=%.2f", role, k, len(eval_idx),
                pre_acc, post, retain_post)
    return {
        "history": history,
        "post": post,
        "pre": pre_acc,
        "retain_post": retain_post,
        "post_correct": best_correct,
        "n_relearn": int(k),
        "n_eval": int(len(eval_idx)),
    }


# ──────────────────────────────────────────────────────────────────────────
# Gold recovery curve (L2-cached: method-independent)
# ──────────────────────────────────────────────────────────────────────────

def _serialize_results(results: dict[str, dict[str, Any]]) -> bytes:
    payload = {
        label: {
            "history": [float(x) for x in r["history"]],
            "post": float(r["post"]),
            "retain_post": float(r["retain_post"]),
            "post_correct": np.asarray(r["post_correct"], dtype=np.float32).tolist(),
        }
        for label, r in results.items()
    }
    return json.dumps(payload, sort_keys=True).encode()


def _deserialize_results(blob: bytes) -> dict[str, dict[str, Any]]:
    payload = json.loads(blob.decode())
    for r in payload.values():
        r["post_correct"] = np.asarray(r["post_correct"], dtype=np.float32)
    return payload


def _split_identity(ctx: "EvalContext") -> dict[str, Any]:
    """Dataset/split identity for the gold-curve L2 cache key.

    The gold recovery curve depends on WHICH forget/retain split the gold
    model is attacked and evaluated on, not just on the gold checkpoint: the
    same checkpoint evaluated under a different DatasetSpec (forget_class,
    split_seed, forget_fraction, ...) must MISS the cache. Spec-resolved data
    contributes the full spec params (they determine the splits exactly);
    raw user bundles contribute dataset_id plus the forget/retain split
    fingerprints.
    """
    data = ctx.request.data
    if hasattr(data, "model_dump"):  # DatasetSpec — params determine splits
        return {"mode": ctx.mode, "spec": data.model_dump()}
    from trail.data.fingerprint import split_fingerprint
    dataset_id = getattr(data, "dataset_id", None) or "user_bundle"
    fps: dict[str, str] = {}
    for split in ("forget", "retain"):
        fp, _warn = split_fingerprint(dataset_id, ctx.loader(split))
        fps[split] = fp
    return {"mode": ctx.mode, "dataset_id": dataset_id, "split_fps": fps}


def gold_relearn_curve(
    ctx: "EvalContext",
    *,
    budgets: Sequence[Budget],
    source: str,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    batch_size: int,
    retain_fraction: float = 1.0,
) -> dict[str, dict[str, Any]]:
    """Run (or fetch) the relearning attack against the gold reference model.

    Gold recovery curves are method-independent, so they are cached at the L2
    layer keyed on (gold checkpoint sha, split identity, resolved eval-basis
    splits, attack kwargs, seed) and amortized across every method evaluated
    against the same gold on the same splits. Split
    identity (dataset/mode/spec params or forget/retain fingerprints) and the
    eval bases participate in the key so that re-evaluating the same gold
    checkpoint under a different DatasetSpec can never serve a stale curve.

    Raises:
        MissingReference: propagated from ``ctx.gold()`` / ``ctx.model("gold")``
            when the gold reference is disabled or absent (callers omit the
            anamnesis component).
    """
    gold = ctx.gold()  # raises MissingReference with the right skip code
    _, forget_basis, _, retain_basis = _resolve_eval_loaders(ctx)
    kw: dict[str, Any] = {
        "budgets": [str(b) for b in budgets],
        "source": source,
        "epochs": int(epochs),
        "lr": float(lr),
        "momentum": float(momentum),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "retain_fraction": float(retain_fraction),
    }
    key = config_hash({
        "gold_sha": gold.sha256,
        "kw": kw,
        "seed": ctx.seed,
        "split_identity": _split_identity(ctx),
        "eval_basis": {"forget": forget_basis, "retain": retain_basis},
    })

    cache = getattr(ctx, "cache", None)
    if cache is not None and hasattr(cache, "get_blob"):
        try:
            blob = cache.get_blob("L2", key)
        except Exception:  # cache failure must never break the attack
            logger.warning("gold_relearn_curve: L2 get_blob failed for key=%s",
                           key, exc_info=True)
            blob = None
        if blob is not None:
            logger.info("gold_relearn_curve: L2 cache hit (key=%s)", key)
            return _deserialize_results(blob)
    else:
        # TODO: EvalContext exposes no blob-cache handle
        # yet; computing uncached. Wire to ctx.cache.{get,put}_blob("L2", ...)
        # once the cache surface lands.
        logger.debug("gold_relearn_curve: no ctx.cache blob layer; "
                     "computing uncached (key=%s)", key)

    results = run_relearn_attack(
        ctx, role="gold", budgets=budgets, source=source, epochs=epochs,
        lr=lr, momentum=momentum, weight_decay=weight_decay,
        batch_size=batch_size, retain_fraction=retain_fraction,
    )

    if cache is not None and hasattr(cache, "put_blob"):
        try:
            cache.put_blob("L2", key, _serialize_results(results))
        except Exception:
            logger.warning("gold_relearn_curve: L2 put_blob failed for key=%s",
                           key, exc_info=True)
    return results
