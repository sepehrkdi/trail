"""Structural metrics: the collapse validity guard and the
representation-space distance to a reference model.

Mask-overlap metrics (M22-M24 proper) require sparse checkpoints and land in
a later milestone; v0.1 ships the guard plus the activation-distance adjunct.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from trail.core.bootstrap import bootstrap_ci
from trail.core.errors import MetricError, MetricSkip, SplitNotAvailable
from trail.core.registry import register_metric
from trail.core.report import MetricResult
from trail.core.seeding import numpy_rng

if TYPE_CHECKING:  # pragma: no cover
    from trail.core.context import EvalContext

logger = logging.getLogger("trail.metrics.structural")

_CLS: set[str] = {"classification"}

#: Cap on examples used by activation_distance (cost control; the canonical
#: eval view is deterministically ordered, so first-N is reproducible).
_AD_MAX_SAMPLES = 4096


def _offclass_stats(preds: np.ndarray, forget_class: int,
                    num_classes: int) -> tuple[float, float]:
    """(max_offclass_frac, normalized pred_entropy) of forget-sample
    predictions over the NON-forget classes.

    Port of the original evaluation pipeline (``interclass_confusion``)
    body: per-class prediction counts, forget-class bin zeroed, max fraction
    and normalized Shannon entropy of the off-class distribution. The
    degenerate all-forget-predictions case returns (0.0, 1.0) as in the
    source (line 791).
    """
    counts = np.bincount(preds.astype(np.int64),
                         minlength=num_classes).astype(np.float64)
    off = counts.copy()
    off[int(forget_class)] = 0.0   # ignore (residual) correct forget predictions
    off_total = off.sum()
    if off_total <= 0:
        return 0.0, 1.0
    p = off / off_total
    nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum() / np.log(max(2, num_classes - 1)))
    return float(p.max()), ent


@register_metric(name="collapse_resistance", table_id="guard",
                 category="structural", modalities=_CLS,
                 input_modes={"outputs", "model"}, cost="cheap")
def collapse_resistance(ctx: "EvalContext") -> MetricResult:
    """Validity guard: ``1 - max_offclass_frac`` of forget-sample predictions.

    Detects collapse gaming: a model that funnels all
    forget-class inputs into one substitute class has redirected, not
    forgotten. Value near ``1 - 1/(C-1)``-complement ... i.e. high values
    mean predictions spread across the other classes; 0 means total collapse
    onto a single class. Companion ``pred_entropy`` (normalized Shannon over
    the off-class distribution) ships in components. Qualifies ``ua``; not a
    ranking metric.

    Computed on ``forget_test`` when the mode defines it, falling back to the
    train-side ``forget`` split with a recorded warning. The forget class is
    derived from the split's targets (single-valued in class modes); if the
    targets are multi-valued (random-subset forgetting) the guard is not
    defined and the metric skips with ``not_applicable_mode``.
    """
    try:
        out = ctx.outputs("unlearned", "forget_test")
        basis = "forget_test"
    except SplitNotAvailable:
        logger.warning("collapse_resistance: forget_test empty in mode %r; "
                       "falling back to forget split", ctx.mode)
        ctx.warnings.append(
            f"collapse_resistance: forget_test unavailable in mode "
            f"{ctx.mode!r}; computed on the train-side forget split")
        out = ctx.outputs("unlearned", "forget")
        basis = "forget"
    if out.logits is None:
        raise MetricError("collapse_resistance requires logits; the outputs "
                          "payload has none")
    unique_targets = np.unique(out.targets)
    if unique_targets.size != 1:
        raise MetricSkip(
            "not_applicable_mode",
            "collapse_resistance requires a single forget class; the "
            f"{basis} split targets take {unique_targets.size} distinct "
            "values (random-subset forgetting has no collapse target)")
    forget_class = int(unique_targets[0])
    num_classes = int(out.logits.shape[1])
    preds = np.argmax(out.logits, axis=1)

    max_frac, ent = _offclass_stats(preds, forget_class, num_classes)
    ctx.stamp("structural.collapse_resistance.basis", basis)

    rng = numpy_rng(ctx.seed, "collapse_resistance:bootstrap")
    ci = bootstrap_ci(
        preds.astype(np.float64),
        lambda a: 1.0 - _offclass_stats(a, forget_class, num_classes)[0],
        n=ctx.hp.bootstrap.n, alpha=ctx.hp.bootstrap.alpha, rng=rng)
    return MetricResult(value=1.0 - max_frac, ci=ci, n=out.n,
                        components={"max_offclass_frac": max_frac,
                                    "pred_entropy": ent})


@register_metric(name="activation_distance", table_id="M22-adj",
                 category="structural", modalities=_CLS,
                 input_modes={"model"}, needs_gold=True, cost="moderate")
def activation_distance(ctx: "EvalContext") -> MetricResult:
    """Mean per-sample cosine distance (1 - cos) between penultimate features
    of the unlearned model and a reference model on ``retain_test``.

    Port of the original evaluation pipeline (``activation_distance``),
    recast over cached SplitOutputs.features (the adapter's probe captures
    penultimate features under model-in) instead of live hooks; the source's
    ``max_batches=8`` cost cap becomes a deterministic first-{N} sample cap
    over the canonical eval order.

    Reference role defaults to ``gold`` (protocol pin) and may be overridden
    via ``hp.metric_overrides["activation_distance"]["reference"]`` (e.g.
    ``"original"`` for drift-from-original analyses). 0 = identical
    representation.
    """
    overrides = ctx.hp.metric_overrides.get("activation_distance", {})
    ref_role = overrides.get("reference", "gold")
    feats_u = ctx.outputs("unlearned", "retain_test").features
    feats_r = ctx.outputs(ref_role, "retain_test").features
    if feats_u is None or feats_r is None:
        raise MetricError(
            "activation_distance requires penultimate features for both "
            f"roles ('unlearned', {ref_role!r}); features are only captured "
            "by model-in probes")
    a = np.asarray(feats_u, dtype=np.float64)
    b = np.asarray(feats_r, dtype=np.float64)
    a = a.reshape(len(a), -1)
    b = b.reshape(len(b), -1)
    if a.shape != b.shape:
        raise MetricError(f"activation_distance: feature shape mismatch "
                          f"{a.shape} vs {b.shape}")
    a = a[:_AD_MAX_SAMPLES]
    b = b[:_AD_MAX_SAMPLES]
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = (a * b).sum(axis=1) / np.maximum(denom, 1e-12)
    dists = 1.0 - cos
    ctx.stamp("structural.activation_distance.reference", ref_role)

    rng = numpy_rng(ctx.seed, "activation_distance:bootstrap")
    ci = bootstrap_ci(dists, n=ctx.hp.bootstrap.n,
                      alpha=ctx.hp.bootstrap.alpha, rng=rng)
    return MetricResult(value=float(np.mean(dists)), ci=ci, n=len(dists),
                        components=None)


# ---------------------------------------------------------------------------
# M62 — knn_transfer (Tier-3; external; downstream-transfer probe).
# ---------------------------------------------------------------------------

def knn_accuracy(features: np.ndarray, labels: np.ndarray, *, k: int,
                 rng: "np.random.Generator", test_fraction: float = 0.2,
                 metric: str = "cosine") -> float:
    """Deterministic kNN transfer accuracy (0-100) on a seeded split — numpy
    top-k, NOT sklearn (G1 bit-exact). Ties broken by LOWER train index (stable
    argsort) and the vote by LOWEST label (np.unique is ascending)."""
    X = np.asarray(features, dtype=np.float64).reshape(len(features), -1)
    y = np.asarray(labels)
    n = len(X)
    perm = rng.permutation(n)
    n_test = max(1, int(n * test_fraction))
    test_idx = np.sort(perm[:n_test])      # sort -> deterministic, rng-order-free
    train_idx = np.sort(perm[n_test:])
    Xtr, ytr, Xte, yte = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
    if len(Xtr) == 0:
        return float("nan")
    if metric == "cosine":
        Xtr_n = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-12)
        Xte_n = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-12)
        order = np.argsort(-(Xte_n @ Xtr_n.T), axis=1, kind="stable")  # most similar first
    else:  # l2
        d2 = ((Xte * Xte).sum(1)[:, None] - 2.0 * (Xte @ Xtr.T)
              + (Xtr * Xtr).sum(1)[None, :])
        order = np.argsort(d2, axis=1, kind="stable")                  # nearest first
    topk = order[:, :max(1, k)]
    preds = np.empty(len(Xte), dtype=y.dtype)
    for i in range(len(Xte)):
        vals, counts = np.unique(ytr[topk[i]], return_counts=True)
        preds[i] = vals[int(np.argmax(counts))]  # ascending vals -> lowest label on tie
    return float(100.0 * np.mean(preds == yte))


@register_metric(name="knn_transfer", table_id="M62", category="structural",
                 modalities=_CLS, input_modes={"model"}, needs_gold=True,
                 external=True, cost="expensive")
def knn_transfer(ctx: "EvalContext") -> MetricResult:
    """M62 — downstream-transfer gap (external, needs_gold): kNN-probe transfer
    accuracy of the UNLEARNED minus the GOLD penultimate features on a downstream
    dataset (hp.downstream). Negative = unlearning damaged transferable
    representations relative to the oracle."""
    from torch.utils.data import DataLoader, Subset

    from trail.data.datasets import build_datasets
    from trail.data.specs import get_test_transform

    ds = ctx.hp.downstream
    base = build_datasets(ds.name, ds.data_dir, splits=("train",),
                          train_transform=get_test_transform(ds.name))["train"]
    cap = min(ds.n_samples, len(base))
    loader = DataLoader(Subset(base, list(range(cap))), batch_size=256, shuffle=False)
    out_u = ctx.probe_external("unlearned", loader, seed_name="knn_transfer:unlearned")
    out_g = ctx.probe_external("gold", loader, seed_name="knn_transfer:gold")
    if out_u.features is None or out_g.features is None:
        raise MetricError("knn_transfer requires penultimate features (model-in)")
    labels = np.asarray(out_u.targets)

    def _acc(feats):  # same seeded split for both models -> comparable
        return knn_accuracy(feats, labels, k=ds.k,
                            rng=numpy_rng(ctx.seed, "knn_transfer:split"),
                            test_fraction=ds.test_fraction, metric=ds.metric)

    acc_u, acc_g = _acc(out_u.features), _acc(out_g.features)
    value = acc_u - acc_g
    ctx.stamp("structural.knn_transfer.downstream", ds.name)
    return MetricResult(value=value, ci=(value, value), n=int(cap),
                        components={"acc_unlearned": acc_u, "acc_gold": acc_g})


# ---------------------------------------------------------------------------
# Multi-depth representation (Tier-3; external; on F4 features_multi).
# M63 rsa_rdm_distance (CPU bit-exact), M65 linear_probe_at_depth.
# (M64 idi_gap — InfoNCE critic, GPU tolerance — reserved/deferred.)
# ---------------------------------------------------------------------------

_RSA_MAX_SAMPLES = 512   # the RDM is N x N — keep N modest (deterministic prefix)


def rdm(features: np.ndarray) -> np.ndarray:
    """Representational dissimilarity matrix: ``1 − Pearson`` between example
    feature vectors (CPU bit-exact)."""
    x = np.asarray(features, dtype=np.float64)
    return 1.0 - np.corrcoef(x)


def rsa_distance(feats_a: np.ndarray, feats_b: np.ndarray) -> float:
    """RSA distance ``1 − Spearman`` between the two RDMs' upper triangles
    (off-diagonal). 0 = identical representational geometry. CPU bit-exact."""
    from scipy import stats  # lazy
    a = np.asarray(feats_a, dtype=np.float64)[:_RSA_MAX_SAMPLES]
    b = np.asarray(feats_b, dtype=np.float64)[:_RSA_MAX_SAMPLES]
    ra, rb = rdm(a), rdm(b)
    iu = np.triu_indices_from(ra, k=1)
    rho, _ = stats.spearmanr(ra[iu], rb[iu])
    return float(1.0 - rho) if np.isfinite(rho) else float("nan")


def _features_multi(ctx: "EvalContext", role: str, name: str) -> dict:
    """Per-layer features for a role on retain_test; falls back to the single
    penultimate as a 1-depth map."""
    out = ctx.outputs(role, "retain_test")
    if out.features_multi:
        return {k: np.asarray(v, dtype=np.float64) for k, v in out.features_multi.items()}
    if out.features is not None:
        return {"penultimate": np.asarray(out.features, dtype=np.float64)}
    raise MetricError(f"{name} requires penultimate/multi-layer features (model-in)")


@register_metric(name="rsa_rdm_distance", table_id="M63", category="structural",
                 modalities=_CLS, input_modes={"model"}, needs_gold=True,
                 external=True, cost="moderate")
def rsa_rdm_distance(ctx: "EvalContext") -> MetricResult:
    """M63 — RSA distance between unlearned and gold representations, averaged
    over the captured depths (external, needs_gold, CPU bit-exact)."""
    fu = _features_multi(ctx, "unlearned", "rsa_rdm_distance")
    fg = _features_multi(ctx, "gold", "rsa_rdm_distance")
    shared = sorted(set(fu) & set(fg))
    if not shared:
        raise MetricError("rsa_rdm_distance: no shared feature layers")
    per = {layer: rsa_distance(fu[layer], fg[layer]) for layer in shared}
    value = float(np.mean([v for v in per.values() if np.isfinite(v)]))
    n = min(len(next(iter(fu.values()))), _RSA_MAX_SAMPLES)
    return MetricResult(value=value, ci=(value, value), n=int(n),
                        components={f"rsa@{k}": v for k, v in per.items()})


@register_metric(name="linear_probe_at_depth", table_id="M65", category="structural",
                 modalities=_CLS, input_modes={"model"}, external=True,
                 cost="moderate")
def linear_probe_at_depth(ctx: "EvalContext") -> MetricResult:
    """M65 — closed-form ridge linear-probe accuracy of the unlearned features at
    each captured depth (mean over depths; external, bit-exact — reuses the
    Tier-1 ridge, NOT lbfgs, for G1)."""
    from trail.metrics.representation import _RIDGE_LAMBDA, ridge_probe_accuracy

    out = ctx.outputs("unlearned", "retain_test")
    y = np.asarray(out.targets)[:_AD_MAX_SAMPLES]
    if np.unique(y).size < 2:
        raise MetricSkip("not_applicable_mode",
                         "linear_probe_at_depth needs >=2 classes in retain_test")
    fu = _features_multi(ctx, "unlearned", "linear_probe_at_depth")
    per = {layer: ridge_probe_accuracy(
        np.asarray(f, dtype=np.float64).reshape(len(f), -1)[:_AD_MAX_SAMPLES],
        y, lam=_RIDGE_LAMBDA) for layer, f in fu.items()}
    value = float(np.mean(list(per.values())))
    return MetricResult(value=value, ci=(value, value), n=int(len(y)),
                        components={f"probe@{k}": v for k, v in per.items()})
