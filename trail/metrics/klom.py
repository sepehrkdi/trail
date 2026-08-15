"""KLoM — KL divergence of Margins (M68, opt-in / ``external``).

KLoM (Rinberg et al. 2026, "Data-Unlearn-Bench: Making Evaluating Data
Unlearning Easy", arXiv 2602.16400) scores how close an unlearned model's
per-example margin DISTRIBUTION is to the oracle (retrained-from-scratch)
distribution. For each example x it bins the KLoM logit-margin of x under an
ENSEMBLE of oracle models and under an ENSEMBLE of unlearned models into fixed
histograms and takes their KL divergence::

    margin:   phi(x; theta) = f_y(x) - log sum_{k!=y} exp(f_k(x))          (clipped)
    KLoM(x) = D_KL( Hist({phi(x; theta^oracle_i)}) || Hist({phi(x; theta^unl_i)}) )

Direction is **KL(oracle || unlearned)** (oracle is the reference distribution);
lower is better, ->0 is near-perfect unlearning. Paper defaults (``hp.klom``):
20 bins over the clipped ``[-100, 100]`` margin range, Laplace smoothing
``eps=1e-5`` so empty bins stay finite (this caps KLoM at ~12), and N=100 oracle
+ 100 unlearned models. Unlike U-LiRA (M9) it needs no Gaussian assumption and
no adversarial distinguisher, and degenerate "cheat" methods are punished
because their margin law drifts from the oracle's. The headline is the mean
per-example KLoM over the FORGET split; retain/test are ``klom_<split>``
components.

REDUCED-ENSEMBLE CAVEAT (this deployment). trail runs KLoM at whatever
ensemble sizes are supplied via ``hp.references.gold_ensemble`` /
``unlearned_ensemble``. At the reduced multi-seed budget (N~3 existing seed
checkpoints, one-GPU reality) the 20-bin histograms are dominated by the
``eps`` smoothing floor, so the ABSOLUTE values are NOT comparable to the
paper's N=100 KLoM — they are valid only for RELATIVE ranking of methods
evaluated under the identical reduced protocol. ``n_oracle`` / ``n_unlearned``
are always reported so the N travels with the number, and ``external=True``
keeps KLoM off the default panel (opt in with ``metrics=["klom"]`` plus the two
ensemble lists). Whole ensembles are L3-cached (references/gold_ensemble.py);
the gold ensemble is method-independent and shared across methods.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from trail.core.errors import MetricError
from trail.core.registry import register_metric
from trail.core.report import MetricResult
from trail.core.seeding import numpy_rng

if TYPE_CHECKING:  # pragma: no cover
    from trail.core.context import EvalContext

logger = logging.getLogger("trail.metrics.klom")

_CLS: set[str] = {"classification"}


# ---------------------------------------------------------------------------
# Pure numerics (unit-testable without a context)
# ---------------------------------------------------------------------------

def margin_histogram(phi: np.ndarray, *, bins: int, clip: tuple[float, float],
                     eps: float) -> np.ndarray:
    """Smoothed, normalized histogram of one example's margins across an
    ensemble. Margins are clipped into ``clip`` (out-of-range mass folds into
    the edge bins); ``eps`` is added to every bin count before normalizing, so
    the distribution is strictly positive and the downstream KL stays finite."""
    lo, hi = clip
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(
        np.clip(np.asarray(phi, dtype=np.float64), lo, hi), bins=edges)
    p = counts.astype(np.float64) + eps
    return p / p.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """``D_KL(p || q) = sum_i p_i log(p_i / q_i)`` for two positive, normalized
    histograms (p = oracle, q = unlearned)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def klom_per_example(oracle_phi: np.ndarray, unlearned_phi: np.ndarray, *,
                     bins: int = 20, clip: tuple[float, float] = (-100.0, 100.0),
                     eps: float = 1e-5) -> np.ndarray:
    """Per-example KLoM over aligned ensembles.

    Args:
        oracle_phi: ``[G, N]`` margins of the oracle ensemble.
        unlearned_phi: ``[U, N]`` margins of the unlearned ensemble.
        bins/clip/eps: histogram recipe (paper defaults).

    Returns:
        ``[N]`` per-example ``KL(oracle_hist || unlearned_hist)``. Both
        ensembles must cover the SAME N examples in the same order.
    """
    oracle_phi = np.asarray(oracle_phi, dtype=np.float64)
    unlearned_phi = np.asarray(unlearned_phi, dtype=np.float64)
    if oracle_phi.ndim != 2 or unlearned_phi.ndim != 2:
        raise ValueError("oracle/unlearned margins must be [n_models, N]")
    n = oracle_phi.shape[1]
    if unlearned_phi.shape[1] != n:
        raise ValueError(
            f"ensemble N mismatch: oracle {n} vs unlearned "
            f"{unlearned_phi.shape[1]} (splits/order must align)")
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        p = margin_histogram(oracle_phi[:, i], bins=bins, clip=clip, eps=eps)
        q = margin_histogram(unlearned_phi[:, i], bins=bins, clip=clip, eps=eps)
        out[i] = kl_divergence(p, q)
    return out


# ---------------------------------------------------------------------------
# Registered metric (external=True: off the default panel)
# ---------------------------------------------------------------------------

@register_metric(name="klom", table_id="M68", category="distributional",
                 modalities=_CLS, input_modes={"model"},
                 external=True, cost="moderate")
def klom(ctx: "EvalContext") -> MetricResult:
    """KLoM (M68): KL divergence of per-example margin histograms, oracle vs
    unlearned ensembles, over the forget split (retain/test as components).

    Skips (via MissingReference) with ``reference_disabled:gold`` when either
    ensemble list is empty, ``requires_model_in`` under outputs-in, and
    ``not_applicable_mode`` when the forget split is absent.
    """
    gold = ctx.ensemble_margins("gold")           # MissingReference -> skip
    unlearned = ctx.ensemble_margins("unlearned")

    kcfg = ctx.hp.klom
    bins = int(kcfg.bins)
    clip = (float(kcfg.clip_low), float(kcfg.clip_high))
    eps = float(kcfg.eps)

    if "forget" not in gold.margins or "forget" not in unlearned.margins:
        raise MetricError("klom: forget-split margins missing from an ensemble")

    per_ex = klom_per_example(gold.margins["forget"], unlearned.margins["forget"],
                              bins=bins, clip=clip, eps=eps)

    reduced = min(gold.n_models, unlearned.n_models) < 100
    components: dict[str, float] = {
        "n_oracle": float(gold.n_models),
        "n_unlearned": float(unlearned.n_models),
        "reduced_ensemble": 1.0 if reduced else 0.0,
    }
    for split in ("retain", "test"):
        if split in gold.margins and split in unlearned.margins:
            comp = klom_per_example(gold.margins[split], unlearned.margins[split],
                                    bins=bins, clip=clip, eps=eps)
            components[f"klom_{split}"] = (
                float(np.mean(comp)) if comp.size else float("nan"))

    ctx.stamp("distributional.klom.direction",
              "KL(oracle || unlearned) of per-example margin histograms; "
              "lower=better (->0 near-perfect)")
    ctx.stamp("distributional.klom.ensemble", {
        "n_oracle": gold.n_models, "n_unlearned": unlearned.n_models,
        "bins": bins, "clip": list(clip), "eps": eps,
        "note": ("reduced-N proxy: N<100 histograms are smoothing-dominated; "
                 "absolute values are NOT comparable to the paper's N=100 KLoM "
                 "— use for relative ranking under the identical protocol only")
        if reduced else "paper-scale ensemble (N>=100)"})

    rng = numpy_rng(ctx.seed, "klom:bootstrap")
    return MetricResult.from_per_example(
        per_ex, rng=rng, n_boot=ctx.hp.bootstrap.n, alpha=ctx.hp.bootstrap.alpha,
        components=components)
