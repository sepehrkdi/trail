"""Privacy metrics: membership-inference attacks M8/M9.

Attack stack (fixed, tiered, balanced):

* ``mia_threshold_population`` (M8, default panel) — per-class-threshold
  population MIA (Jia et al. 2023 line), four signals, headline = modified
  entropy. Ported from the original evaluation pipeline.
* ``mia_loss_logreg`` (M8 cross-check, default panel) — loss-feature
  logistic-regression MIA, following the protocol described by Kurmanji
  et al. 2023 (SCRUB); implemented here directly on scikit-learn.
* ``mia_svc`` (M8, opt-in) — RBF-SVC shadow attack. Ported from
  the original evaluation pipeline.
* ``mia_lira`` (M9, opt-in, shadow tier) — online per-example LiRA (Carlini
  et al. 2022 / U-LiRA). Opt in via ``hp.references.shadow>0``; trains a
  method-independent shadow ensemble (L3-cached) and reports AUC + TPR@low-FPR
  over forget (member-candidate) vs test (non-member). Not in the default panel.

Conventions pinned here:

* MIA accuracies are reported on the 0-1 scale (balanced attack accuracy);
  the ``advantage`` component ``2*|acc - 0.5|`` is the cross-paper number.
* Every attack stamps an explicit ``direction`` field — the member /
  non-member orientation it was run in.
* An augmentation-off guard is subsumed by the canonical-eval-view contract:
  ``ctx.outputs(...)`` is always probed over the augmentation-stripped
  canonical loader, so members and non-members see identical preprocessing.

Documented deviation: all shadow/target permutations and balanced subsamples
here derive from named numpy substreams (``trail.core.seeding``) —
``mia:shadow_retain`` / ``mia:shadow_test`` / ``mia:logreg_balance``.
Implementations that instead seed ``torch.randperm`` globally will draw
different shadow calibration and target populations, so M8 attack
accuracies are not byte-comparable against them; the difference is
population-level (small at CIFAR scale, but nonzero).
"""
# Portions of this file (the population threshold attack) are adapted from
# OPTML-Group/Unlearn-Sparse, Copyright (c) 2023 OPTML Group, MIT License.
# See the third-party notice in LICENSE.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Mapping

import numpy as np

from trail.core.bootstrap import bootstrap_ci, bootstrap_ci_groups
from trail.core.errors import MetricError, NonFiniteLossError
from trail.core.registry import register_metric
from trail.core.report import MetricResult
from trail.core.seeding import numpy_rng
from trail.metrics._signals import (
    entropy,
    modified_entropy,
    per_class_threshold,
    softmax_np,
)

if TYPE_CHECKING:  # pragma: no cover
    from trail.core.context import EvalContext
    from trail.core.types import SplitOutputs

logger = logging.getLogger("trail.metrics.privacy")


# ──────────────────────────────────────────────────────────────────────────
# Split-role wiring (port of the benchmark_panel shadow/target construction,
# the original evaluation pipeline, with torch.randperm replaced by the
# named numpy substreams mandated by trail.core.seeding).
# ──────────────────────────────────────────────────────────────────────────

def _mia_split_roles(ctx: "EvalContext") -> dict[str, np.ndarray]:
    """Build the four shadow/target index sets for the population MIAs.

    Roles (identical to the Unlearn-Sparse forget pipeline; see
    the original evaluation pipeline):

    * ``shadow_train``      — retain 1st half  (members, threshold calibration)
    * ``target_train``      — retain 2nd half  (members, scoring)
    * ``shadow_test``       — test 1st half    (non-members, calibration)
    * ``target_nonmember``  — the FORGET set   (the population under test)

    Returns a dict of numpy index arrays. ``shadow_train``/``target_train``
    index into ``ctx.outputs("unlearned", "retain")`` arrays,
    ``shadow_test`` into the ``test`` arrays, and ``target_nonmember`` into
    the ``forget`` arrays. Permutations come from the named substreams
    ``mia:shadow_retain`` and ``mia:shadow_test`` so adding/removing other
    metrics can never perturb the shadow split (guarantee G2).
    """
    n_retain = ctx.outputs("unlearned", "retain").n
    n_test = ctx.outputs("unlearned", "test").n
    n_forget = ctx.outputs("unlearned", "forget").n

    rperm = numpy_rng(ctx.seed, "mia:shadow_retain").permutation(n_retain)
    tperm = numpy_rng(ctx.seed, "mia:shadow_test").permutation(n_test)
    rhalf, thalf = n_retain // 2, n_test // 2

    return {
        "shadow_train": rperm[:rhalf],
        "target_train": rperm[rhalf:],
        "shadow_test": tperm[:thalf],
        "target_nonmember": np.arange(n_forget, dtype=np.int64),
    }


def _probs_and_labels(out: "SplitOutputs", split: str) -> tuple[np.ndarray, np.ndarray]:
    """Softmax probabilities + integer labels from one SplitOutputs.

    Raises MetricError when logits are absent (non-classification payload) or
    non-finite (non-finite stats surface at the consuming metric).
    """
    if out.logits is None:
        raise MetricError(
            f"split {split!r}: logits are required for the shadow-calibrated "
            "MIAs but are absent from the outputs payload"
        )
    if not np.all(np.isfinite(out.logits)):
        raise MetricError(f"split {split!r}: non-finite logits encountered")
    return softmax_np(out.logits), out.targets.astype(np.int64)


def _gather_role_probs(ctx: "EvalContext") -> dict[str, np.ndarray]:
    """Slice softmax probs/labels for the four MIA roles of _mia_split_roles."""
    roles = _mia_split_roles(ctx)
    retain_p, retain_y = _probs_and_labels(ctx.outputs("unlearned", "retain"), "retain")
    test_p, test_y = _probs_and_labels(ctx.outputs("unlearned", "test"), "test")
    forget_p, forget_y = _probs_and_labels(ctx.outputs("unlearned", "forget"), "forget")
    return {
        "s_member_probs": retain_p[roles["shadow_train"]],
        "s_member_y": retain_y[roles["shadow_train"]],
        "s_non_probs": test_p[roles["shadow_test"]],
        "s_non_y": test_y[roles["shadow_test"]],
        "t_member_probs": retain_p[roles["target_train"]],
        "t_member_y": retain_y[roles["target_train"]],
        "t_non_probs": forget_p[roles["target_nonmember"]],
        "t_non_y": forget_y[roles["target_nonmember"]],
    }


def _true_class_confidence(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """p(true class) per example — port of the original evaluation pipeline ``_conf``."""
    return probs[np.arange(len(labels)), labels]


def _calibrate_per_class(member_vals: np.ndarray, member_y: np.ndarray,
                         non_vals: np.ndarray, non_y: np.ndarray,
                         num_classes: int) -> np.ndarray:
    """Per-class thresholds maximizing 0.5*(TPR+TNR) on the shadow arrays.

    Port of the per-class loop in the original evaluation pipeline (``_thre_attack``),
    returning an array indexed by class. Classes absent from either shadow
    side fall back to a single global threshold (logged); the CIFAR-scale
    fixtures never hit this branch.
    """
    thresholds = np.empty(num_classes, dtype=np.float64)
    fallback: float | None = None
    for c in range(num_classes):
        m = member_vals[member_y == c]
        n = non_vals[non_y == c]
        if len(m) == 0 or len(n) == 0:
            if fallback is None:
                fallback = per_class_threshold(member_vals, non_vals)
            logger.warning(
                "class %d absent from shadow calibration split; "
                "falling back to global threshold", c)
            thresholds[c] = fallback
        else:
            thresholds[c] = per_class_threshold(m, n)
    return thresholds


def _threshold_attack_acc(thresholds: np.ndarray,
                          t_member_vals: np.ndarray, t_member_y: np.ndarray,
                          t_non_vals: np.ndarray, t_non_y: np.ndarray,
                          ) -> tuple[float, float]:
    """(member_acc, nonmember_acc) under fixed per-class thresholds.

    Vectorized equivalent of the original evaluation pipeline: a target example counts
    as member when its signal value >= its class threshold.
    """
    member_acc = float(np.mean(t_member_vals >= thresholds[t_member_y]))
    nonmember_acc = float(np.mean(t_non_vals < thresholds[t_non_y]))
    return member_acc, nonmember_acc


# ──────────────────────────────────────────────────────────────────────────
# M8 — population threshold MIA (the registered default).
# ──────────────────────────────────────────────────────────────────────────

@register_metric(name="mia_threshold_population", table_id="M8",
                 category="privacy", modalities={"classification"},
                 input_modes={"outputs", "model"}, cost="moderate")
def mia_threshold_population(ctx: "EvalContext") -> "MetricResult":
    """Population threshold-MIA, per-class-threshold formulation (M8).

    Port of ``run_threshold_mia`` (the original evaluation pipeline;
    originally OPTML-Group/Unlearn-Sparse MIA.py ``black_box_benchmarks``).
    Per-class thresholds are calibrated on shadow data (retain 1st half =
    members vs test 1st half = non-members) to maximize balanced accuracy
    0.5*(TPR+TNR), then transferred to the disjoint target population
    (retain 2nd half = members vs forget set = non-members). Four signals:
    correctness, confidence, (negated) entropy, (negated) modified entropy
    (Song & Mittal). Balanced attack accuracy per signal =
    0.5*(member_acc + nonmember_acc); 0.5 is chance.

    Headline value (protocol pin 1): the ``m_entropy`` signal's balanced
    attack accuracy, on the 0-1 scale. Components carry all four per-signal
    accuracies plus ``advantage`` = 2*|value - 0.5|.

    CI: thresholds are FIXED at the point estimate; the member and
    non-member target arrays are bootstrap-resampled jointly
    (``bootstrap_ci_groups``) with the m_entropy balanced accuracy as the
    statistic. Re-calibrating thresholds per replicate would mix attacker
    variance into the population CI.
    """
    data = _gather_role_probs(ctx)
    num_classes = data["s_member_probs"].shape[1]

    components: dict[str, float] = {}

    # Signal 1 — correctness (no threshold; the original evaluation pipeline).
    corr_member = float(np.mean(
        np.argmax(data["t_member_probs"], axis=1) == data["t_member_y"]))
    corr_nonmember = float(np.mean(
        np.argmax(data["t_non_probs"], axis=1) != data["t_non_y"]))
    components["correctness"] = 0.5 * (corr_member + corr_nonmember)

    # Signals 2-4 — thresholded; sign conventions follow the source protocol
    # (entropy/m_entropy negated so that "higher = more member-like").
    signal_values: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {
        "confidence": (
            _true_class_confidence(data["s_member_probs"], data["s_member_y"]),
            _true_class_confidence(data["s_non_probs"], data["s_non_y"]),
            _true_class_confidence(data["t_member_probs"], data["t_member_y"]),
            _true_class_confidence(data["t_non_probs"], data["t_non_y"]),
        ),
        "entropy": (
            -entropy(data["s_member_probs"]),
            -entropy(data["s_non_probs"]),
            -entropy(data["t_member_probs"]),
            -entropy(data["t_non_probs"]),
        ),
        "m_entropy": (
            -modified_entropy(data["s_member_probs"], data["s_member_y"]),
            -modified_entropy(data["s_non_probs"], data["s_non_y"]),
            -modified_entropy(data["t_member_probs"], data["t_member_y"]),
            -modified_entropy(data["t_non_probs"], data["t_non_y"]),
        ),
    }

    me_thresholds: np.ndarray | None = None
    me_t_member: np.ndarray | None = None
    me_t_non: np.ndarray | None = None
    for name, (s_m, s_n, t_m, t_n) in signal_values.items():
        thresholds = _calibrate_per_class(
            s_m, data["s_member_y"], s_n, data["s_non_y"], num_classes)
        member_acc, nonmember_acc = _threshold_attack_acc(
            thresholds, t_m, data["t_member_y"], t_n, data["t_non_y"])
        components[name] = 0.5 * (member_acc + nonmember_acc)
        if name == "m_entropy":
            me_thresholds, me_t_member, me_t_non = thresholds, t_m, t_n

    assert me_thresholds is not None  # m_entropy is always in signal_values
    value = components["m_entropy"]
    components["advantage"] = float(2.0 * abs(value - 0.5))

    # Strong MIA reporting (C2 / principles.md §C): AUC + TPR @ low FPR on the
    # headline m_entropy signal's target member/non-member scores. No LiRA /
    # shadow models — these read off the same population scores.
    from trail.metrics._mia_curves import roc_auc, tpr_at_fpr
    fpr_targets = tuple(ctx.hp.mia.fpr_targets)
    components["auc"] = roc_auc(me_t_member, me_t_non)
    components.update(tpr_at_fpr(me_t_member, me_t_non, fpr_targets))

    ctx.stamp("privacy.mia_threshold_population.direction",
              "members=retain_eval; nonmembers=forget")

    # CI: fixed thresholds, joint member/non-member resampling. Rows pack
    # (signal value, label) so each replicate keeps value-label pairing.
    member_rows = np.column_stack(
        [me_t_member, data["t_member_y"].astype(np.float64)])
    nonmember_rows = np.column_stack(
        [me_t_non, data["t_non_y"].astype(np.float64)])
    thresholds_fixed = me_thresholds

    def _balanced_acc(groups: Mapping[str, np.ndarray]) -> float:
        mem, non = groups["member"], groups["nonmember"]
        m_acc = float(np.mean(mem[:, 0] >= thresholds_fixed[mem[:, 1].astype(np.int64)]))
        n_acc = float(np.mean(non[:, 0] < thresholds_fixed[non[:, 1].astype(np.int64)]))
        return 0.5 * (m_acc + n_acc)

    rng = numpy_rng(ctx.seed, "mia_threshold_population:bootstrap")
    ci = bootstrap_ci_groups(
        {"member": member_rows, "nonmember": nonmember_rows}, _balanced_acc,
        n=ctx.hp.bootstrap.n, alpha=ctx.hp.bootstrap.alpha, rng=rng)

    n_balanced = 2 * min(len(member_rows), len(nonmember_rows))
    return MetricResult(value=float(value), ci=ci, n=n_balanced,
                        components=components)


# ──────────────────────────────────────────────────────────────────────────
# M8 cross-check — loss logistic-regression MIA (SCRUB-style).
# ──────────────────────────────────────────────────────────────────────────

@register_metric(name="mia_loss_logreg", table_id="M8", category="privacy",
                 modalities={"classification", "llm"},
                 input_modes={"outputs", "model"}, cost="cheap")
def mia_loss_logreg(ctx: "EvalContext") -> "MetricResult":
    """Loss-based logistic-regression MIA (M8 cross-check, default panel).

    Implements the loss-feature attack protocol described by Kurmanji et al.
    2023 (SCRUB), written directly against scikit-learn rather than ported
    from any implementation. Members = FORGET-set per-example losses,
    non-members = TEST-set per-example losses (protocol pin 4 — the legacy
    forget vs forget_test wiring is deliberately NOT implemented). Both sides
    are balanced-subsampled to min(n_forget, n_test); the sole feature is the
    per-example loss clipped to [-100, 100]; a LogisticRegression is scored
    by 5-fold StratifiedShuffleSplit with ``test_size=0.1`` — sklearn's
    default, which the legacy ``StratifiedShuffleSplit(n_splits=5,
    random_state=seed)`` (the original evaluation pipeline) resolved to; pinned explicitly
    here so the CV eval-fold size matches the ported protocol — using
    balanced confusion-matrix accuracy 0.5*(TPR+TNR) per fold (hardened from
    the port's plain ``_cm_accuracy``, the original evaluation pipelinewhich on a
    balanced set is equivalent in expectation).

    Documented deviation: the balanced subsample and the CV random_state
    derive from named numpy substreams (``mia:logreg_balance`` /
    ``mia:logreg_cv``) rather than the legacy global-RNG draws — see the
    module-docstring deviation note (not byte-reproducible vs legacy panels).

    Value = mean fold balanced accuracy (0-1 scale; ~0.5 = no leakage).
    Components: ``std`` (fold std), ``advantage`` = 2*|value - 0.5|.
    CI: percentile bootstrap over the 5-fold accuracy array — degenerate-ish
    with five points, but honest about what was measured; no refitting.
    n = 2*min(n_forget, n_test), the balanced attack population.
    """
    # Lazy sklearn imports (heavyweight; only paid when the metric runs).
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

    forget_losses = np.asarray(ctx.outputs("unlearned", "forget").losses)
    test_losses = np.asarray(ctx.outputs("unlearned", "test").losses)
    if not (np.all(np.isfinite(forget_losses)) and np.all(np.isfinite(test_losses))):
        raise NonFiniteLossError(
            "mia_loss_logreg: non-finite per-example losses in forget/test outputs")

    # Balanced subsample, drawn from a named Generator substream.
    rng = numpy_rng(ctx.seed, "mia:logreg_balance")
    n = int(min(len(forget_losses), len(test_losses)))
    forget_sub = rng.choice(forget_losses, size=n, replace=False)
    test_sub = rng.choice(test_losses, size=n, replace=False)

    # Feature/label construction order: test=0 first, forget=1 second.
    features = np.clip(
        np.concatenate([test_sub, forget_sub]).reshape(-1, 1), -100.0, 100.0)
    labels = np.array([0] * n + [1] * n)

    def _balanced_cm_accuracy(estimator, X, y) -> float:
        """0.5*(TPR+TNR) from the fold confusion matrix."""
        y_pred = estimator.predict(X)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        tnr = cm[0, 0] / max(int(cm[0].sum()), 1)
        tpr = cm[1, 1] / max(int(cm[1].sum()), 1)
        return 0.5 * (tpr + tnr)

    cv = StratifiedShuffleSplit(
        n_splits=5, test_size=0.1,  # sklearn defaults
        random_state=ctx.seed_for("mia:logreg_cv") % 2**32)
    scores = np.asarray(cross_val_score(
        LogisticRegression(), features, labels, cv=cv,
        scoring=_balanced_cm_accuracy))

    value = float(scores.mean())
    ctx.stamp("privacy.mia_loss_logreg.direction", "members=forget; nonmembers=test")

    rng_boot = numpy_rng(ctx.seed, "mia_loss_logreg:bootstrap")
    ci = bootstrap_ci(scores, n=ctx.hp.bootstrap.n,
                      alpha=ctx.hp.bootstrap.alpha, rng=rng_boot)
    # Strong MIA reporting: AUC + TPR @ low FPR on the loss signal.
    # Members (forget) have LOWER loss, so the member-like score is -loss.
    from trail.metrics._mia_curves import roc_auc, tpr_at_fpr
    fpr_targets = tuple(ctx.hp.mia.fpr_targets)
    components = {
        "std": float(scores.std()),
        "advantage": float(2.0 * abs(value - 0.5)),
        "auc": roc_auc(-forget_sub, -test_sub),
    }
    components.update(tpr_at_fpr(-forget_sub, -test_sub, fpr_targets))
    return MetricResult(value=value, ci=ci, n=2 * n, components=components)


# ──────────────────────────────────────────────────────────────────────────
# M8 — RBF-SVC MIA (opt-in; not in the adapter default panel).
# ──────────────────────────────────────────────────────────────────────────

@register_metric(name="mia_svc", table_id="M8", category="privacy",
                 modalities={"classification"},
                 input_modes={"outputs", "model"}, cost="expensive",
                 external=False)
def mia_svc(ctx: "EvalContext") -> "MetricResult":
    """RBF-SVC membership inference attack (M8 variant, opt-in).

    Port of ``run_svc_mia`` (the original evaluation pipeline;
    originally OPTML-Group/Unlearn-Sparse SVC_MIA.py). For each of five
    signals — correctness, confidence, entropy, modified entropy, and the
    full softmax probability vector — an ``SVC(C=3, gamma="auto",
    kernel="rbf")`` is fit on the shadow split (retain 1st half = members,
    test 1st half = non-members) and scored as balanced attack accuracy on
    the target split (retain 2nd half = members, forget set = non-members).
    Same split roles as ``mia_threshold_population``. Unlike the threshold
    attack, the SVC consumes raw (un-negated) entropy signals — it learns
    the decision boundary itself (the original evaluation pipeline).

    Value = the ``m_entropy`` signal's balanced attack accuracy (0-1).
    Components: all five per-signal accuracies plus ``advantage``.
    CI: the fitted m_entropy SVC is FIXED; its per-example member /
    non-member correctness indicators on the target split are jointly
    bootstrap-resampled (no refitting — analogous to the fixed-threshold CI).
    """
    from sklearn.svm import SVC  # lazy import

    data = _gather_role_probs(ctx)

    def _corr(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return (np.argmax(probs, axis=1) == labels).astype(np.float64).reshape(-1, 1)

    def _conf(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return _true_class_confidence(probs, labels).reshape(-1, 1)

    # Signal features, order/shape per the original evaluation pipeline.
    signal_feats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {
        "correctness": (
            _corr(data["s_member_probs"], data["s_member_y"]),
            _corr(data["s_non_probs"], data["s_non_y"]),
            _corr(data["t_member_probs"], data["t_member_y"]),
            _corr(data["t_non_probs"], data["t_non_y"]),
        ),
        "confidence": (
            _conf(data["s_member_probs"], data["s_member_y"]),
            _conf(data["s_non_probs"], data["s_non_y"]),
            _conf(data["t_member_probs"], data["t_member_y"]),
            _conf(data["t_non_probs"], data["t_non_y"]),
        ),
        "entropy": tuple(  # type: ignore[assignment]
            entropy(p).reshape(-1, 1) for p in
            (data["s_member_probs"], data["s_non_probs"],
             data["t_member_probs"], data["t_non_probs"])
        ),
        "m_entropy": (
            modified_entropy(data["s_member_probs"], data["s_member_y"]).reshape(-1, 1),
            modified_entropy(data["s_non_probs"], data["s_non_y"]).reshape(-1, 1),
            modified_entropy(data["t_member_probs"], data["t_member_y"]).reshape(-1, 1),
            modified_entropy(data["t_non_probs"], data["t_non_y"]).reshape(-1, 1),
        ),
        "prob": (data["s_member_probs"], data["s_non_probs"],
                 data["t_member_probs"], data["t_non_probs"]),
    }

    components: dict[str, float] = {}
    me_member_ind: np.ndarray | None = None
    me_non_ind: np.ndarray | None = None
    for name, (s_m, s_n, t_m, t_n) in signal_feats.items():
        # Port of _svc_fit_predict (the original evaluation pipeline): members=1.
        x_shadow = np.concatenate([s_m, s_n]).reshape(len(s_m) + len(s_n), -1)
        y_shadow = np.concatenate([np.ones(len(s_m)), np.zeros(len(s_n))])
        clf = SVC(C=3, gamma="auto", kernel="rbf")
        clf.fit(x_shadow, y_shadow)
        member_ind = (clf.predict(t_m.reshape(len(t_m), -1)) == 1).astype(np.float64)
        non_ind = (clf.predict(t_n.reshape(len(t_n), -1)) == 0).astype(np.float64)
        components[name] = 0.5 * (float(member_ind.mean()) + float(non_ind.mean()))
        if name == "m_entropy":
            me_member_ind, me_non_ind = member_ind, non_ind
        logger.debug("mia_svc signal %s: attack_acc=%.4f", name, components[name])

    assert me_member_ind is not None and me_non_ind is not None
    value = components["m_entropy"]
    components["advantage"] = float(2.0 * abs(value - 0.5))

    ctx.stamp("privacy.mia_svc.direction", "members=retain_eval; nonmembers=forget")

    def _balanced_acc(groups: Mapping[str, np.ndarray]) -> float:
        return 0.5 * (float(groups["member"].mean()) +
                      float(groups["nonmember"].mean()))

    rng = numpy_rng(ctx.seed, "mia_svc:bootstrap")
    ci = bootstrap_ci_groups(
        {"member": me_member_ind, "nonmember": me_non_ind}, _balanced_acc,
        n=ctx.hp.bootstrap.n, alpha=ctx.hp.bootstrap.alpha, rng=rng)

    n_balanced = 2 * min(len(me_member_ind), len(me_non_ind))
    return MetricResult(value=float(value), ci=ci, n=n_balanced,
                        components=components)


# ──────────────────────────────────────────────────────────────────────────
# M9 — LiRA (opt-in shadow tier). Online per-example likelihood-ratio MIA.
# ──────────────────────────────────────────────────────────────────────────

@register_metric(name="mia_lira", table_id="M9", category="privacy",
                 modalities={"classification"},
                 input_modes={"model"},
                 needs_shadow=True, cost="expensive")
def mia_lira(ctx: "EvalContext") -> "MetricResult":
    """Online LiRA membership inference (M9, Carlini et al. 2022; U-LiRA framing
    of Hayes et al. 2025) — the opt-in rigorous privacy tier.

    Opt in by adding ``"mia_lira"`` to ``metrics`` AND setting
    ``hp.references.shadow > 0`` (the pinned budget is 8). NOT in the default
    panel; off by default, so default runs never pay the shadow-training cost.

    Scoring:

    1. ``ctx.shadow_stats()`` builds (or loads from L3) the shadow ensemble —
       method-INDEPENDENT, amortized across every method on the same data —
       recording each audit example's confidence signal phi under each shadow
       plus an IN/OUT membership mask (references/shadow.py).
    2. The target (unlearned) model's phi is read off the SAME audit pool
       (canonical forget then test), via ``ctx.outputs`` (no extra forward
       pass beyond the L1-cached probes).
    3. ``attacks.lira.lira_scores`` computes the per-example two-sided Gaussian
       likelihood ratio (global-variance variant, robust at 8 shadows). The
       AUC + TPR @ low FPR distinguishing forget (member-candidate) from test
       (non-member) is the LiRA membership signal.

    Headline value: the LiRA **AUC** (higher = more residual membership
    leakage = the unlearning scrubbed the forget set less). Reported as a
    reproducible privacy *ceiling at the fixed shadow budget*, not a definitive
    bound — consistent with the framework's ``evaluation_guarantee="empirical"``.

    Disabled (``hp.references.shadow == 0``): ``ctx.shadow_stats()`` raises
    MissingReference -> ``reference_disabled:shadow`` skip.
    """
    from trail.attacks.lira import confidence_logit, lira_scores
    from trail.metrics._mia_curves import roc_auc, tpr_at_fpr

    ss = ctx.shadow_stats()  # MissingReference -> reference_disabled:shadow skip

    # Target phi over the audit pool, in the builder's order: forget then test.
    f_out = ctx.outputs("unlearned", "forget")
    t_out = ctx.outputs("unlearned", "test")
    for split, out in (("forget", f_out), ("test", t_out)):
        if out.logits is None:
            raise MetricError(
                f"split {split!r}: LiRA needs logits, absent from the payload")
        if not np.all(np.isfinite(out.logits)):
            raise MetricError(f"split {split!r}: non-finite logits encountered")
    target_phi = np.concatenate([
        confidence_logit(f_out.logits, f_out.targets.astype(np.int64)),
        confidence_logit(t_out.logits, t_out.targets.astype(np.int64)),
    ])

    if target_phi.shape[0] != ss.shadow_phi.shape[1]:
        raise MetricError(
            f"audit-pool size mismatch: target has {target_phi.shape[0]} "
            f"examples, shadow ensemble has {ss.shadow_phi.shape[1]} "
            "(forget/test split sizes changed since the ensemble was cached)")

    scores, n_fallback = lira_scores(ss.shadow_phi, ss.member_mask, target_phi)
    labels = np.asarray(ss.audit_labels).astype(np.int64)
    member_scores = scores[labels == 1]      # forget = member-candidate
    test_scores = scores[labels == 0]        # held-out test = true non-member

    # Non-member basis (protocol pin "test", or opt-in "class_matched"). For
    # single-class forgetting, class_matched restricts non-members to test
    # examples of the forget class (forget_test) so the AUC isolates
    # *membership* from class-identity; selection is post-hoc over the cached
    # ensemble (no rebuild). The test-portion labels are index-aligned with
    # test_scores (both follow the canonical test loader order).
    basis = ctx.hp.mia.lira_nonmembers
    if basis == "class_matched":
        forget_classes = np.unique(f_out.targets.astype(np.int64))
        test_labels = t_out.targets.astype(np.int64)
        sel = np.isin(test_labels, forget_classes)
        nonmember_scores = test_scores[sel]
        if nonmember_scores.size == 0:
            raise MetricError(
                "lira_nonmembers='class_matched' selected zero test examples "
                f"of the forget class(es) {forget_classes.tolist()}")
    else:
        nonmember_scores = test_scores
    ctx.stamp("privacy.mia_lira.nonmember_basis", basis)

    fpr_targets = tuple(ctx.hp.mia.fpr_targets)
    auc = roc_auc(member_scores, nonmember_scores)
    components: dict[str, float] = {
        "advantage": float(2.0 * abs(auc - 0.5)) if np.isfinite(auc) else float("nan"),
        "n_shadow": float(ss.n_shadow),
        "n_members": float(member_scores.size),
        "n_nonmembers": float(nonmember_scores.size),
        "n_mean_fallback": float(n_fallback),
    }
    components.update(tpr_at_fpr(member_scores, nonmember_scores, fpr_targets))

    ctx.stamp("privacy.mia_lira.direction",
              "members=forget; nonmembers=test")

    # CI: bootstrap the AUC over jointly-resampled member/non-member LiRA
    # scores (the shadow ensemble is FIXED — its training variance is not part
    # of this population CI, exactly as the threshold-MIA fixes its thresholds).
    member_rows = member_scores.reshape(-1, 1)
    nonmember_rows = nonmember_scores.reshape(-1, 1)

    def _auc_stat(groups: Mapping[str, np.ndarray]) -> float:
        return roc_auc(groups["member"][:, 0], groups["nonmember"][:, 0])

    rng = numpy_rng(ctx.seed, "mia_lira:bootstrap")
    ci = bootstrap_ci_groups(
        {"member": member_rows, "nonmember": nonmember_rows}, _auc_stat,
        n=ctx.hp.bootstrap.n, alpha=ctx.hp.bootstrap.alpha, rng=rng)

    n = int(member_scores.size + nonmember_scores.size)
    return MetricResult(value=float(auc), ci=ci, n=n, components=components)
