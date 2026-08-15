"""MIA ROC primitives: AUC and TPR @ low FPR (principles.md §C / adopted_fixes).

Reviewers demand strong MIA *reporting* — AUC and TPR at low FPR, not balanced
accuracy alone. These operate on the same per-example member / non-member
*score* arrays the population threshold-MIA already computes (higher score =
more member-like), so they add the strong-reporting axis without LiRA or shadow
models. sklearn (already a dependency) provides the ROC machinery.
"""
from __future__ import annotations

import numpy as np

#: Default operating points (principles.md §C: "TPR @ low FPR").
DEFAULT_FPR_TARGETS: tuple[float, ...] = (0.001, 0.01, 0.1)


def _labels_scores(member: np.ndarray, nonmember: np.ndarray):
    member = np.asarray(member, dtype=np.float64).ravel()
    nonmember = np.asarray(nonmember, dtype=np.float64).ravel()
    y = np.concatenate([np.ones(member.size), np.zeros(nonmember.size)])
    s = np.concatenate([member, nonmember])
    return member, nonmember, y, s


def roc_auc(member: np.ndarray, nonmember: np.ndarray) -> float:
    """ROC AUC with the member class as positive (higher score = member-like).

    Returns ``nan`` if either side is empty (AUC undefined); 0.5 when scores
    carry no signal.
    """
    member, nonmember, y, s = _labels_scores(member, nonmember)
    if member.size == 0 or nonmember.size == 0:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def tpr_at_fpr(member: np.ndarray, nonmember: np.ndarray,
               fpr_targets: "tuple[float, ...] | list[float]" = DEFAULT_FPR_TARGETS,
               ) -> dict[str, float]:
    """True-positive rate at each target false-positive rate.

    For each target FPR, the TPR is read off the ROC curve by interpolation
    (``np.interp`` over the sklearn ``roc_curve`` points). Keys are
    ``"tpr@<fpr>"``. Empty side -> all ``nan``.
    """
    member, nonmember, y, s = _labels_scores(member, nonmember)
    out: dict[str, float] = {f"tpr@{t:g}": float("nan") for t in fpr_targets}
    if member.size == 0 or nonmember.size == 0:
        return out
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)  # fpr ascending, endpoints (0,0)/(1,1)
    for t in fpr_targets:
        out[f"tpr@{t:g}"] = float(np.interp(t, fpr, tpr))
    return out
