"""LiRA — Likelihood-Ratio membership-inference scoring (M9 opt-in tier).

The per-example calibrated attack of Carlini et al. 2022 ("Membership
Inference Attacks From First Principles"), in the unlearning-audit framing of
Hayes et al. 2025 (U-LiRA): for each audit example, the target (unlearned)
model's confidence is tested against the per-example distributions of that
confidence under shadow models that DID (IN) and DID NOT (OUT) train on it.
The membership ground truth distinguished by the resulting score is forget
(member-candidate of the original training set) vs. test (true non-member).

This module is pure numerics: it consumes a frozen shadow ensemble
(``ShadowStats`` from references/shadow.py) plus the target's per-example
confidence signal, and returns a per-example LiRA statistic. Training the
shadow ensemble lives in references/shadow.py; AUC / TPR@low-FPR reduction and
the report wiring live in metrics/privacy.py.

Two deliberate choices, both standard for a SMALL ensemble (the pinned budget
is 8 shadows, where per-example variance is too noisy to estimate reliably):

* **Global variance.** Per-example IN / OUT *means* are estimated per example,
  but the IN / OUT *standard deviations* are pooled across the whole audit
  pool (Carlini's "global variance" variant). This is markedly more stable
  than per-example variance at 8 shadows.
* **Mean fallback.** If an example happens to be IN (or OUT) of zero shadows,
  its missing-side mean falls back to the global mean of that side, so the
  two-sided likelihood ratio is always defined. Such examples are counted and
  surfaced (``n_mean_fallback``) rather than silently dropped (G4).
"""
from __future__ import annotations

import numpy as np

#: True-class probability is clipped to ``[clip, 1-clip]`` before the logit
#: transform so that a fully-confident (p=1) or fully-wrong (p=0) example
#: yields a finite phi. 1e-6 matches the LiRA reference implementations.
DEFAULT_CONF_CLIP: float = 1e-6

#: Floor on the pooled IN / OUT standard deviations (phi scale) so the Gaussian
#: log-density stays finite when a side is degenerate.
DEFAULT_VAR_FLOOR: float = 1e-2


def confidence_logit(logits: np.ndarray, labels: np.ndarray, *,
                     clip: float = DEFAULT_CONF_CLIP) -> np.ndarray:
    """The LiRA confidence signal phi = logit(p_true) per example.

    ``p_true`` is the softmax probability of the example's true class; the
    logit transform ``log(p / (1 - p))`` variance-stabilizes it (Carlini et
    al. 2022 §V.A) so the IN / OUT distributions are closer to Gaussian.
    Higher phi = more confident on the true class = more member-like.

    Args:
        logits: ``[N, C]`` raw model logits.
        labels: ``[N]`` integer true-class indices.
        clip: probability clip for numerical stability.

    Returns:
        ``[N]`` float64 array of phi values.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64).ravel()
    # Stable softmax over the class axis.
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    probs = ez / ez.sum(axis=1, keepdims=True)
    p_true = probs[np.arange(len(labels)), labels]
    p = np.clip(p_true, clip, 1.0 - clip)
    return np.log(p) - np.log1p(-p)


def _gaussian_logpdf(x: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    """Log N(x; mu, sigma^2) — the additive constant cancels in the LiRA ratio
    but is kept for an interpretable per-term value."""
    return (-0.5 * np.log(2.0 * np.pi) - np.log(sigma)
            - 0.5 * ((x - mu) / sigma) ** 2)


def lira_scores(shadow_phi: np.ndarray, member_mask: np.ndarray,
                target_phi: np.ndarray, *,
                var_floor: float = DEFAULT_VAR_FLOOR,
                ) -> tuple[np.ndarray, int]:
    """Per-example online LiRA statistic (global-variance variant).

    For audit example ``i`` with target confidence ``t_i``::

        score_i = log N(t_i; mu_in_i,  sigma_in)
                - log N(t_i; mu_out_i, sigma_out)

    where ``mu_in_i`` / ``mu_out_i`` are the means of ``shadow_phi[:, i]`` over
    shadows where ``i`` was IN / OUT, and ``sigma_in`` / ``sigma_out`` are the
    pooled (global) IN / OUT standard deviations. Higher score = more
    member-like. Missing-side means fall back to the global side mean.

    Args:
        shadow_phi: ``[S, N]`` phi of each audit example under each shadow.
        member_mask: ``[S, N]`` bool, True where the example was IN the shadow.
        target_phi: ``[N]`` phi of each audit example under the target model.
        var_floor: floor on the pooled IN / OUT standard deviations.

    Returns:
        ``(scores [N] float64, n_mean_fallback int)`` — ``n_mean_fallback`` is
        the count of audit examples that used a global-mean fallback on at
        least one side (IN or OUT empty across the whole ensemble).
    """
    shadow_phi = np.asarray(shadow_phi, dtype=np.float64)
    member_mask = np.asarray(member_mask, dtype=bool)
    target_phi = np.asarray(target_phi, dtype=np.float64).ravel()
    s, n = shadow_phi.shape
    if member_mask.shape != (s, n):
        raise ValueError(f"member_mask {member_mask.shape} != shadow_phi {(s, n)}")
    if target_phi.shape != (n,):
        raise ValueError(f"target_phi {target_phi.shape} != ({n},)")

    in_all = shadow_phi[member_mask]            # 1-D: all IN entries, any example
    out_all = shadow_phi[~member_mask]          # 1-D: all OUT entries
    global_in_mu = float(in_all.mean()) if in_all.size else 0.0
    global_out_mu = float(out_all.mean()) if out_all.size else 0.0
    sigma_in = max(float(in_all.std()) if in_all.size > 1 else 0.0, var_floor)
    sigma_out = max(float(out_all.std()) if out_all.size > 1 else 0.0, var_floor)

    mu_in = np.empty(n, dtype=np.float64)
    mu_out = np.empty(n, dtype=np.float64)
    n_fallback = 0
    for i in range(n):
        in_i = shadow_phi[member_mask[:, i], i]
        out_i = shadow_phi[~member_mask[:, i], i]
        if in_i.size:
            mu_in[i] = in_i.mean()
        else:
            mu_in[i] = global_in_mu
            n_fallback += 1
        if out_i.size:
            mu_out[i] = out_i.mean()
        else:
            mu_out[i] = global_out_mu
            n_fallback += 1 if in_i.size else 0  # don't double-count one example

    scores = (_gaussian_logpdf(target_phi, mu_in, sigma_in)
              - _gaussian_logpdf(target_phi, mu_out, sigma_out))
    return scores, int(n_fallback)
