"""Per-example bootstrap confidence intervals (guarantee G5).

Every MetricResult carries a CI. Degenerate inputs (constant arrays, n==1
self-reported scalars) collapse to (value, value).
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np


def bootstrap_ci(per_example: np.ndarray,
                 stat_fn: Callable[[np.ndarray], float] | None = None,
                 *,
                 n: int = 1000,
                 alpha: float = 0.05,
                 rng: np.random.Generator) -> tuple[float, float]:
    """Percentile bootstrap of ``stat_fn`` (default: mean) over one array."""
    per_example = np.asarray(per_example)
    if per_example.size == 0:
        return (float("nan"), float("nan"))
    stat = stat_fn or (lambda a: float(np.mean(a)))
    if per_example.size == 1 or np.all(per_example == per_example.flat[0]):
        v = float(stat(per_example))
        return (v, v)
    size = len(per_example)
    stats = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, size, size=size)
        stats[i] = stat(per_example[idx])
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(lo), float(hi))


def bootstrap_ci_groups(groups: Mapping[str, np.ndarray],
                        stat_fn: Callable[[Mapping[str, np.ndarray]], float],
                        *,
                        n: int = 1000,
                        alpha: float = 0.05,
                        rng: np.random.Generator) -> tuple[float, float]:
    """Bootstrap a statistic over several independent per-example groups
    (each group resampled independently per iteration — e.g. M21's four splits,
    or paired pre/post arrays passed as one group of paired rows)."""
    groups = {k: np.asarray(v) for k, v in groups.items()}
    if any(v.size == 0 for v in groups.values()):
        return (float("nan"), float("nan"))
    stats = np.empty(n, dtype=np.float64)
    for i in range(n):
        resampled = {
            k: v[rng.integers(0, len(v), size=len(v))] for k, v in groups.items()
        }
        stats[i] = stat_fn(resampled)
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(lo), float(hi))
