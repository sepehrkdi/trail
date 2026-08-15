"""Split fingerprinting.

Each split's identity is ``hash(dataset_id, ordered_example_ids,
transform_repr)`` via trail.core.hashing.dataset_fingerprint; the
fingerprint participates in every cache key and in provenance. Opaque
datasets (no index chain, no length) degrade to a class-name+length hash
with an explicit warning — never silently.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset

from trail.core.hashing import dataset_fingerprint, sha256_bytes

logger = logging.getLogger("trail.data.fingerprint")


def transform_repr(transform: Any) -> str:
    """Normalized single-line repr of a transform pipeline.

    torchvision reprs already embed the Normalize constants (mean/std), which
    is what makes hidden-normalization drift auto-invalidate L1 caches (spec
); this function only collapses whitespace so the repr is stable
    across torchvision's multi-line formatting.
    """
    if transform is None:
        return "none"
    return " ".join(repr(transform).split())


def transform_sha(transform: Any) -> str:
    """SHA-256 of the normalized transform repr."""
    return sha256_bytes(transform_repr(transform).encode())


def _base_dataset(dataset: Dataset) -> Dataset:
    """Unwrap a Subset chain to the underlying base dataset."""
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def loader_indices(loader: DataLoader | Dataset) -> np.ndarray | None:
    """Ordered base-dataset example ids served by ``loader``.

    Walks ``torch.utils.data.Subset`` chains, composing ``.indices`` outward-in
    (``Subset(Subset(base, I1), I2)`` -> ``I1[I2]``). A plain sized dataset
    yields ``arange(len)``; an opaque dataset (no length) yields None.
    """
    dataset = loader.dataset if isinstance(loader, DataLoader) else loader
    idx: np.ndarray | None = None
    while isinstance(dataset, Subset):
        layer = np.asarray(dataset.indices, dtype=np.int64)
        idx = layer if idx is None else layer[idx]
        dataset = dataset.dataset
    if idx is not None:
        return idx
    try:
        return np.arange(len(dataset), dtype=np.int64)  # type: ignore[arg-type]
    except TypeError:
        return None


def split_fingerprint(dataset_id: str, loader: DataLoader,
                      indices: np.ndarray | None = None) -> tuple[str, str | None]:
    """Fingerprint one split: ``(fingerprint, warning_or_None)``.

    Uses ``core.hashing.dataset_fingerprint(dataset_id, ordered_ids,
    transform_repr)``. If no index chain can be recovered (opaque dataset),
    falls back to a hash of (class name, length, transform repr) and returns
    a warning string for the report's ``warnings`` list; warning is None on
    the normal path.
    """
    dataset = loader.dataset
    trepr = transform_repr(getattr(_base_dataset(dataset), "transform", None))
    if indices is None:
        indices = loader_indices(loader)
    if indices is not None:
        return dataset_fingerprint(dataset_id, indices, trepr), None

    try:
        n: int = len(dataset)  # type: ignore[arg-type]
    except TypeError:
        n = -1
    cls = type(dataset).__name__
    fallback = sha256_bytes(f"opaque:{dataset_id}:{cls}:{n}:{trepr}".encode())
    warning = (
        f"split fingerprint fallback for dataset_id={dataset_id!r}: opaque "
        f"dataset {cls} (len={n}); identity hash uses class name + length "
        f"only — per-example cache reuse across runs is NOT guaranteed")
    logger.warning(warning)
    return fallback, warning


def check_disjoint(ids: Mapping[str, np.ndarray] | None) -> list[str]:
    """Warn (never error) on forget/retain example-id overlap.

    Some legitimate protocols (sub-class forgetting with shared augmentation
    pools) trip naive disjointness checks, so violations downgrade to report
    warnings. Returns a list of warning strings (empty when disjoint or when
    ids are unavailable).
    """
    warnings: list[str] = []
    if not ids:
        return warnings
    forget, retain = ids.get("forget"), ids.get("retain")
    if forget is None or retain is None:
        return warnings
    overlap = np.intersect1d(np.asarray(forget), np.asarray(retain))
    if overlap.size:
        msg = (
            f"forget/retain splits overlap on {int(overlap.size)} example ids "
            f"(first 5: {overlap[:5].tolist()}); proceeding "
            f"(warn, not error)")
        logger.warning(msg)
        warnings.append(msg)
    return warnings
