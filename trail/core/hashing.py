"""Hash utilities behind provenance (G3) and cache keys (G7)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace drift, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def _default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not canonically serializable: {type(o)}")


def config_hash(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode())[:16]


def dataset_fingerprint(dataset_id: str,
                        indices: Sequence[int] | np.ndarray,
                        transform_repr: str) -> str:
    """Split identity = dataset name + ordered example ids + preprocessing repr.

    The transform repr's participation is what makes hidden-preprocessing
    changes auto-invalidate L1 caches.
    """
    idx = np.asarray(indices, dtype=np.int64)
    h = hashlib.sha256()
    h.update(dataset_id.encode())
    h.update(b"|")
    h.update(idx.tobytes())
    h.update(b"|")
    h.update(transform_repr.encode())
    return h.hexdigest()
