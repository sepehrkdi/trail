"""Content-addressed cache, layers L1-L3 (guarantee G7).

Materialized in v0.1: L1 (forward-pass ``SplitOutputs``), L2 (gold model
blobs). L3 (shadow ensembles) has its key function reserved here so the key
discipline is fixed before the builder lands.

Key discipline:

* L2/L3 keys contain NO checkpoint hash — gold retrains and shadow ensembles
  are method-independent and amortize across every evaluated method.
* The optional L2 ``stage`` (F6) is split-recipe identity for a per-stage gold
  (the raw-bundle / sequential path), NEVER a checkpoint hash or round index —
  so the no-ckpt-hash invariant above is preserved. The DatasetSpec gold path
  encodes split identity through ``split_params_hash`` (context._split_params_hash)
  and passes no ``stage``, so its keys are byte-unchanged.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from trail.core.errors import TRAILError
from trail.core.hashing import canonical_json, sha256_bytes
from trail.core.types import SplitOutputs

if TYPE_CHECKING:  # pragma: no cover
    from trail.core.request import CacheConfig

logger = logging.getLogger("trail.cache")

LAYERS: tuple[str, ...] = ("L1", "L2", "L3")
_MATERIALIZED: tuple[str, ...] = ("L1", "L2", "L3")  # L3 = shadow ensembles (M9 LiRA)


def _key(layer: str, *fields: Any) -> str:
    """Content-address: sha256 of the canonical-JSON field tuple."""
    return sha256_bytes(canonical_json([layer, *fields]).encode())


def l1_key(ckpt_sha: str, split_fp: str, seed: int, probe_cfg_hash: str) -> str:
    """Forward-pass outputs: one probe per (checkpoint, split, seed, probe cfg)."""
    return _key("L1", ckpt_sha, split_fp, int(seed), probe_cfg_hash)


def l2_key(dataset_id: str, mode: str, split_params_hash: str, seed: int,
           *, stage: str | None = None) -> str:
    """Gold model A_r. NO checkpoint hash: gold is method-independent.

    ``stage`` (F6) distinguishes the per-stage golds of the raw-bundle /
    sequential path — it is the stage's split-recipe identity, never a
    checkpoint hash or round index. When ``stage is None`` (every Phase-0
    caller, incl. the DatasetSpec gold path) the key is byte-identical to the
    pre-F6 four-field key, so existing L2 entries stay valid.
    """
    if stage is None:
        return _key("L2", dataset_id, mode, split_params_hash, int(seed))
    return _key("L2", dataset_id, mode, split_params_hash, int(seed), str(stage))


def l3_key(dataset_id: str, seed: int, n_shadow: int) -> str:
    """Shadow ensemble (reserved in v0.1). NO checkpoint hash: method-independent."""
    return _key("L3", dataset_id, int(seed), int(n_shadow))


class Cache:
    """Filesystem store under ``cfg.dir`` with per-layer subdirectories.

    * ``cfg.readonly=True`` (CI): gets are served, puts raise.
    * ``cfg.disable`` (set of layer names): both gets and puts become no-ops.
    * Writes are atomic (tmp file + ``os.replace``).
    * ``hits`` records served entries for ``provenance.cache_hits``.
    """

    def __init__(self, cfg: "CacheConfig") -> None:
        self.cfg = cfg
        self.dir = Path(cfg.dir)
        self.hits: list[str] = []
        if not cfg.readonly:
            for layer in _MATERIALIZED:
                if layer not in cfg.disable:
                    (self.dir / layer).mkdir(parents=True, exist_ok=True)

    # -- shared plumbing ----------------------------------------------------

    def _enabled(self, layer: str) -> bool:
        if layer not in LAYERS:
            raise ValueError(f"unknown cache layer {layer!r}; known: {LAYERS}")
        return layer not in self.cfg.disable

    def _path(self, layer: str, key: str, suffix: str) -> Path:
        return self.dir / layer / f"{key}{suffix}"

    def _atomic_replace(self, write_fn, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        os.close(fd)
        try:
            write_fn(tmp)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def record_hit(self, layer: str, tag: str) -> None:
        """Record a served cache entry, e.g. ``record_hit("L1", "unlearned/forget")``."""
        self.hits.append(f"{layer}:{tag}")

    # -- L1: SplitOutputs <-> npz --------------------------------------------

    def get_outputs(self, key: str) -> SplitOutputs | None:
        """Load cached probe outputs; None on miss/disabled/corrupt entry."""
        if not self._enabled("L1"):
            return None
        path = self._path("L1", key, ".npz")
        if not path.exists():
            return None
        try:
            with np.load(path) as data:
                fm = {k[len("fm__"):]: data[k]
                      for k in data.files if k.startswith("fm__")}
                return SplitOutputs(
                    losses=data["losses"],
                    targets=data["targets"],
                    logits=data["logits"] if "logits" in data.files else None,
                    features=data["features"] if "features" in data.files else None,
                    n=int(data["n"]),
                    features_multi=fm or None,
                )
        except Exception as e:  # corrupt entry == miss, never a crash
            logger.warning("corrupt L1 entry %s (%s); treating as miss", path.name, e)
            return None

    def put_outputs(self, key: str, outputs: SplitOutputs) -> None:
        """Store probe outputs (compressed npz; None arrays omitted). Atomic."""
        if not self._enabled("L1"):
            return
        if self.cfg.readonly:
            raise TRAILError("cache is readonly: refusing put_outputs (L1)")
        arrays: dict[str, np.ndarray] = {
            "losses": outputs.losses,
            "targets": outputs.targets,
            "n": np.asarray(outputs.n, dtype=np.int64),
        }
        if outputs.logits is not None:
            arrays["logits"] = outputs.logits
        if outputs.features is not None:
            arrays["features"] = outputs.features
        # F4: per-layer features stored under an ``fm__<layer>`` member prefix
        # (reconstructed by member-name on load). G7: a cache hit must restore
        # the full SplitOutputs, features_multi included.
        if outputs.features_multi is not None:
            for name, arr in outputs.features_multi.items():
                arrays[f"fm__{name}"] = arr
        path = self._path("L1", key, ".npz")

        def _write(tmp: str) -> None:
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **arrays)

        self._atomic_replace(_write, path)
        logger.debug("L1 put %s (n=%d)", key[:12], outputs.n)

    # -- generic blobs (L2 in v0.1) -------------------------------------------

    def get_blob(self, layer: str, key: str) -> Any | None:
        """Load an arbitrary blob; None on miss/disabled/corrupt entry."""
        if not self._enabled(layer):
            return None
        path = self._path(layer, key, ".pt")
        if not path.exists():
            return None
        import torch  # local import keeps cache importable without torch
        try:
            # weights_only=False is acceptable: only trail itself writes here.
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning("corrupt %s entry %s (%s); treating as miss",
                           layer, path.name, e)
            return None

    def put_blob(self, layer: str, key: str, obj: Any) -> None:
        """Store an arbitrary blob via ``torch.save``. Atomic; readonly raises."""
        if not self._enabled(layer):
            return
        if self.cfg.readonly:
            raise TRAILError(f"cache is readonly: refusing put_blob ({layer})")
        import torch
        path = self._path(layer, key, ".pt")
        self._atomic_replace(lambda tmp: torch.save(obj, tmp), path)
        logger.debug("%s put %s", layer, key[:12])
