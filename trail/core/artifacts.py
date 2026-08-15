"""Artifact subsystem (F5): an out-of-band emitter for heavy visual/array
artifacts (npz / png / svg). t-SNE coords, 2-D embeddings, and plots land here
in Tier 4; this module is the plumbing.

Design invariants:

* **Never CSV.** The aggregator (``trail.aggregate``) stays the ONLY CSV
  writer; this emitter writes only ``npz`` / ``png`` / ``svg`` and refuses any
  other kind. This intentionally relaxes the "numbers-not-plots" norm for visual
  artifacts, but the CSV chokepoint is preserved.
* **Scalar descriptor only.** Heavy ``[N, D]`` arrays go to disk; only a small
  :class:`ArtifactDescriptor` (path + sha256 + bytes + method + params +
  lib-versions + seed) flows into ``report.artifacts`` /
  ``provenance.artifact_sha256``. The arrays never pass through ``ctx.stamp`` /
  the report ``_plain`` serializer (which would explode the JSON).
* **Provenance gate first.** Emission calls ``provenance.validate_complete``
  BEFORE writing any bytes — an artifact is never written for a report that
  could not itself serialize (G3, strengthened).
* **Content-addressed, never timestamped.** Filenames are
  ``<name>-<sha16>.<kind>`` so identical content dedups and the path is
  reproducible across hosts/runs.
* **Gated off by default.** Emission is a no-op unless ``enabled`` (wired to
  ``RuntimeConfig.plots``, default False) and not ``readonly``. matplotlib /
  sklearn are lazy-imported behind the ``trail[plots]`` extra.
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from trail.core.hashing import sha256_bytes
from trail.core.report import _plain

logger = logging.getLogger("trail.artifacts")

#: The only artifact kinds this emitter writes. ``csv`` is deliberately absent —
#: the aggregator is the sole CSV writer.
ARTIFACT_KINDS: tuple[str, ...] = ("npz", "png", "svg")


def load_input_artifact(path: str, *, key: str | None = None,
                        expected_sha256: str | None = None) -> tuple:
    """Load an externally-produced INPUT artifact (Tier-2 attack family).

    Reads ``.npy`` / ``.npz`` with ``allow_pickle=False`` (never executes pickled
    code) and content-addresses the FILE bytes (sha256). For ``.npz``, ``key``
    selects an array (else the first). Returns ``(ndarray, sha256)``. The caller
    stamps the sha into ``Provenance.artifact_sha256``. Raises FileNotFoundError
    when the path is absent (the metric maps that to a ``missing_artifact`` skip);
    raises ValueError on a sha mismatch.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    raw = p.read_bytes()
    sha = sha256_bytes(raw)
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError(
            f"artifact {path}: sha256 {sha[:12]} != expected {expected_sha256[:12]}")
    if p.suffix == ".npz":
        with np.load(p, allow_pickle=False) as data:
            arr = np.asarray(data[key if key is not None else data.files[0]])
    else:
        arr = np.asarray(np.load(p, allow_pickle=False))
    return arr, sha


@dataclass(frozen=True)
class ArtifactDescriptor:
    """Scalar, ``_plain``-serializable record of one emitted artifact."""

    name: str
    kind: str
    path: str
    sha256: str
    n_bytes: int
    method: str
    params: dict = field(default_factory=dict)
    lib_versions: dict = field(default_factory=dict)
    seed: int = 0

    def to_dict(self) -> dict:
        """The descriptor as a plain dict for ``report.artifacts``."""
        return {
            "name": self.name, "kind": self.kind, "path": self.path,
            "sha256": self.sha256, "n_bytes": int(self.n_bytes),
            "method": self.method, "params": dict(self.params),
            "lib_versions": dict(self.lib_versions), "seed": int(self.seed),
        }


class ArtifactEmitter:
    """Writes content-addressed npz/png/svg artifacts; returns scalar descriptors.

    Args:
        out_dir: directory artifacts are written under (created on first write).
        enabled: master switch (wired to ``RuntimeConfig.plots``); disabled =
            every emit is a no-op returning None.
        readonly: when True, emit is a no-op (mirrors the cache readonly gate).
        seed: default seed stamped into descriptors.
        input_mode: passed to ``provenance.validate_complete`` (model vs outputs).
    """

    def __init__(self, out_dir: str | Path, *, enabled: bool = False,
                 readonly: bool = False, seed: int = 0,
                 input_mode: str = "model") -> None:
        self.out_dir = Path(out_dir)
        self.enabled = enabled
        self.readonly = readonly
        self.seed = int(seed)
        self.input_mode = input_mode

    def _active(self) -> bool:
        return self.enabled and not self.readonly

    def emit_npz(self, name: str, arrays: dict[str, np.ndarray], *,
                 method: str, params: dict, provenance: Any,
                 seed: int | None = None) -> ArtifactDescriptor | None:
        """Write ``arrays`` as a compressed ``.npz`` (e.g. raw embeddings + coords).

        No-op (returns None) when disabled/readonly. Calls
        ``provenance.validate_complete`` before writing (G3)."""
        if not self._active():
            return None
        provenance.validate_complete(self.input_mode)  # G3 gate, before any write
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        return self._write_bytes(name, "npz", buf.getvalue(),
                                 method=method, params=params, seed=seed)

    def emit_figure(self, name: str, render: Callable[[Any], Any], *,
                    kind: str = "png", method: str, params: dict,
                    provenance: Any, seed: int | None = None,
                    ) -> ArtifactDescriptor | None:
        """Render ``render(pyplot) -> Figure`` to png/svg (lazy matplotlib).

        No-op when disabled/readonly. Calls ``provenance.validate_complete``
        before importing matplotlib or writing (G3)."""
        if not self._active():
            return None
        if kind not in ("png", "svg"):
            raise ValueError(f"emit_figure kind must be 'png' or 'svg', got {kind!r}")
        provenance.validate_complete(self.input_mode)
        import matplotlib  # lazy: trail[plots]
        matplotlib.use("Agg")  # headless, deterministic backend
        import matplotlib.pyplot as plt

        fig = render(plt)
        buf = io.BytesIO()
        fig.savefig(buf, format=kind)
        plt.close(fig)
        return self._write_bytes(
            name, kind, buf.getvalue(), method=method, params=params, seed=seed,
            extra_libs={"matplotlib": matplotlib.__version__})

    def _write_bytes(self, name: str, kind: str, data: bytes, *,
                     method: str, params: dict, seed: int | None,
                     extra_libs: dict | None = None) -> ArtifactDescriptor:
        """Atomically write ``data`` to a content-addressed file; build a
        descriptor. The hard CSV/kind guard lives here (the only write path)."""
        if kind not in ARTIFACT_KINDS:
            raise ValueError(
                f"the artifact emitter writes only {ARTIFACT_KINDS}; refused "
                f"{kind!r} — the aggregator is the only CSV writer")
        sha = sha256_bytes(data)
        path = self.out_dir / f"{name}-{sha[:16]}.{kind}"  # content-addressed
        self.out_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.out_dir, suffix=".tmp")
        os.close(fd)
        try:
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        libs = {"numpy": np.__version__}
        if extra_libs:
            libs.update(extra_libs)
        logger.info("artifact %s -> %s (%d bytes)", name, path.name, len(data))
        return ArtifactDescriptor(
            name=name, kind=kind, path=str(path), sha256=sha, n_bytes=len(data),
            method=method, params=_plain(dict(params or {})), lib_versions=libs,
            seed=int(self.seed if seed is None else seed))
