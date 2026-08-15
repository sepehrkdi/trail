"""EvalContext — Layer 1 orchestration state.

The memoizing facade metrics receive: probes (L1-cached), references, named
seed substreams, loaders/views, and the provenance side-channel. Metrics never
touch checkpoints, raw loaders, or RNGs directly.

Model-mutation rule for metric authors: ``ctx.model(role)`` returns the
memoized shared instance — NEVER mutate it. ``copy.deepcopy`` the module
before fine-tuning it (relearning attack).
"""
from __future__ import annotations

import logging
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import torch

from trail.core import seeding
from trail.core.cache import Cache, l1_key, l2_key
from trail.core.errors import (
    InputModeError,
    MissingReference,
    RequestError,
    SplitNotAvailable,
)
from trail.core.hashing import config_hash, sha256_bytes, sha256_file
from trail.core.report import Provenance
from trail.core.types import (
    PROBE_VERSION,
    ROLES,
    EnsembleMargins,
    Reference,
    ShadowStats,
    SplitOutputs,
)

if TYPE_CHECKING:  # pragma: no cover
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    from trail.adapters.base import ModalityAdapter
    from trail.core.request import EvalRequest

logger = logging.getLogger("trail.context")

_UNSET = object()


@dataclass
class ScopeStats:
    """Per-metric scope measurements; the runner copies them into MetricResult."""

    name: str
    cost_s: float
    peak_mem_mb: float | None
    cache_hit: bool


def _resolve_device(requested: str) -> torch.device:
    """Map the runtime device string ("auto" picks cuda when available)."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


class EvalContext:
    """Orchestration state for one evaluation request."""

    def __init__(self, request: "EvalRequest", adapter: "ModalityAdapter",
                 cache: Cache) -> None:
        self.request = request
        self.adapter = adapter
        self.cache = cache
        self.task: str = request.task
        self.mode: str = request.mode
        self.seed: int = request.seed
        self.hp = request.hp
        self.input_mode: str = request.input_mode
        self.device: torch.device = _resolve_device(request.runtime.device)
        self.warnings: list[str] = []
        self.wandb_run_id: str | None = None
        #: Live W&B session (set by the runner). Metrics that train (the LiRA
        #: shadow ensemble) log incremental curves through ``wandb_log``; a
        #: no-op when tracking is off, so callers never branch on it.
        self.wandb_session: Any | None = None
        self.last_scope: ScopeStats | None = None
        # memoization state
        self._data_bundle: Any = None
        self._derived_splits: Any = _UNSET
        self._canonical: dict[str, "DataLoader"] = {}
        self._models: dict[str, "nn.Module"] = {}
        self._outputs: dict[tuple[str, str], SplitOutputs] = {}
        self._ckpt_sha: dict[str, str] = {}
        self._split_fps: dict[str, str] = {}
        self._preprocessing: dict[str, Any] = {}
        self._attack_manifests: dict[str, Any] = {}
        self._stamps: dict[str, Any] = {}
        #: F5 artifact side-channel: buffered (heavy) artifact requests, flushed
        #: AFTER provenance is assembled. The arrays live here, never in _stamps.
        self._artifact_requests: list[dict] = []
        self._gold: Reference | None = None
        self._shadow_stats: ShadowStats | None = None
        self._ensembles: dict[str, EnsembleMargins] = {}  # KLoM (M68) ckpt ensembles
        self._relearn_cache: dict[str, dict[str, dict[str, Any]]] = {}
        self._disjoint_checked = False
        self._current_metric: str | None = None
        self._scope_cache_hit = False
        self._t0 = time.perf_counter()

    # ------------------------------------------------------------------ data

    def _bundle(self) -> Any:
        """Resolve request.data to a SplitBundle (memoized; model-in only)."""
        if self.input_mode != "model":
            raise InputModeError("data loaders are not available under an "
                                 "outputs-in request")
        if self._data_bundle is None:
            from trail.data.specs import DatasetSpec, resolve_dataset_spec
            data = self.request.data
            if isinstance(data, DatasetSpec):
                self._data_bundle = resolve_dataset_spec(
                    data,
                    batch_size=self.request.runtime.batch_size or 256,
                    num_workers=self.request.runtime.num_workers,
                    base_seed=self.seed,
                )
            else:
                self._data_bundle = data
        return self._data_bundle

    def _derived(self) -> dict[str, "DataLoader"] | None:
        """Adapter-derived test partitions (forget_test/retain_test), memoized."""
        if self._derived_splits is _UNSET:
            self._derived_splits = self.adapter.derived_test_splits(
                self._raw_loader("test"), self.mode)
        return self._derived_splits

    def _raw_loader(self, split: str) -> "DataLoader":
        """User/spec-built loader for a split, before canonicalization."""
        bundle = self._bundle()
        loader = getattr(bundle, split, None)
        if loader is not None:
            return loader
        if split in ("forget_test", "retain_test"):
            derived = self._derived()
            if derived and split in derived:
                return derived[split]
            raise SplitNotAvailable(split, self.mode)
        raise RequestError(f"split {split!r} is not present in the data bundle")

    def loader(self, split: str, canonical: bool = True) -> "DataLoader":
        """Evaluation loader for ``split``; canonical view by default."""
        raw = self._raw_loader(split)
        if not canonical:
            return raw
        if split not in self._canonical:
            view, manifest = self.adapter.canonical_eval_view(
                raw, seed=self.seed,
                num_workers=self.request.runtime.num_workers)
            # Fingerprint + manifest BEFORE memoizing the view: a failed
            # materialization must leave no partial state, so the real
            # exception repeats on retry instead of a downstream KeyError
            # on _split_fps (atomic memoization).
            from trail.data.fingerprint import split_fingerprint
            fp, warn = split_fingerprint(self._dataset_id(), view)
            self._split_fps[split] = fp
            if warn is not None:
                self.warnings.append(warn)
            if manifest:
                self._record_split_manifest(split, manifest)  # G6
            self._canonical[split] = view
            self._maybe_check_disjoint()
            logger.debug("canonical view for %s materialized (fp=%s)",
                         split, fp[:12])
        return self._canonical[split]

    #: per-split manifest keys hoisted to the preprocessing top level while
    #: they stay uniform across all materialized splits (G6 shared view).
    _SHARED_MANIFEST_KEYS = ("aug_stripped", "normalization", "loader_order",
                             "dtype")

    def _record_split_manifest(self, split: str, manifest: dict) -> None:
        """Store a split's preprocessing manifest without last-split-wins
        overwrites (G6): the full entry lands under
        ``preprocessing['splits'][split]``; shared keys are re-derived at the
        top level while uniform and removed once any split diverges.
        ``bn_forced_eval`` is excluded — it is probe-owned and stamped at the
        top level by :meth:`_absorb_probe_flags`."""
        entry = dict(manifest)
        entry.pop("bn_forced_eval", None)
        self._preprocessing.setdefault("splits", {})[split] = entry
        self._preprocessing.setdefault("bn_forced_eval", False)
        per_split = list(self._preprocessing["splits"].values())
        for key in self._SHARED_MANIFEST_KEYS:
            values = [m.get(key) for m in per_split]
            if all(v == values[0] for v in values[1:]):
                self._preprocessing[key] = values[0]
            else:
                self._preprocessing.pop(key, None)

    def _absorb_probe_flags(self, model: "nn.Module") -> None:
        """Copy ``forward_stats`` disclosure attributes off the probed model
        into the preprocessing manifest (G6): ``bn_forced_eval`` accumulates
        with OR across probes; an OOM-halved retry is recorded once."""
        if getattr(model, "_trail_bn_forced_eval", False):
            self._preprocessing["bn_forced_eval"] = True
        if getattr(model, "_trail_oom_batch_halved", False):
            self._preprocessing["oom_batch_halved"] = True

    def dataset(self, split: str) -> "Dataset":
        """Canonical (augmentation-stripped) dataset view of ``split``."""
        return self.loader(split).dataset

    def train_view(self, split: str) -> "Dataset":
        """Train-transform view of ``split`` — for attack fine-tuning only.

        Resolution order: the bundle's own ``train_view(split)`` hook; for
        spec-resolved data, ``specs.train_view_dataset`` over the bundle's
        original-train ids (the documented augmented attack view — the raw
        spec-resolved dataset is the canonical aug-stripped one, which would
        silently change the attack distribution); otherwise the raw loader's
        dataset (which carries the user's own train-time transforms),
        disclosed under ``preprocessing.train_view_fallback``.
        """
        raw = self._raw_loader(split)
        bundle = self._bundle()
        hook = getattr(bundle, "train_view", None)
        if callable(hook):
            return hook(split)
        from trail.data.specs import DatasetSpec, train_view_dataset
        data = self.request.data
        ids = getattr(bundle, "ids", None) or {}
        if (isinstance(data, DatasetSpec) and split in ("forget", "retain")
                and split in ids):
            return train_view_dataset(data, ids[split])
        logger.warning(
            "train_view(%r): bundle exposes no train_view hook; attack "
            "fine-tuning uses the raw loader's dataset as-is", split)
        fallback = self._preprocessing.setdefault("train_view_fallback", [])
        if split not in fallback:
            fallback.append(split)
        return raw.dataset

    def _maybe_check_disjoint(self) -> None:
        """Best-effort forget/retain index disjointness check (warns)."""
        if self._disjoint_checked:
            return
        if "forget" not in self._canonical or "retain" not in self._canonical:
            return
        self._disjoint_checked = True
        try:
            f_idx = getattr(self._canonical["forget"].dataset, "indices", None)
            r_idx = getattr(self._canonical["retain"].dataset, "indices", None)
            if f_idx is None or r_idx is None:
                return
            overlap = set(map(int, f_idx)) & set(map(int, r_idx))
            if overlap:
                msg = (f"forget/retain splits share {len(overlap)} indices; "
                       "legitimate for some protocols (sub-class forgetting) "
                       "but verify this is intended")
                self.warnings.append(msg)
                logger.warning(msg)
        except Exception:  # diagnostics only — never fail the run for this
            logger.debug("disjointness check inapplicable", exc_info=True)

    # ------------------------------------------------------------ checkpoints

    def _checkpoint_path(self, role: str) -> str:
        """Path for a role; raises MissingReference with the role's skip code."""
        ck = self.request.checkpoints
        if role == "unlearned":
            return ck.unlearned
        if role == "original":
            if ck.original:
                return ck.original
            raise MissingReference(
                "missing_role:original",
                "supply checkpoints.original to unlock original-relative metrics")
        if role == "gold":
            # A supplied path always unlocks gold; hp.references.gold='off'
            # only means "do not BUILD".
            if ck.gold:
                return ck.gold
            g = self.hp.references.gold
            if isinstance(g, str) and g and g not in ("off", "build"):
                return g
            if g == "build":
                raise MissingReference(
                    "reference_disabled:gold",
                    "gold building is not implemented in v0.1; "
                    "supply a checkpoint path")
            raise MissingReference(
                "reference_disabled:gold",
                "hp.references.gold='off'; supply a gold checkpoint or set 'build'")
        raise ValueError(f"unknown checkpoint role {role!r}")

    def _ckpt_sha256(self, role: str) -> str:
        """SHA-256 of the role's checkpoint file (computed at first touch)."""
        if role not in self._ckpt_sha:
            self._ckpt_sha[role] = sha256_file(self._checkpoint_path(role))
        return self._ckpt_sha[role]

    def model(self, role: str) -> "nn.Module":
        """Memoized runnable model for ``role``. NEVER mutate the returned
        module — ``copy.deepcopy`` it before fine-tuning."""
        if self.input_mode != "model":
            raise InputModeError(
                f"model({role!r}) requires a model-in request; this request "
                "supplies outputs only")
        if role not in self._models:
            path = self._checkpoint_path(role)
            sha = self._ckpt_sha256(role)
            logger.info("loading checkpoint role=%s sha=%s", role, sha[:12])
            model = self.adapter.load_checkpoint(path, self.device)
            # G6/: the adapter records the applied unwrapping recipe on
            # the model; the orchestration layer owns the manifest row.
            fmt = getattr(model, "_trail_ckpt_format", None)
            if fmt is not None:
                self._preprocessing.setdefault("ckpt_format", {})[role] = fmt
            self._models[role] = model
        return self._models[role]

    # ---------------------------------------------------------------- probes

    def _probe_cfg_hash(self) -> str:
        # F4: probe_layers joins the L1 key so a different captured layer set
        # cannot serve a stale entry (G7). Sourced from the adapter (modality-
        # blind here); absent -> omitted, preserving the pre-F4 key shape for
        # adapters that expose no probe_cfg.
        cfg: dict[str, Any] = {"dtype": "fp32", "probe_version": PROBE_VERSION}
        probe_cfg = getattr(self.adapter, "probe_cfg", None)
        if callable(probe_cfg):
            layers = probe_cfg().get("probe_layers")
            if layers is not None:
                cfg["probe_layers"] = list(layers)
        return config_hash(cfg)

    def outputs(self, role: str, split: str) -> SplitOutputs:
        """Per-example outputs for (role, split).

        Model-in: L1-memoized canonical probe via ``adapter.forward_stats``.
        Outputs-in: validated payload parse. Raises MissingReference with the
        role-appropriate skip code when the role is absent; SplitNotAvailable
        for splits that are empty by design in this mode.
        """
        key = (role, split)
        if key in self._outputs:
            if self._current_metric is not None:
                self._scope_cache_hit = True
            return self._outputs[key]

        if self.input_mode == "outputs":
            payload = self.request.payload_for(role, split)
            if payload is None:
                if role == "unlearned":
                    # the required splits are enforced at request validation;
                    # an absent optional split is empty by design in this mode
                    raise SplitNotAvailable(split, self.mode)
                raise MissingReference(
                    f"missing_role:{role}",
                    f"outputs-in request lacks {role}_outputs[{split!r}]")
            out = self.adapter.validate_outputs_payload(payload)
            if role == "unlearned" and split not in self._split_fps:
                # payload-content fingerprint anchors outputs-in provenance
                self._split_fps[split] = sha256_bytes(
                    out.losses.tobytes() + out.targets.tobytes())
            self._outputs[key] = out
            return out

        # model-in: canonical loader first (materializes the split fingerprint)
        loader = self.loader(split)
        ckey = l1_key(self._ckpt_sha256(role), self._split_fps[split],
                      self.seed, self._probe_cfg_hash())
        cached = self.cache.get_outputs(ckey)
        if cached is not None:
            self.cache.record_hit("L1", f"{role}/{split}")
            if self._current_metric is not None:
                self._scope_cache_hit = True
            logger.info("probe %s/%s served from L1 cache", role, split)
            self._outputs[key] = cached
            return cached
        logger.info("probing %s/%s (n via canonical loader)", role, split)
        model = self.model(role)
        out = self.adapter.forward_stats(
            model, loader,
            seed=self.seed_for(f"probe:{role}:{split}"), device=self.device)
        self._absorb_probe_flags(model)  # bn_forced_eval / oom_batch_halved
        if not self.cache.cfg.readonly:
            self.cache.put_outputs(ckey, out)
        self._outputs[key] = out
        return out

    def probe_external(self, role: str, loader: "DataLoader", *,
                       seed_name: str) -> SplitOutputs:
        """UNCACHED probe of ``role``'s model on an ARBITRARY external loader
        (Phase 5; e.g. a downstream-transfer dataset for knn_transfer). Accepts
        ``role='gold'``. Returns SplitOutputs (features included) over the
        adapter's canonical (aug-stripped) view of the loader. Not cached — the
        downstream set is outside the L1 split-identity space."""
        if self.input_mode != "model":
            raise InputModeError("probe_external() requires a model-in request")
        model = self.model(role)  # loads unlearned/gold/original
        canonical, _ = self.adapter.canonical_eval_view(
            loader, seed=self.seed,
            num_workers=self.request.runtime.num_workers)
        out = self.adapter.forward_stats(
            model, canonical, seed=self.seed_for(seed_name), device=self.device)
        self._absorb_probe_flags(model)
        return out

    def probe_model(self, model: "nn.Module", split: str, *,
                    seed_name: str) -> SplitOutputs:
        """UNCACHED probe of an arbitrary model (e.g. a relearned
        copy) over the split's canonical loader, under a named substream."""
        if self.input_mode != "model":
            raise InputModeError("probe_model() requires a model-in request")
        out = self.adapter.forward_stats(
            model, self.loader(split),
            seed=self.seed_for(seed_name), device=self.device)
        self._absorb_probe_flags(model)
        return out

    # ------------------------------------------------------------ references

    def _dataset_id(self) -> str:
        from trail.data.specs import DatasetSpec
        data = self.request.data
        if isinstance(data, DatasetSpec):
            return getattr(data, "name", "dataset")
        return getattr(data, "dataset_id", "user_bundle")

    def _split_params_hash(self) -> str:
        """Stable split-construction identity for the L2 gold cache key.

        DatasetSpec data hashes the full spec; raw bundles hash dataset_id +
        mode + per-split ordered-id digests when available — NEVER the
        transient ``_split_fps`` memoization dict, whose contents depend on
        which splits happened to be materialized before gold() was called.
        """
        import numpy as np

        from trail.data.specs import DatasetSpec
        data = self.request.data
        if isinstance(data, DatasetSpec):
            return config_hash(data.model_dump())
        payload: dict[str, Any] = {
            "dataset_id": self._dataset_id(),
            "mode": getattr(data, "mode", self.mode),
        }
        ids = getattr(data, "ids", None)
        if ids:
            payload["split_ids_sha"] = {
                str(k): sha256_bytes(np.ascontiguousarray(
                    np.asarray(v, dtype=np.int64)).tobytes())
                for k, v in sorted(ids.items())}
        return config_hash(payload)

    def gold(self) -> Reference:
        """Resolved gold (retrained-from-scratch) reference.

        A supplied checkpoint path (request.checkpoints.gold, or a path in
        hp.references.gold) always unlocks gold — "off" only disables BUILDING.
        """
        if self._gold is not None:
            return self._gold
        if self.input_mode == "outputs":
            raise MissingReference(
                "missing_role:gold",
                "gold() returns a checkpoint reference and is model-in only; "
                "outputs-in requests supply gold via gold_outputs payloads")
        path = self._checkpoint_path("gold")  # raises with the right skip code
        self._gold = Reference(
            kind="gold", path=path, sha256=self._ckpt_sha256("gold"),
            source="supplied",
            cache_key=l2_key(self._dataset_id(), self.mode,
                             self._split_params_hash(), self.seed))
        logger.info("gold reference resolved (sha=%s)", self._gold.sha256[:12])
        return self._gold

    def shadow_stats(self) -> ShadowStats:
        """Shadow ensemble for the opt-in LiRA tier (M9), built or L3-cached.

        Raises MissingReference (``reference_disabled:shadow``) when
        ``hp.references.shadow == 0`` so the metric records the skip. When
        ``shadow > 0`` this trains the ensemble (expensive — opt-in only) via
        references/shadow.py and memoizes it for the rest of the panel.
        """
        if self._shadow_stats is not None:
            return self._shadow_stats
        # Lazy import: references.shadow pulls in attacks/adapters that import
        # this module, so importing at call time avoids the cycle.
        from trail.references.shadow import build_shadow_stats
        self._shadow_stats = build_shadow_stats(self)
        return self._shadow_stats

    def ensemble_margins(self, kind: str) -> EnsembleMargins:
        """KLoM (M68) checkpoint-ensemble margins for ``kind`` ('gold' |
        'unlearned'), built or L3-cached, then memoized for the rest of the
        panel. Raises MissingReference (``reference_disabled:gold``) when the
        corresponding ``hp.references.<kind>_ensemble`` list is empty, so the
        metric records the skip. Model-in only (loads + probes checkpoints)."""
        if kind in self._ensembles:
            return self._ensembles[kind]
        # Lazy import mirrors shadow_stats: avoid the references<->context cycle.
        from trail.references.gold_ensemble import build_ensemble_margins
        self._ensembles[kind] = build_ensemble_margins(self, kind)
        return self._ensembles[kind]

    # ------------------------------------------------------ seeds, stamps, misc

    def seed_for(self, name: str) -> int:
        """Named seed substream derived from the request seed (G2)."""
        return seeding.seed_for(self.seed, name)

    def stamp(self, dotted_key: str, value: Any) -> None:
        """Record a provenance side-channel value under a dotted key, e.g.
        ``ctx.stamp("relearning.relearn_forget.curves", history)``."""
        node = self._stamps
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"stamp key collision at {part!r} in "
                                 f"{dotted_key!r}")
        node[parts[-1]] = value

    def emit_artifact(self, name: str, *, kind: str = "npz",
                      arrays: dict | None = None,
                      render: Any | None = None, method: str,
                      params: dict | None = None,
                      seed: int | None = None) -> None:
        """Buffer a heavy artifact for out-of-band emission (F5).

        ``kind='npz'`` carries ``arrays`` ({name: ndarray}); ``kind in
        {'png','svg'}`` carries a ``render(pyplot) -> Figure`` callable. The
        heavy payload is held HERE — never routed through ``ctx.stamp`` / the
        report ``_plain`` serializer — and emitted by the runner after
        ``provenance.validate_complete`` (gated on ``RuntimeConfig.plots`` +
        ``CacheConfig.readonly``). A no-op at scoring time beyond the append."""
        self._artifact_requests.append({
            "name": name, "kind": kind, "arrays": arrays, "render": render,
            "method": method, "params": dict(params or {}),
            "seed": self.seed if seed is None else seed})

    def flush_artifacts(self, emitter: Any, provenance: Any) -> list:
        """Emit every buffered artifact request through ``emitter`` (which
        validates provenance + applies the enabled/readonly gates). Returns the
        produced descriptors (None results — gated off — are dropped)."""
        descriptors = []
        for req in self._artifact_requests:
            if req["kind"] == "npz":
                desc = emitter.emit_npz(
                    req["name"], req["arrays"] or {}, method=req["method"],
                    params=req["params"], provenance=provenance, seed=req["seed"])
            else:
                desc = emitter.emit_figure(
                    req["name"], req["render"], kind=req["kind"],
                    method=req["method"], params=req["params"],
                    provenance=provenance, seed=req["seed"])
            if desc is not None:
                descriptors.append(desc)
        return descriptors

    def wandb_log(self, metrics: dict, step: int | None = None) -> None:
        """Log incremental training curves to the live W&B run (no-op when
        tracking is off). Used by the LiRA shadow trainer so a user can watch
        per-shadow / per-epoch progress live; the final report scalars are
        logged separately by the runner."""
        session = self.wandb_session
        if session is not None and getattr(session, "active", False):
            session.log(metrics, step=step)

    def self_reported(self) -> dict | None:
        """The method author's self-reported cost dict, if supplied."""
        return getattr(self.request, "self_reported_cost", None)

    @contextmanager
    def scoped(self, name: str) -> Iterator["EvalContext"]:
        """Per-metric scope: wall timer, CUDA peak-memory window, cache-hit
        flag, and the active-metric marker. The runner copies ``last_scope``
        into the MetricResult after the metric function returns."""
        self._current_metric = name
        self._scope_cache_hit = False
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        t0 = time.perf_counter()
        try:
            yield self
        finally:
            elapsed = time.perf_counter() - t0
            peak = None
            if self.device.type == "cuda":
                peak = float(torch.cuda.max_memory_allocated(self.device)) / 2**20
            self.last_scope = ScopeStats(name=name, cost_s=elapsed,
                                         peak_mem_mb=peak,
                                         cache_hit=self._scope_cache_hit)
            self._current_metric = None

    # -------------------------------------------------------------- provenance

    def provenance(self) -> Provenance:
        """Assemble the reproducibility manifest (G3) from accumulated state."""
        import trail  # lazy: __init__ lazily re-exports from this module's peers

        code_git_sha: str | None = None
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                capture_output=True, text=True, timeout=10)
            if proc.returncode == 0:
                code_git_sha = proc.stdout.strip()
        except Exception:
            code_git_sha = None

        preprocessing = dict(self._preprocessing)
        # Phase 5 (G3): record the architecture in the manifest when the adapter
        # exposes one (classification), so cross-arch runs are distinguishable.
        arch = getattr(self.adapter, "arch", None)
        if arch is not None:
            preprocessing.setdefault("arch", arch)
        references: dict[str, Any] = {}
        if self._gold is not None:
            references["gold"] = {"sha256": self._gold.sha256,
                                  "cache_key": self._gold.cache_key,
                                  "source": self._gold.source}
        attack_manifests = dict(self._attack_manifests)
        stamps = dict(self._stamps)
        for field_name, target in (("references", references),
                                   ("attack_manifests", attack_manifests),
                                   ("preprocessing", preprocessing)):
            extra = stamps.pop(field_name, None)
            if isinstance(extra, dict):
                target.update(extra)
        if stamps:  # never drop a stamp: leftovers land under preprocessing
            preprocessing.setdefault("stamps", {}).update(stamps)

        gpu_name = (torch.cuda.get_device_name(self.device)
                    if self.device.type == "cuda" else None)
        timestamp = datetime.now(timezone.utc).isoformat(
            timespec="seconds").replace("+00:00", "Z")

        # Provenance must record the SHA of every SUPPLIED checkpoint, even if
        # no metric ever loaded that role (e.g. `original` supplied but no
        # needs_original metric in the panel) — lazily hash the leftovers.
        if self.input_mode == "model":
            for role in ROLES:
                if role in self._ckpt_sha:
                    continue
                path = getattr(getattr(self.request, "checkpoints", None), role, None)
                if path:
                    try:
                        self._ckpt_sha[role] = sha256_file(path)
                    except OSError:  # pragma: no cover — surfaced as warning
                        self.warnings.append(
                            f"could not hash supplied {role} checkpoint: {path}")

        return Provenance(
            library_version=trail.__version__,
            code_git_sha=code_git_sha,
            checkpoint_sha256={role: self._ckpt_sha.get(role) for role in ROLES},
            dataset_fingerprints=dict(self._split_fps),
            preprocessing=preprocessing,
            references=references,
            attack_manifests=attack_manifests,
            cache_hits=list(self.cache.hits),
            wall_clock_s=time.perf_counter() - self._t0,
            device=str(self.device),
            gpu_name=gpu_name,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            wandb_run_id=self.wandb_run_id,
            timestamp=timestamp,
        )
