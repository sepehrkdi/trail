"""Runner: feasibility pass, metric fan-out, report assembly.

The runner is modality-blind: all model/data knowledge lives behind the
adapter, all metric knowledge in the registry, all skip semantics in the
exceptions metrics raise (fail-soft, guarantee G4).
"""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

from trail.core.cache import Cache
from trail.core.context import EvalContext
from trail.core.errors import (
    MetricSkip,
    MissingReference,
    SplitNotAvailable,
    TRAILError,
)
from trail.core.registry import COST_RANK, METRIC_REGISTRY, MetricSpec, resolve
from trail.core.report import EvalReport, MetricResult, SkipInfo
from trail.core.request import (
    CacheConfig,
    CheckpointSet,
    EvalRequest,
    Hyperparams,
    LogConfig,
    ModelRequest,
    OutputsRequest,
    RuntimeConfig,
)

if TYPE_CHECKING:  # pragma: no cover
    from trail.adapters.base import ModalityAdapter
    from trail.data.specs import DatasetSpec, SplitBundle

logger = logging.getLogger("trail.runner")


@dataclass
class Plan:
    """Output of the feasibility pass: what runs, and why the rest does not."""

    runnable: list[MetricSpec] = field(default_factory=list)
    skipped: dict[str, SkipInfo] = field(default_factory=dict)


def _gold_skip(request: EvalRequest) -> SkipInfo | None:
    """Gold feasibility: None when gold is available, else the skip to record.

    A supplied checkpoint path (checkpoints.gold, outputs-in gold_outputs, or
    a path in hp.references.gold) always unlocks; 'off' only disables building.
    """
    if request.has_role("gold"):
        return None
    gold_hp = request.hp.references.gold
    if gold_hp == "off":
        return SkipInfo("reference_disabled:gold",
                        "hp.references.gold='off'; supply a gold checkpoint "
                        "or set 'build'")
    if gold_hp == "build":
        return SkipInfo("reference_disabled:gold",
                        "gold building is not implemented in v0.1; supply a "
                        "checkpoint path")
    if isinstance(gold_hp, str) and gold_hp:
        return None  # a path: supplied gold unlocks the tier
    return SkipInfo("missing_role:gold", "no gold checkpoint supplied")


def feasibility_pass(request: EvalRequest,
                     registry: dict[str, MetricSpec],
                     adapter: "ModalityAdapter") -> Plan:
    """Compute the runnable metric set and the skip ledger.

    Checks run in a fixed order so the recorded skip code is deterministic:
    modality -> mode -> input mode -> original -> gold -> shadow.
    """
    plan = Plan()
    for spec in resolve(request.metrics, registry, request.task, adapter):
        if request.task not in spec.modalities:
            plan.skipped[spec.name] = SkipInfo(
                "not_applicable_modality",
                f"{spec.name} applies to {sorted(spec.modalities)}, not "
                f"{request.task!r}")
            continue
        if spec.modes and request.mode not in spec.modes:
            plan.skipped[spec.name] = SkipInfo(
                "not_applicable_mode",
                f"{spec.name} applies to modes {sorted(spec.modes)}, not "
                f"{request.mode!r}")
            continue
        if request.input_mode not in spec.input_modes:
            plan.skipped[spec.name] = SkipInfo(
                "requires_model_in",
                f"{spec.name} must mutate/run a model; supply checkpoints "
                "(model-in) to unlock it")
            continue
        if spec.needs_original and not request.has_role("original"):
            plan.skipped[spec.name] = SkipInfo(
                "missing_role:original",
                "supply checkpoints.original (the pre-unlearning parent) to "
                f"unlock {spec.name}")
            continue
        if spec.needs_gold:
            skip = _gold_skip(request)
            if skip is not None:
                plan.skipped[spec.name] = skip
                continue
        if spec.needs_shadow and request.hp.references.shadow == 0:
            plan.skipped[spec.name] = SkipInfo(
                "reference_disabled:shadow",
                "hp.references.shadow=0; set shadow=8 to build the ensemble")
            continue
        plan.runnable.append(spec)
    logger.info("feasibility: %d runnable, %d skipped",
                len(plan.runnable), len(plan.skipped))
    return plan


def _start_wandb(request: EvalRequest) -> Any | None:
    """Start the tracking session; tracking failures never block evaluation."""
    try:
        from trail.io.wandb_hooks import WandbSession
    except ImportError as e:
        logger.warning("wandb hooks unavailable (%s); continuing untracked", e)
        return None
    try:
        summary = {
            "task": request.task,
            "mode": request.mode,
            "seed": request.seed,
            "input_mode": request.input_mode,
            "metrics": (request.metrics if isinstance(request.metrics, str)
                        else list(request.metrics)),
        }
        return WandbSession.start(request.log, summary)
    except Exception as e:
        logger.warning("wandb session start failed (%s); continuing untracked", e)
        return None


def _flush_artifacts(ctx: EvalContext, request: EvalRequest, prov) -> list[dict]:
    """Emit any buffered F5 artifacts after provenance assembly, gated on
    ``RuntimeConfig.plots`` (default off) + ``CacheConfig.readonly``. Stamps
    ``provenance.artifact_sha256`` and returns the scalar descriptors for
    ``report.artifacts``. A no-op (empty list) unless a metric emitted and
    plots are enabled — so the default scored panel is untouched."""
    if not getattr(request.runtime, "plots", False):
        return []
    if not getattr(ctx, "_artifact_requests", None):
        return []
    from pathlib import Path

    from trail.core.artifacts import ArtifactEmitter

    out_dir = Path(request.cache.dir) / "artifacts"
    emitter = ArtifactEmitter(
        out_dir, enabled=True, readonly=request.cache.readonly,
        seed=request.seed, input_mode=request.input_mode)
    descriptors = ctx.flush_artifacts(emitter, prov)
    prov.artifact_sha256 = {d.name: d.sha256 for d in descriptors}
    return [d.to_dict() for d in descriptors]


def run(request: EvalRequest) -> EvalReport:
    """The canonical core: execute one evaluation request end-to-end."""
    from trail.adapters import ADAPTERS
    from trail.core.request import check_disclosure
    # F2: construct via the per-adapter from_request classmethod (the blocker
    # fix) so the requested arch/dataset/num_classes actually reach the adapter,
    # instead of the old zero-arg ADAPTERS[task]() that pinned every run to the
    # construction-time defaults.
    adapter = ADAPTERS[request.task].from_request(request)
    # registration side-effects: importing populates METRIC_REGISTRY
    # deterministically
    importlib.import_module("trail.metrics")
    importlib.import_module("trail.attacks")

    # Disclosure contract: raises DisclosureError before any work if an
    # attack metric will actually run with an undisclosed recipe; otherwise
    # returns the per-knob disclosure block for the report.
    disclosure = check_disclosure(request)

    plan = feasibility_pass(request, METRIC_REGISTRY, adapter)
    ctx = EvalContext(request, adapter, Cache(request.cache))
    session = _start_wandb(request)
    if session is not None:
        ctx.wandb_run_id = getattr(session, "run_id", None)
        ctx.wandb_session = session  # live curves for metrics that train (LiRA)

    results: dict[str, dict[str, MetricResult]] = {}
    skipped: dict[str, SkipInfo] = dict(plan.skipped)
    try:
        ordered = sorted(plan.runnable,
                         key=lambda s: (COST_RANK[s.cost], s.name))
        for spec in ordered:
            logger.info("metric %s [%s/%s] begin",
                        spec.name, spec.category, spec.cost)
            result: MetricResult | None = None
            with ctx.scoped(spec.name):
                try:
                    result = spec.fn(ctx)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except SplitNotAvailable as e:
                    skipped[spec.name] = SkipInfo("not_applicable_mode", str(e))
                except MetricSkip as e:
                    skipped[spec.name] = SkipInfo(e.code, str(e))
                except MissingReference as e:
                    skipped[spec.name] = SkipInfo(e.skip_code, str(e))
                except (TRAILError, RuntimeError, ValueError, KeyError,
                        IndexError, TypeError) as e:  # fail-soft boundary (G4)
                    # TRAILError covers RequestError/MetricError/etc.; the
                    # skip-coded subclasses are intercepted by the branches
                    # above, so only unmapped failures land here.
                    logger.exception("metric %s failed", spec.name)
                    skipped[spec.name] = SkipInfo(
                        "runtime_error", f"{type(e).__name__}: {e}")
            scope = ctx.last_scope
            if result is not None:
                result.cost_s = scope.cost_s
                result.peak_mem_mb = scope.peak_mem_mb
                result.cache_hit = scope.cache_hit
                results.setdefault(spec.category, {})[spec.name] = result
                logger.info("metric %s end value=%.4f cost=%.1fs cache=%s",
                            spec.name, result.value, scope.cost_s,
                            "hit" if scope.cache_hit else "miss")
            else:
                logger.info("metric %s skipped (%s) after %.1fs",
                            spec.name, skipped[spec.name].code, scope.cost_s)

        hp_dump = request.hp.model_dump()
        hp_dump["metric_overrides_applied"] = bool(request.hp.metric_overrides)
        prov = ctx.provenance()
        artifacts = _flush_artifacts(ctx, request, prov)  # F5 (gated; empty otherwise)
        report = EvalReport(
            task=request.task, mode=request.mode, seed=request.seed,
            input_mode=request.input_mode, metrics=results, skipped=skipped,
            warnings=list(ctx.warnings), hyperparams=hp_dump,
            disclosure=disclosure, provenance=prov, artifacts=artifacts)
        if session is not None:
            try:
                session.log_report(report)
            except Exception as e:
                logger.warning("wandb log_report failed: %s", e)
        return report
    finally:
        if session is not None:
            try:
                session.finish()
            except Exception as e:
                logger.warning("wandb finish failed: %s", e)


# ---------------------------------------------------------------------------
# Facades
# ---------------------------------------------------------------------------

def evaluate(data: "SplitBundle | DatasetSpec",
             seed: int,
             checkpoints: CheckpointSet | Sequence[str | None] | dict,
             *,
             task: str = "classification",
             mode: str | None = None,
             metrics: list[str] | str = "default",
             hp: Hyperparams | None = None,
             runtime: RuntimeConfig | None = None,
             cache: CacheConfig | None = None,
             log: LogConfig | None = None,
             self_reported_cost: dict | None = None) -> EvalReport:
    """Model-in convenience facade over :func:`run`.

    ``checkpoints`` accepts a CheckpointSet, a role-named dict, or the
    documented 3-sequence ``[original, gold, unlearned]`` with None
    placeholders. ``mode=None`` (the default) is inferred from the data's own
    ``mode`` field (falling back to ``"single_class"``); an explicit mode
    must agree with the data's mode (validated by ModelRequest).
    """
    if mode is None:
        mode = getattr(data, "mode", None) or "single_class"
    if isinstance(checkpoints, CheckpointSet):
        ckpts = checkpoints
    elif isinstance(checkpoints, dict):
        ckpts = CheckpointSet.model_validate(checkpoints)
    else:
        ckpts = CheckpointSet.model_validate(list(checkpoints))
    request = ModelRequest(
        task=task, mode=mode, seed=seed, checkpoints=ckpts, data=data,
        metrics=metrics, hp=hp or Hyperparams(),
        runtime=runtime or RuntimeConfig(), cache=cache or CacheConfig(),
        log=log or LogConfig(), self_reported_cost=self_reported_cost)
    return run(request)


def evaluate_outputs(request: OutputsRequest, **cfg: Any) -> EvalReport:
    """Outputs-in facade: score user-supplied per-example outputs.

    Keyword overrides (e.g. ``metrics=[...]``, ``hp=...``) are applied to a
    validated copy of the request before execution.
    """
    if cfg:
        request = request.model_copy(update=cfg)
        request = OutputsRequest.model_validate(request.model_dump())
    return run(request)


def map_items(fn: Callable[[Any], Any], items: Iterable[Any]) -> list[Any]:
    """The single named scale-out point (distributed hooks):
    a plain sequential loop in v1; a future data-parallel runner replaces
    exactly this function."""
    return [fn(item) for item in items]
