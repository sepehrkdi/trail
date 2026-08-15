"""Report schema: MetricResult, SkipInfo, Provenance, EvalReport.

Plain dataclasses; serialization is numpy-free (every number is converted to a
builtin ``float``/``int`` before JSON). ``EvalReport.to_json`` refuses to
serialize when provenance is incomplete (guarantee G3).
"""
from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from trail.core.bootstrap import bootstrap_ci
from trail.core.errors import ProvenanceError

logger = logging.getLogger("trail.report")

#: 1.0 -> 1.1 (F6): adds the ``not_applicable_scale`` skip code.
#: 1.1 -> 1.2 (F5): adds ``report.artifacts`` + ``provenance.artifact_sha256``
#: (the out-of-band npz/png/svg artifact subsystem). Both bumps are forward-
#: compatible — ``from_json`` defaults the new fields, so older reports load.
SCHEMA_VERSION = "1.2"

#: Machine-readable skip-reason codes (plus ``not_applicable_mode``,
#: the empty-by-design-split code from / errors.SplitNotAvailable, and
#: ``missing_metadata``: absent user-supplied request metadata, e.g.
#: ``self_reported_cost`` — distinct from a split empty by design).
#: ``not_applicable_scale`` (F6) is the supplied-gold-only scale policy: at
#: ImageNet scale ``gold='build'`` raises it instead of retraining. The raise
#: stays LATENT in Phase 0 — it only fires once F2 widens the dataset name enum
#: to admit a scale that triggers it.
SKIP_CODES: frozenset[str] = frozenset({
    "requires_model_in",
    "missing_role:original",
    "missing_role:gold",
    "reference_disabled:gold",
    "reference_disabled:shadow",
    "missing_metadata",
    "not_applicable_modality",
    "not_applicable_mode",
    "not_applicable_scale",
    "missing_artifact",
    "runtime_error",
})


def validate_skip_code(code: str) -> str:
    """Return ``code`` if it is a known skip code, else raise ``ValueError``."""
    if code not in SKIP_CODES:
        raise ValueError(
            f"unknown skip code {code!r}; known codes: {sorted(SKIP_CODES)}")
    return code


# ---------------------------------------------------------------------------
# Value containers
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    """One metric value with its per-example bootstrap CI (guarantee G5).

    ``cost_s`` / ``peak_mem_mb`` / ``cache_hit`` are overwritten by the
    runner's scoped wrapper after the metric function returns; metric authors
    only fill ``value``/``ci``/``n``/``components``.
    """

    value: float
    ci: tuple[float, float]
    n: int
    components: dict[str, float] | None = None
    cost_s: float = 0.0
    peak_mem_mb: float | None = None
    cache_hit: bool = False

    @classmethod
    def from_per_example(cls,
                         values: "np.ndarray | list[float]",
                         *,
                         rng: np.random.Generator,
                         n_boot: int,
                         alpha: float,
                         stat_fn: Callable[[np.ndarray], float] | None = None,
                         components: dict[str, float] | None = None,
                         ) -> "MetricResult":
        """Build a MetricResult from a per-example array.

        ``value`` is ``stat_fn`` (default: mean) over the array; ``ci`` is the
        percentile bootstrap of the same statistic via ``core.bootstrap``.
        """
        arr = np.asarray(values)
        stat = stat_fn or (lambda a: float(np.mean(a)))
        value = float(stat(arr)) if arr.size else float("nan")
        lo, hi = bootstrap_ci(arr, stat_fn, n=n_boot, alpha=alpha, rng=rng)
        comps = ({str(k): float(v) for k, v in components.items()}
                 if components is not None else None)
        return cls(value=value, ci=(float(lo), float(hi)), n=int(arr.size),
                   components=comps)


@dataclass
class SkipInfo:
    """Why a metric did not run: machine-readable code + human unlock message."""

    code: str
    message: str

    def __post_init__(self) -> None:
        validate_skip_code(self.code)


@dataclass
class Provenance:
    """Reproducibility manifest stamped into every report (guarantee G3)."""

    library_version: str = ""
    code_git_sha: str | None = None
    checkpoint_sha256: dict[str, str | None] = field(default_factory=dict)
    dataset_fingerprints: dict[str, str] = field(default_factory=dict)
    #: F5: artifact name -> sha256 of the emitted npz/png/svg bytes (output
    #: artifacts; the full scalar descriptors live in ``EvalReport.artifacts``).
    artifact_sha256: dict[str, str] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    references: dict[str, Any] = field(default_factory=dict)
    attack_manifests: dict[str, Any] = field(default_factory=dict)
    cache_hits: list[str] = field(default_factory=list)
    wall_clock_s: float = 0.0
    device: str = ""
    gpu_name: str | None = None
    torch_version: str = ""
    cuda_version: str | None = None
    wandb_run_id: str | None = None
    timestamp: str = ""

    def validate_complete(self, input_mode: str = "model") -> None:
        """Raise ProvenanceError when the manifest cannot anchor the report.

        Model-in reports must carry the unlearned checkpoint hash; every
        report must carry split fingerprints (for outputs-in these are
        payload-content fingerprints).
        """
        problems: list[str] = []
        if input_mode == "model" and not self.checkpoint_sha256.get("unlearned"):
            problems.append("checkpoint_sha256['unlearned'] is missing (model-in)")
        if not self.dataset_fingerprints:
            problems.append("dataset_fingerprints is empty")
        if problems:
            raise ProvenanceError("provenance incomplete: " + "; ".join(problems))


# ---------------------------------------------------------------------------
# JSON helpers (numpy-free output)
# ---------------------------------------------------------------------------

def _plain(obj: Any) -> Any:
    """Recursively convert to builtin JSON-serializable types."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_plain(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(str(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON-serializable in a report: {type(obj)}")


def _metric_to_dict(r: MetricResult) -> dict[str, Any]:
    d: dict[str, Any] = {
        "value": float(r.value),
        "ci": [float(r.ci[0]), float(r.ci[1])],
        "n": int(r.n),
        "cost_s": float(r.cost_s),
        "cache_hit": bool(r.cache_hit),
    }
    if r.components is not None:
        d["components"] = {str(k): float(v) for k, v in r.components.items()}
    if r.peak_mem_mb is not None:
        d["peak_mem_mb"] = float(r.peak_mem_mb)
    return d


def _metric_from_dict(d: dict[str, Any]) -> MetricResult:
    return MetricResult(
        value=float(d["value"]),
        ci=(float(d["ci"][0]), float(d["ci"][1])),
        n=int(d["n"]),
        components=d.get("components"),
        cost_s=float(d.get("cost_s", 0.0)),
        peak_mem_mb=(float(d["peak_mem_mb"]) if d.get("peak_mem_mb") is not None
                     else None),
        cache_hit=bool(d.get("cache_hit", False)),
    )


# ---------------------------------------------------------------------------
# EvalReport
# ---------------------------------------------------------------------------

@dataclass
class EvalReport:
    """The artifact of record for one evaluation call."""

    task: str
    mode: str
    seed: int
    input_mode: str
    metrics: dict[str, dict[str, MetricResult]]
    skipped: dict[str, SkipInfo]
    warnings: list[str]
    hyperparams: dict[str, Any]
    provenance: Provenance
    #: Per-knob disclosure record (knob -> "explicit" | "protocol_default";
    #: see request.check_disclosure /). Empty for legacy reports.
    disclosure: dict[str, str] = field(default_factory=dict)
    #: What this framework actually certifies: EMPIRICAL (attack-based)
    #: unlearning evaluation — it makes no exact/certified/DP guarantee
    #: (principles.md §E5). The method's own targeted guarantee, if the user
    #: declares one, is recorded in ``disclosure["method_guarantee"]``.
    evaluation_guarantee: str = "empirical"
    #: F5: scalar descriptors of out-of-band artifacts (npz/png/svg). Heavy
    #: arrays live on disk (content-addressed); only these descriptors are in
    #: the report, and the aggregator never turns them into CSV columns.
    artifacts: list[dict] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def scores(self) -> dict[str, dict[str, dict[str, float]]]:
        """Lossy float view: ``{task: {category: {metric: value}}}``."""
        return {self.task: {cat: {name: float(r.value)
                                  for name, r in results.items()}
                            for cat, results in self.metrics.items()}}

    def to_json(self, path: str | Path | None = None) -> str:
        """Canonical JSON (sorted keys, indent=2). Refuses incomplete provenance.

        Returns the JSON string; additionally writes it when ``path`` is given.
        """
        self.provenance.validate_complete(self.input_mode)
        payload = {
            "schema_version": self.schema_version,
            "task": self.task,
            "mode": self.mode,
            "seed": int(self.seed),
            "input_mode": self.input_mode,
            "metrics": {cat: {name: _metric_to_dict(r)
                              for name, r in results.items()}
                        for cat, results in self.metrics.items()},
            "skipped": {name: {"code": s.code, "message": s.message}
                        for name, s in self.skipped.items()},
            "warnings": list(self.warnings),
            "hyperparams": _plain(self.hyperparams),
            "disclosure": dict(self.disclosure),
            "evaluation_guarantee": self.evaluation_guarantee,
            "artifacts": list(self.artifacts),
            "provenance": _plain(dataclasses.asdict(self.provenance)),
        }
        text = json.dumps(_plain(payload), sort_keys=True, indent=2)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
            logger.info("report written to %s", path)
        return text

    @classmethod
    def from_json(cls, path: str | Path) -> "EvalReport":
        """Reconstruct a report (incl. nested dataclasses and tuple CIs)."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        prov_fields = {f.name for f in dataclasses.fields(Provenance)}
        prov = Provenance(**{k: v for k, v in raw.get("provenance", {}).items()
                             if k in prov_fields})
        return cls(
            task=raw["task"],
            mode=raw["mode"],
            seed=int(raw["seed"]),
            input_mode=raw["input_mode"],
            metrics={cat: {name: _metric_from_dict(d)
                           for name, d in results.items()}
                     for cat, results in raw.get("metrics", {}).items()},
            skipped={name: SkipInfo(code=d["code"], message=d["message"])
                     for name, d in raw.get("skipped", {}).items()},
            warnings=list(raw.get("warnings", [])),
            hyperparams=raw.get("hyperparams", {}),
            disclosure=dict(raw.get("disclosure", {})),
            evaluation_guarantee=raw.get("evaluation_guarantee", "empirical"),
            artifacts=list(raw.get("artifacts", [])),
            provenance=prov,
            schema_version=raw.get("schema_version", SCHEMA_VERSION),
        )

    def to_markdown(self) -> str:
        """Human-readable rendering: one table per category + skips + digest."""
        lines: list[str] = [
            f"# TRAIL report — task `{self.task}`, mode `{self.mode}`, "
            f"seed {self.seed}, input `{self.input_mode}`",
            "",
            f"_Evaluation guarantee: **{self.evaluation_guarantee}** "
            "(attack-based; no exact/certified/DP claim)._",
            "",
        ]
        for cat in sorted(self.metrics):
            lines += [f"## {cat}", "",
                      "| metric | value | ci | n | cost (s) | cache |",
                      "|---|---|---|---|---|---|"]
            for name in sorted(self.metrics[cat]):
                r = self.metrics[cat][name]
                ci = f"[{r.ci[0]:.4g}, {r.ci[1]:.4g}]"
                hit = "hit" if r.cache_hit else "miss"
                lines.append(f"| {name} | {r.value:.4g} | {ci} | {r.n} "
                             f"| {r.cost_s:.1f} | {hit} |")
            lines.append("")
        if self.skipped:
            lines += ["## skipped", "", "| metric | code | unlock |", "|---|---|---|"]
            for name in sorted(self.skipped):
                s = self.skipped[name]
                lines.append(f"| {name} | `{s.code}` | {s.message} |")
            lines.append("")
        if self.warnings:
            lines += ["## warnings", ""]
            lines += [f"- {w}" for w in self.warnings]
            lines.append("")
        p = self.provenance
        ckpt = ", ".join(f"{role}={sha[:12] if sha else 'absent'}"
                         for role, sha in sorted(p.checkpoint_sha256.items()))
        fps = ", ".join(f"{split}={fp[:12]}"
                        for split, fp in sorted(p.dataset_fingerprints.items()))
        lines += [
            "## provenance",
            "",
            f"- library `{p.library_version}` @ git `{p.code_git_sha or 'n/a'}`,"
            f" schema `{self.schema_version}`",
            f"- checkpoints: {ckpt or 'none'}",
            f"- splits: {fps or 'none'}",
            f"- device `{p.device}`"
            + (f" ({p.gpu_name})" if p.gpu_name else "")
            + f", torch `{p.torch_version}`, cuda `{p.cuda_version or 'n/a'}`",
            f"- wall clock {p.wall_clock_s:.1f}s, {len(p.cache_hits)} cache hits,"
            f" timestamp {p.timestamp}",
        ]
        if p.wandb_run_id:
            lines.append(f"- wandb run `{p.wandb_run_id}`")
        return "\n".join(lines) + "\n"
