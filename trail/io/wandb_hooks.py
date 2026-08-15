"""Weights & Biases session wrapper (optional).

Tracking is ON by default; reports remain complete and serializable without
it. Opt-out paths: ``LogConfig.wandb = False``, environment
``TRAIL_NO_WANDB=1``, or ``WANDB_MODE=disabled``. Missing ``wandb``
package or a doubly-failed ``wandb.init`` degrade to a no-op session.

All state lives on the session instance — deliberately NO module-global
wandb run, so two interleaved evaluations cannot cross-log.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from trail.core.report import EvalReport

logger = logging.getLogger("trail.io.wandb")


class WandbSession:
    """One W&B run scoped to one evaluation request.

    Construct via :meth:`start`. A no-op session (``run_id is None``) is
    returned whenever tracking is disabled or unavailable, so callers never
    branch on tracking state.
    """

    def __init__(self, wandb_module: Any | None = None,
                 run: Any | None = None) -> None:
        self._wandb = wandb_module
        self._run = run

    @property
    def run_id(self) -> str | None:
        """W&B run id for provenance stamping; ``None`` for a no-op session."""
        if self._run is None:
            return None
        return getattr(self._run, "id", None)

    @property
    def active(self) -> bool:
        """True when a live (or offline) W&B run is attached."""
        return self._run is not None

    @classmethod
    def start(cls, log_cfg: Any,
              request_summary: dict | None = None) -> "WandbSession":
        """Open a session for one evaluation (default ON).

        Args:
            log_cfg: the request's ``LogConfig`` (fields used: ``wandb``,
                ``wandb_project``, ``wandb_entity``).
            request_summary: JSON-safe dict describing the request; becomes
                the W&B run config (optional — an empty config when omitted,
                so a sparse caller can never crash the session start).

        Returns:
            A live session, an offline session (init retried with
            ``mode="offline"``), or a no-op session — in that preference
            order. Never raises.
        """
        enabled = bool(getattr(log_cfg, "wandb", True))
        if (not enabled
                or os.environ.get("TRAIL_NO_WANDB", "") == "1"
                or os.environ.get("WANDB_MODE", "") == "disabled"):
            logger.info("wandb tracking disabled (config or environment)")
            return cls()

        try:
            import wandb  # lazy: tracking extra, not a core dependency
        except ImportError:
            logger.warning(
                "wandb requested but not installed; tracking disabled "
                "(pip install trail[tracking])")
            return cls()

        project = getattr(log_cfg, "wandb_project", None) or "trail"
        entity = getattr(log_cfg, "wandb_entity", None)
        init_kwargs: dict[str, Any] = dict(
            project=project, entity=entity,
            config=dict(request_summary or {}), job_type="trail-eval")
        try:
            run = wandb.init(**init_kwargs)
        except Exception as exc:  # noqa: BLE001 — tracking must never crash an eval
            logger.warning("wandb.init failed (%s); retrying offline", exc)
            try:
                run = wandb.init(mode="offline", **init_kwargs)
            except Exception as exc2:  # noqa: BLE001
                logger.warning(
                    "wandb offline init failed (%s); tracking disabled", exc2)
                return cls()
        return cls(wandb, run)

    def log(self, metrics: dict, step: int | None = None) -> None:
        """Log incremental scalars (training curves) to the live run.

        No-op for a no-op session; best-effort (never raises) so tracking can
        never crash an evaluation. Used by metrics that train internally (the
        LiRA shadow ensemble) to stream per-epoch / per-shadow curves."""
        if self._run is None:
            return
        try:
            self._run.log(dict(metrics), step=step)
        except Exception as exc:  # noqa: BLE001 — tracking must never crash an eval
            logger.warning("wandb incremental log failed: %s", exc)

    def log_report(self, report: "EvalReport") -> None:
        """Log per-metric scalars and attach the report JSON as an artifact.

        Scalars are keyed ``f"{category}/{name}"`` with the MetricResult
        value. Artifact upload is best-effort (a report whose provenance is
        incomplete refuses serialization; that refusal is logged, not raised).
        """
        if self._run is None:
            return
        scalars: dict[str, float] = {}
        for category, results in report.metrics.items():
            for name, res in results.items():
                scalars[f"{category}/{name}"] = float(res.value)
        if scalars:
            try:
                self._run.log(scalars)
            except Exception as exc:  # noqa: BLE001
                logger.warning("wandb scalar logging failed: %s", exc)
        try:
            payload = report.to_json()
            artifact = self._wandb.Artifact(
                f"trail-report-{self.run_id}", type="eval-report")
            with artifact.new_file("report.json", mode="w") as fh:
                fh.write(payload)
            self._run.log_artifact(artifact)
        except Exception as exc:  # noqa: BLE001
            logger.warning("wandb report-artifact upload failed: %s", exc)

    def finish(self) -> None:
        """Close the underlying run (idempotent; never raises)."""
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception as exc:  # noqa: BLE001
            logger.warning("wandb finish failed: %s", exc)
        finally:
            self._run = None
