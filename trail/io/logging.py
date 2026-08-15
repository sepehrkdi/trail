"""Machine-parseable logging for trail.

Standard-library ``logging`` only — no bare prints anywhere in the package.
Per-metric begin/end lines carry wall time and cache status so a log scrape
can reconstruct the panel schedule.
"""
from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_HANDLER_FLAG = "_trail_handler"


def setup_logging(level: int | str = "INFO") -> logging.Logger:
    """Configure the package-root ``trail`` logger.

    Idempotent: repeated calls adjust the level but never stack handlers.

    Args:
        level: stdlib logging level name or number (e.g. ``"INFO"``, ``10``).

    Returns:
        The configured ``trail`` root logger.
    """
    root = logging.getLogger("trail")
    root.setLevel(level)
    if not any(getattr(h, _HANDLER_FLAG, False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        setattr(handler, _HANDLER_FLAG, True)
        root.addHandler(handler)
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a logger inside the ``trail`` namespace.

    ``get_logger("metrics")`` and ``get_logger("trail.metrics")`` both
    yield the ``trail.metrics`` logger, so callers cannot accidentally
    escape the package-root handler configured by :func:`setup_logging`.
    """
    if name == "trail" or name.startswith("trail."):
        return logging.getLogger(name)
    return logging.getLogger(f"trail.{name}")


def metric_log(logger: logging.Logger, name: str, phase: str,
               wall_s: float | None = None,
               cache_hit: bool | None = None) -> None:
    """Emit one machine-parseable metric lifecycle line.

    Format: ``metric=<name> phase=<phase> [wall_s=<sec>] [cache_hit=<0|1>]``
    — fields are space-separated ``key=value`` pairs; absent optionals are
    omitted rather than written as ``None``.

    Args:
        logger: destination logger (typically ``get_logger("runner")``).
        name: registry metric name, e.g. ``"fa_train"``.
        phase: lifecycle phase, conventionally ``"begin"`` or ``"end"``.
        wall_s: elapsed wall-clock seconds (end lines).
        cache_hit: whether the metric's probe was served from cache.
    """
    parts = [f"metric={name}", f"phase={phase}"]
    if wall_s is not None:
        parts.append(f"wall_s={wall_s:.3f}")
    if cache_hit is not None:
        parts.append(f"cache_hit={int(bool(cache_hit))}")
    logger.info(" ".join(parts))
