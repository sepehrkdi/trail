"""TRAIL: a unified, reproducible evaluation library for machine unlearning.

Public surface (lazily resolved so that ``import trail`` stays cheap and
free of torch/pydantic imports until first use):

- :func:`evaluate` / :func:`evaluate_outputs` / :func:`run` — entry points
- :class:`EvalReport` — the artifact of record
- :class:`Hyperparams`, :class:`CheckpointSet`, :class:`DatasetSpec` — request types
- :func:`register_metric` — plugin metric registration
"""
from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

#: attribute name -> (module, attribute) for PEP 562 lazy re-export
_LAZY: dict[str, tuple[str, str]] = {
    "evaluate": ("trail.core.runner", "evaluate"),
    "evaluate_outputs": ("trail.core.runner", "evaluate_outputs"),
    "run": ("trail.core.runner", "run"),
    "EvalReport": ("trail.core.report", "EvalReport"),
    "Hyperparams": ("trail.core.request", "Hyperparams"),
    "CheckpointSet": ("trail.core.request", "CheckpointSet"),
    "DatasetSpec": ("trail.data.specs", "DatasetSpec"),
    "register_metric": ("trail.core.registry", "register_metric"),
}

__all__ = ["__version__", *sorted(_LAZY)]


def __getattr__(name: str) -> Any:
    """Resolve a lazy re-export on first access (PEP 562)."""
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value  # cache: subsequent accesses skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
