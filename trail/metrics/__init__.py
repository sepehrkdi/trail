"""trail.metrics — importing this package populates METRIC_REGISTRY.

Registration is static and import-time: every metric module is
imported here for its ``register_metric`` side effects, in the taxonomy order
in a fixed order, so the registry's insertion order is deterministic across runs.
"""
from __future__ import annotations

from trail.metrics import (  # noqa: F401  (imported for registration side effects)
    accuracy,
    privacy,
    relearning,
    efficiency,
    structural,
    generative,
    representation,
    distributional,
    klom,
    whitebox,
    poisoning,
    recovery,
    robustness,
)

__all__ = [
    "accuracy",
    "privacy",
    "relearning",
    "efficiency",
    "structural",
    "generative",
    "representation",
    "distributional",
    "klom",
    "whitebox",
    "poisoning",
    "recovery",
    "robustness",
]
