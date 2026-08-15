"""Attack recipes.

The relearning attack is the model-mutating attack axis in v0.1: it fine-tunes
a deepcopy of the unlearned model on a small budget and measures forget-accuracy
recovery (``trail.attacks.relearn``). It is driven directly by the relearning
metric via the model + train-view, not through a recipe/manifest indirection.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("trail.attacks")

try:
    from trail.attacks import relearn  # noqa: F401 (re-export)
except ModuleNotFoundError:  # pragma: no cover - integration window only
    relearn = None  # type: ignore[assignment]
    logger.warning("trail.attacks.relearn unavailable")

__all__ = ["relearn"]
