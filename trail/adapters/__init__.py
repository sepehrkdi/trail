"""Adapter registry — name -> ModalityAdapter subclass.

Registration is static and import-time, mirroring the metric registry
: importing ``trail.adapters`` deterministically populates
``ADAPTERS`` with the three shipped adapters. The runner resolves
``request.task`` against this dict.
"""
from __future__ import annotations

import logging

from trail.adapters.base import ModalityAdapter

logger = logging.getLogger("trail.adapters")

#: task name ("classification" | "llm" | plugin) -> adapter class
ADAPTERS: dict[str, type[ModalityAdapter]] = {}


def register_adapter(cls: type[ModalityAdapter]) -> type[ModalityAdapter]:
    """Register a ModalityAdapter subclass under its ``name`` class attribute.

    Usable as a decorator by plugin adapters. Idempotent for the same class;
    raises on a conflicting re-registration of an existing name.

    Args:
        cls: a concrete ModalityAdapter subclass with a ``name`` ClassVar.

    Returns:
        ``cls`` unchanged (decorator convention).

    Raises:
        ValueError: if ``cls`` lacks a ``name`` or the name is already
            registered to a different class.
    """
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"adapter class {cls.__name__} must define a non-empty "
            "'name' ClassVar")
    existing = ADAPTERS.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"duplicate adapter registration for {name!r}: "
            f"{existing.__name__} vs {cls.__name__}")
    ADAPTERS[name] = cls
    return cls


# Static import-time registration of the shipped adapters.
# Classification is the implemented modality; the LLM adapter ships as an
# interface stub (future work); other modalities are out of scope for v0.1.
from trail.adapters.classification import ClassificationAdapter  # noqa: E402
from trail.adapters.llm import LLMAdapter  # noqa: E402

register_adapter(ClassificationAdapter)
register_adapter(LLMAdapter)

__all__ = [
    "ADAPTERS",
    "ModalityAdapter",
    "register_adapter",
    "ClassificationAdapter",
    "LLMAdapter",
]
