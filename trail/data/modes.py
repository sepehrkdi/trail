"""Forgetting-mode (scenario) registry.

A :class:`ModeSpec` is the declarative description of one forgetting scenario:
which ``DatasetSpec`` fields it ``requires``, whether it ``yields_forget_test``
partitions, which params form its split identity (``id_params``), and the
``split_fn`` that resolves the (forget, retain) positions *within the carved
trainset*.

This module holds ONLY the registry machinery — no split logic and (crucially)
NO import of ``trail.data.specs`` — so there is no import cycle. The built-in
modes (``single_class`` / ``random`` / ``sub_class_atypical``) are registered by
``data/specs.py`` at import time, by decorating the byte-exact split bodies
ported from the original data pipeline. Those wrappers call the existing
branch bodies VERBATIM (no reordering of ``RandomState`` construction or
consumption), so the G10 split fixtures (trail/fixtures/splits/*.json) keep
reproducing the index sequences bit-for-bit.

``MODE_REGISTRY`` is therefore populated as a side effect of importing
``trail.data.specs`` (the split path lives there); ``register_mode`` is
idempotent for the built-ins so a module re-import / re-entrant registration is
a no-op rather than a duplicate-registration error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

#: A split function: ``(spec, train_targets, train_idx, targets_carved) ->
#: (forget_pos, retain_pos)`` where positions index *into the carved trainset*.
#: ``targets_carved == train_targets[train_idx]`` is precomputed once by the
#: dispatcher and passed in (single_class/random read it; sub_class_atypical
#: works off the original-order ``train_targets`` instead).
SplitFn = Callable[..., "tuple"]


@dataclass(frozen=True)
class ModeSpec:
    """Declarative description of one forgetting scenario.

    Attributes:
        name: the mode key (matches ``DatasetSpec.mode``).
        requires: ``DatasetSpec`` field names this mode needs set (mirrors the
            ``DatasetSpec`` validator; declarative metadata for downstream
            feasibility checks).
        yields_forget_test: whether the mode derives ``forget_test`` /
            ``retain_test`` partitions (class-forgetting modes do; random and
            sub-class atypical leave them empty by design).
        id_params: the ``DatasetSpec`` params that vary this mode's split — the
            mode-relevant slice of the dataset identity.
        split_fn: resolver of (forget, retain) carved-trainset positions.
        builtin: True for the three ported built-ins (enables idempotent
            re-registration; protects them from being clobbered).
    """

    name: str
    requires: tuple[str, ...]
    yields_forget_test: bool
    id_params: tuple[str, ...]
    split_fn: SplitFn
    builtin: bool = False


MODE_REGISTRY: dict[str, ModeSpec] = {}


def register_mode(*, name: str, requires: Sequence[str],
                  yields_forget_test: bool, id_params: Sequence[str],
                  builtin: bool = False) -> Callable[[SplitFn], SplitFn]:
    """Decorator registering ``split_fn`` as the resolver for mode ``name``.

    Idempotent for the built-ins: re-registering an existing mode with
    ``builtin=True`` is a no-op (returns the function unchanged) so re-importing
    the split module cannot raise. Any OTHER attempt to register a name that is
    already taken raises ``ValueError`` — a plugin may not silently clobber an
    existing (built-in or third-party) mode.
    """

    def _register(fn: SplitFn) -> SplitFn:
        if name in MODE_REGISTRY:
            if builtin:
                # idempotent re-registration of a built-in: leave the existing
                # entry in place, no duplicate error.
                return fn
            raise ValueError(f"duplicate mode registration: {name!r}")
        MODE_REGISTRY[name] = ModeSpec(
            name=name, requires=tuple(requires),
            yields_forget_test=bool(yields_forget_test),
            id_params=tuple(id_params), split_fn=fn, builtin=builtin,
        )
        return fn

    return _register
