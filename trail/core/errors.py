"""Exception taxonomy. The raiser knows *why* something is unavailable, so the
raiser picks the machine-readable skip code; the runner only transcribes it.
"""
from __future__ import annotations


class TRAILError(Exception):
    """Base class for all trail errors."""


class RequestError(TRAILError):
    """Invalid request: bad config, malformed payload, empty required split."""


class DisclosureError(RequestError):
    """A legitimately-varying protocol knob (e.g. the relearning recipe) was
    left to silent defaults for an attack that actually runs. The user must
    disclose it: set the knob explicitly, or acknowledge the frozen defaults
    via ``hp.accept_protocol_defaults=True`` (controlled-freedom contract)."""


class CheckpointError(TRAILError):
    """A checkpoint failed to load or validate for a given role."""

    def __init__(self, role: str, path: str, reason: str):
        self.role, self.path, self.reason = role, path, reason
        super().__init__(f"checkpoint role={role!r} path={path!r}: {reason}")


class InputModeError(TRAILError):
    """Model/loader access attempted under an outputs-in request."""


class MissingReference(TRAILError):
    """A reference (gold/shadow/concept clf/attack payload) is unavailable.

    ``skip_code`` carries the exact code, e.g. ``reference_disabled:gold`` vs
    ``missing_role:gold`` — the raiser knows which applies.
    """

    def __init__(self, skip_code: str, message: str):
        self.skip_code = skip_code
        super().__init__(message)


class SplitNotAvailable(TRAILError):
    """Split is EMPTY by design in this forgetting mode (e.g. forget_test in
    random mode) -> skip code ``not_applicable_mode``."""

    def __init__(self, split: str, mode: str):
        self.split, self.mode = split, mode
        super().__init__(f"split {split!r} is empty by design in mode {mode!r}")


class MetricError(TRAILError):
    """Fail-soft boundary (G4): a metric body failed; panel continues."""


class NonFiniteLossError(MetricError):
    """Non-finite per-example losses encountered at consumption."""


class MetricSkip(TRAILError):
    """Raised inside a metric body to request a skip with an explicit code
    (e.g. a mode guard raising ``not_applicable_mode``)."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class ProvenanceError(TRAILError):
    """Report serialization refused: provenance incomplete (G3)."""
