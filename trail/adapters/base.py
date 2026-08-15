"""ModalityAdapter — the single abstract base class in trail.

All modality knowledge lives behind this boundary; runner, cache, registry,
and report schema are modality-blind.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Literal

from trail.core.types import SplitOutputs

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from trail.core.request import EvalRequest


class ModalityAdapter(ABC):
    name: ClassVar[str]  # "classification" | "llm"

    @classmethod
    def from_request(cls, request: "EvalRequest") -> "ModalityAdapter":
        """Construct the adapter for ``request`` (the runner's entry point).

        Default: zero-arg construction — correct for adapters that derive no
        configuration from the request (e.g. the LLM stub, whose ctor takes no
        arch/num_classes). ``ClassificationAdapter`` overrides this to plumb
        ``arch`` / ``num_classes`` / ``dataset`` from ``request.runtime`` and the
        data's dataset name. NOT blind ``**kwargs`` — each adapter owns its own
        construction, so heterogeneous ctors never collide.
        """
        return cls()

    @abstractmethod
    def load_checkpoint(self, path: str, device: "torch.device") -> "nn.Module":
        """Unwrap checkpoint formats, build the architecture, validate strictly.
        Raises CheckpointError naming the failure. Caller hashes the file."""

    @abstractmethod
    def forward_stats(self, model: "nn.Module", loader: "DataLoader", *,
                      seed: int, device: "torch.device") -> SplitOutputs:
        """THE leaf probe: one deterministic pass producing SplitOutputs."""

    @abstractmethod
    def canonical_eval_view(self, loader: "DataLoader",
                            *, seed: int, num_workers: int = 0,
                            ) -> tuple["DataLoader", dict]:
        """Augmentation-stripped, deterministically ordered evaluation view of a
        user loader, plus the preprocessing manifest entry it generates.
        Must NOT mutate the user's dataset (constructive, not in-place)."""

    @abstractmethod
    def validate_outputs_payload(self, payload: dict) -> SplitOutputs:
        """Outputs-in parsing + validation against this modality's documented
        payload format. Raises RequestError on shape/dtype/semantic errors."""

    @abstractmethod
    def default_metrics(self) -> list[str]:
        """The default panel (names resolved against METRIC_REGISTRY)."""

    # Optional capability hooks; returning None auto-skips dependents.
    def train_reference(self, kind: Literal["gold", "shadow"], data,
                        seed: int) -> str | None:
        return None

    def derived_test_splits(self, test_loader: "DataLoader",
                            mode: str) -> dict[str, "DataLoader"] | None:
        return None
