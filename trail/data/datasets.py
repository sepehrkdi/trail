"""Dataset registry.

De-hardcodes dataset/torchvision construction out of ``data/specs.py``. A
:class:`DatasetEntry` binds a dataset name to its class count, human-readable
class names, and a builder ``(data_dir, *, train, transform) -> Dataset``.

Public surface (the F2 contract):
  * ``register_dataset`` — add a dataset (idempotent for an identical entry).
  * ``build_datasets``   — THE construction entry point (per-split dispatch).
  * ``num_classes_for`` / ``class_names_for`` — identity queries.
  * ``CLASS_NAMES``      — name -> class-name tuple (denormalized convenience).
  * ``_targets_of``      — normalize the torchvision label attribute
    (``.targets`` list, CIFAR/MNIST vs ``.labels`` ndarray, SVHN/STL10) to an
    ndarray.

This module imports NO sibling data module (no ``specs.py``) — transforms are
passed in by the caller — so there is no import cycle. Only ``cifar10`` is
registered in Phase 1; cifar100/svhn/stl10/mnist/fmnist arrive in Phase 5,
imagenet1k in Phase 6. The CIFAR-10 builder reproduces the EXACT torchvision
call the old hardcoded ``specs.py`` sites made (``CIFAR10(root, train,
download=True, transform)``), so the byte-exact split fixtures are unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from trail.core.errors import RequestError

#: A dataset builder: ``(data_dir, *, train: bool, transform) -> Dataset``.
DatasetBuilder = Callable[..., Any]


@dataclass(frozen=True)
class DatasetEntry:
    """One registered dataset: identity metadata + a torchvision builder."""

    name: str
    num_classes: int
    class_names: tuple[str, ...]
    builder: DatasetBuilder


DATASET_REGISTRY: dict[str, DatasetEntry] = {}

#: name -> class-name tuple (kept in sync by ``register_dataset``).
CLASS_NAMES: dict[str, tuple[str, ...]] = {}


def register_dataset(name: str, *, num_classes: int,
                     class_names: Sequence[str], builder: DatasetBuilder,
                     ) -> DatasetEntry:
    """Register a dataset under ``name``.

    Idempotent for an identical re-registration (same num_classes, class names,
    and builder object) — a re-import or re-entrant registration is a no-op. A
    conflicting re-registration under an existing name raises ``ValueError``.
    """
    class_names = tuple(class_names)
    if len(class_names) != num_classes:
        raise ValueError(
            f"dataset {name!r}: len(class_names)={len(class_names)} != "
            f"num_classes={num_classes}")
    existing = DATASET_REGISTRY.get(name)
    if existing is not None:
        if (existing.num_classes == num_classes
                and existing.class_names == class_names
                and existing.builder is builder):
            return existing  # idempotent
        raise ValueError(f"duplicate dataset registration: {name!r}")
    entry = DatasetEntry(name=name, num_classes=num_classes,
                         class_names=class_names, builder=builder)
    DATASET_REGISTRY[name] = entry
    CLASS_NAMES[name] = class_names
    return entry


def _require(name: str) -> DatasetEntry:
    try:
        return DATASET_REGISTRY[name]
    except KeyError:
        raise RequestError(
            f"unknown dataset {name!r}; registered: {sorted(DATASET_REGISTRY)}")


def num_classes_for(name: str) -> int:
    """Classifier head width for a registered dataset."""
    return _require(name).num_classes


def class_names_for(name: str) -> tuple[str, ...]:
    """Ordered human-readable class names for a registered dataset."""
    return _require(name).class_names


def build_datasets(name: str, data_dir: str, *,
                   splits: Sequence[str] = ("train", "test"),
                   train_transform: Any = None,
                   test_transform: Any = None) -> dict[str, Any]:
    """Build the requested torchvision splits for a registered dataset.

    Returns ``{split: Dataset}``. ``train``/``test`` are the only splits; the
    train split uses ``train_transform`` (None = no transform, the
    ``forget_indices`` path) and the test split uses ``test_transform``. The
    builder owns the actual torchvision call (and any download); this layer only
    dispatches, so the construction stays in one place.
    """
    entry = _require(name)
    out: dict[str, Any] = {}
    for split in splits:
        if split == "train":
            out["train"] = entry.builder(data_dir, train=True,
                                         transform=train_transform)
        elif split == "test":
            out["test"] = entry.builder(data_dir, train=False,
                                        transform=test_transform)
        else:
            raise RequestError(
                f"unknown split {split!r}; expected 'train' or 'test'")
    return out


def _targets_of(dataset: Any) -> np.ndarray:
    """Normalize a torchvision dataset's integer labels to an ndarray.

    CIFAR/MNIST expose ``.targets`` (a Python list); SVHN/STL10 expose
    ``.labels`` (an ndarray). Returns ``np.asarray(<labels>)`` with NO dtype
    coercion — byte-identical to the old ``np.asarray(base_train.targets)``.
    """
    for attr in ("targets", "labels"):
        vals = getattr(dataset, attr, None)
        if vals is not None:
            return np.asarray(vals)
    raise RequestError(
        f"dataset {type(dataset).__name__} exposes neither '.targets' nor "
        "'.labels'; cannot read integer labels")


# ---------------------------------------------------------------------------
# Built-in datasets (Phase 1: cifar10 only — byte-exact with the old sites).
# ---------------------------------------------------------------------------

def _build_cifar10(data_dir: str, *, train: bool, transform: Any) -> Any:
    """Exact port of the old hardcoded ``torchvision.datasets.CIFAR10`` calls
    in data/specs.py (resolve_dataset_spec / forget_indices / train_view)."""
    import torchvision  # lazy: keep the data layer importable without torchvision

    return torchvision.datasets.CIFAR10(
        root=data_dir, train=train, download=True, transform=transform)


#: torchvision label ordering (the single source of truth for CIFAR-10 class
#: names; specs.CIFAR10_CLASSES
#: is derived from this entry).
_CIFAR10_CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

register_dataset("cifar10", num_classes=10, class_names=_CIFAR10_CLASS_NAMES,
                 builder=_build_cifar10)


# ---------------------------------------------------------------------------
# Downstream datasets (Phase 5) — for transfer probing (knn_transfer), NOT the
# cifar10 forget/retain split path. SVHN/STL10 take split=str + expose .labels;
# CIFAR100/MNIST/FMNIST take train=bool + expose .targets (_targets_of handles
# both). MNIST/FMNIST grayscale->RGB promotion lives in the transform (specs.py),
# never the model. Tiny-ImageNet deferred (needs an ImageFolder helper).
# ---------------------------------------------------------------------------

def _torchvision_builder(cls_name: str, *, split_arg: bool):
    """Builder factory: ``split_arg=True`` -> split='train'/'test' (SVHN/STL10);
    else train=bool (CIFAR100/MNIST/FMNIST)."""
    def _build(data_dir: str, *, train: bool, transform):
        import torchvision  # lazy
        cls = getattr(torchvision.datasets, cls_name)
        if split_arg:
            return cls(root=data_dir, split="train" if train else "test",
                       download=True, transform=transform)
        return cls(root=data_dir, train=train, download=True, transform=transform)
    return _build


_DIGITS = tuple(str(i) for i in range(10))
_FMNIST_CLASSES = ("tshirt", "trouser", "pullover", "dress", "coat",
                   "sandal", "shirt", "sneaker", "bag", "ankle_boot")
_STL10_CLASSES = ("airplane", "bird", "car", "cat", "deer",
                  "dog", "horse", "monkey", "ship", "truck")

register_dataset("cifar100", num_classes=100,
                 class_names=tuple(f"class_{i}" for i in range(100)),
                 builder=_torchvision_builder("CIFAR100", split_arg=False))
register_dataset("svhn", num_classes=10, class_names=_DIGITS,
                 builder=_torchvision_builder("SVHN", split_arg=True))
register_dataset("stl10", num_classes=10, class_names=_STL10_CLASSES,
                 builder=_torchvision_builder("STL10", split_arg=True))
register_dataset("mnist", num_classes=10, class_names=_DIGITS,
                 builder=_torchvision_builder("MNIST", split_arg=False))
register_dataset("fmnist", num_classes=10, class_names=_FMNIST_CLASSES,
                 builder=_torchvision_builder("FashionMNIST", split_arg=False))
