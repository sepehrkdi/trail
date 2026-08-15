"""Declarative dataset specs and split builders.

``DatasetSpec`` is the declarative input surface; ``resolve_dataset_spec``
turns it into a ``SplitBundle`` of canonical (augmentation-stripped,
deterministically ordered) DataLoaders. The carve-out and the
``single_class`` / ``random`` forget/retain split recipes are BYTE-EXACT
ports of the original data pipeline — the G10 regression fixtures
(trail/fixtures/splits/*.json) cover exactly those two modes and depend
on reproducing the index sequences bit-for-bit.

DEVIATION — ``sub_class_atypical`` is deliberately NOT byte-exact: the
legacy ``split_atypical_subclass`` always received the physically re-indexed
45000-sample trainset, took the ``trainset_orig_indices = arange(len)``
branch (the original data pipeline), and therefore indexed the
*original-order* C-Score memorization array with carved-trainset POSITIONS —
an indexing artifact that misranked sample atypicality (legacy vs corrected
forget sets overlap only ~9%). This module keeps the corrected semantics:
C-Scores are looked up by TRUE original-CIFAR indices, so the forget set
really is the most-atypical fraction. The ±1 pp regression gate covers
single_class/random only; legacy-derived atypical rows are excluded.

Seeding note: this module constructs ``np.random.RandomState(<explicit user
seed>)`` from ``carveout_seed`` / ``split_seed``. That is the sanctioned
exception to the named-substream rule (see core/seeding.py module docstring):
it reproduces the fixture-era data recipe byte-exactly and mutates no global
RNG. All *framework* randomness (loader workers, generators) still routes
through trail.core.seeding.
"""
from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torchvision.transforms as transforms
from pydantic import BaseModel, ConfigDict, model_validator
from torch.utils.data import DataLoader, Dataset, Subset

from trail.core.errors import RequestError, SplitNotAvailable
from trail.core.seeding import make_worker_init_fn, torch_generator
from trail.data.datasets import (
    DATASET_REGISTRY,
    build_datasets,
    class_names_for,
    num_classes_for,
    _targets_of,
)
from trail.data.modes import MODE_REGISTRY, register_mode

logger = logging.getLogger("trail.data.specs")


# ---------------------------------------------------------------------------
# Transforms — ported from the original data pipeline.
# ---------------------------------------------------------------------------

# Per-dataset crop size for training augmentation.
# Ported from the original data pipeline.
DATASET_CROP_SIZE: dict[str, int] = {
    "cifar10": 32,
    "cifar100": 32,
    "svhn": 32,
    "stl10": 96,
    "mnist": 28,
    "fmnist": 28,
    "tiny_imagenet": 64,
}

# Per-dataset channel statistics.
# Ported from the original data pipeline.
DATASET_STATS: dict[str, dict[str, tuple[float, float, float]]] = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "cifar100": {
        "mean": (0.5071, 0.4865, 0.4409),
        "std": (0.2673, 0.2564, 0.2762),
    },
    "svhn": {
        "mean": (0.4377, 0.4438, 0.4728),
        "std": (0.1980, 0.2010, 0.1970),
    },
    "stl10": {
        "mean": (0.4467, 0.4398, 0.4066),
        "std": (0.2603, 0.2566, 0.2713),
    },
    # MNIST/FMNIST are grayscale; after Grayscale(3)->RGB the 3 channels are
    # identical, so the per-channel stats repeat the single-channel value.
    "mnist": {
        "mean": (0.1307, 0.1307, 0.1307),
        "std": (0.3081, 0.3081, 0.3081),
    },
    "fmnist": {
        "mean": (0.2860, 0.2860, 0.2860),
        "std": (0.3530, 0.3530, 0.3530),
    },
    "tiny_imagenet": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
}

#: Datasets that need grayscale->3-channel promotion IN THE TRANSFORM (not the
#: model), so a 3-channel backbone consumes them unchanged (Phase 5).
_GRAYSCALE_DATASETS: frozenset[str] = frozenset({"mnist", "fmnist"})

# Backward-compatible export. The single source of truth is now the ``cifar10``
# entry in data/datasets.py; this is derived from it as a list to preserve
# the public type/contract. Equal element-for-element to the legacy literal
# (the original data pipeline).
CIFAR10_CLASSES: list[str] = list(class_names_for("cifar10"))


def get_normalization_stats(dataset: str) -> tuple[list[float], list[float]]:
    """Return ``(mean, std)`` for the named dataset.

    Ported from the original data pipeline.
    """
    key = dataset.lower()
    if key not in DATASET_STATS:
        raise ValueError(
            f"Unknown dataset {dataset!r} — expected one of {list(DATASET_STATS)}."
        )
    stats = DATASET_STATS[key]
    return list(stats["mean"]), list(stats["std"])


def _channel_transforms(dataset: str) -> list:
    """Grayscale->3-channel promotion for MNIST/FMNIST (empty for everything
    else — so cifar10's Compose is byte-identical to the pre-Phase-5 transform)."""
    if dataset.lower() in _GRAYSCALE_DATASETS:
        return [transforms.Grayscale(num_output_channels=3)]
    return []


def get_train_transform(dataset: str = "cifar10") -> transforms.Compose:
    """Training transform: crop + flip + ToTensor + dataset-specific Normalize.

    Used only for attack-side fine-tuning views (``train_view_dataset``);
    every evaluation path is augmentation-stripped.
    Ported from the original data pipeline; grayscale promotion added
    for MNIST/FMNIST (Phase 5).
    """
    mean, std = get_normalization_stats(dataset)
    crop = DATASET_CROP_SIZE.get(dataset.lower(), 32)
    return transforms.Compose([
        *_channel_transforms(dataset),
        transforms.RandomCrop(crop, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_test_transform(dataset: str = "cifar10") -> transforms.Compose:
    """Canonical evaluation transform: ToTensor + dataset-specific Normalize.

    Ported from the original data pipeline; grayscale promotion added
    for MNIST/FMNIST (Phase 5; empty for cifar10 -> byte-identical).
    """
    mean, std = get_normalization_stats(dataset)
    return transforms.Compose([
        *_channel_transforms(dataset),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ---------------------------------------------------------------------------
# C-Scores — ported from the original data pipeline.
# ---------------------------------------------------------------------------

_CSCORE_URLS: dict[str, str] = {
    "cifar10": "https://pluskid.github.io/structural-regularity/cscores/cifar10-cscores-orig-order.npz",
    "cifar100": "https://pluskid.github.io/structural-regularity/cscores/cifar100-cscores-orig-order.npz",
}


def load_or_download_cscores(dataset_name: str, cache_dir: str | Path) -> np.ndarray:
    """Return memorization values (= ``1 − cscore``) for the dataset.

    On first call, downloads the C-Score npz (Jiang et al., 2021) from the
    public source into ``cache_dir``; subsequent calls hit the cache. The
    returned array is indexed in *original* dataset order (50000 entries for
    CIFAR-10); higher values indicate more atypical samples.

    Ported from the original data pipelinewith the env-var
    cache location replaced by an explicit ``cache_dir`` argument and prints
    replaced by logging.
    """
    key = dataset_name.lower()
    if key not in _CSCORE_URLS:
        raise ValueError(
            f"No C-Score URL configured for dataset {dataset_name!r}. "
            f"Known: {sorted(_CSCORE_URLS)}."
        )

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}-cscores-orig-order.npz"
    if not cache_path.exists():
        url = _CSCORE_URLS[key]
        logger.info("Downloading C-Scores for %s from %s ...", key, url)
        urllib.request.urlretrieve(url, str(cache_path))
        logger.info("Cached C-Scores at %s", cache_path)
    else:
        logger.info("Using cached C-Scores: %s", cache_path)

    data = np.load(cache_path, allow_pickle=True)
    if "scores" not in data.files:
        raise RuntimeError(
            f"C-Score npz at {cache_path} has unexpected layout "
            f"(no 'scores' field). Files: {data.files}."
        )
    cscores = np.asarray(data["scores"], dtype=np.float64)
    return 1.0 - cscores


# ---------------------------------------------------------------------------
# DatasetSpec
# ---------------------------------------------------------------------------

class _Strict(BaseModel):
    """Local strict-config base: unknown keys are request errors, not typos."""

    model_config = ConfigDict(extra="forbid")


class DatasetSpec(_Strict):
    """Declarative dataset request resolved by the library's split builders.

    ``carveout_seed`` drives the per-class 90/10 train/val carve-out;
    ``split_seed`` drives the random-mode forget permutation. Both reproduce
    the fixture-era the original research pipeline recipe byte-exactly.
    """

    name: str = "cifar10"  # any dataset registered in data/datasets.py
    data_dir: str = "./data"
    mode: Literal["single_class", "random", "sub_class_atypical"] = "single_class"
    forget_class: int | None = 0
    forget_fraction: float | None = None
    split_seed: int = 0
    carveout_seed: int = 42
    val_fraction: float = 0.1

    @model_validator(mode="after")
    def _check_mode_fields(self) -> "DatasetSpec":
        if self.mode in ("random", "sub_class_atypical"):
            if self.forget_fraction is None:
                raise ValueError(
                    f"mode={self.mode!r} requires forget_fraction to be set."
                )
            # Ported guard from the original data pipeline.
            if not (0.0 < self.forget_fraction < 1.0):
                raise ValueError(
                    f"forget_fraction must be in (0, 1); got {self.forget_fraction!r}."
                )
        if self.mode in ("single_class", "sub_class_atypical"):
            # Ported guards from the original data pipeline.
            if self.forget_class is None:
                raise ValueError(f"mode={self.mode!r} requires forget_class to be set.")
        # F2: validate the dataset name against the registry and bound
        # forget_class by the dataset's own class count (cifar10 -> 10, identical
        # to the legacy len(CIFAR10_CLASSES) bound).
        if self.name not in DATASET_REGISTRY:
            raise ValueError(
                f"unknown dataset {self.name!r}; registered: {sorted(DATASET_REGISTRY)}")
        n_classes = num_classes_for(self.name)
        if self.forget_class is not None and not (0 <= self.forget_class < n_classes):
            raise ValueError(
                f"forget_class must be in [0, {n_classes}); got {self.forget_class!r}."
            )
        if not (0.0 <= self.val_fraction < 1.0):
            raise ValueError(f"val_fraction must be in [0, 1); got {self.val_fraction!r}.")
        return self


# ---------------------------------------------------------------------------
# SplitBundle
# ---------------------------------------------------------------------------

@dataclass
class SplitBundle:
    """Resolved evaluation splits: canonical loaders plus identity metadata.

    ``forget_test`` / ``retain_test`` exist only in class-forgetting modes;
    in ``random`` / ``sub_class_atypical`` they are None (empty by design —
    accessing them raises ``SplitNotAvailable`` -> ``not_applicable_mode``).
    ``ids`` maps split name to ordered example ids: original-CIFAR-train
    indices for ``forget``/``retain`` (in loader order) and test-set indices
    for the test splits. ``spec`` is the originating ``DatasetSpec`` when the
    bundle was built by ``resolve_dataset_spec`` (None for raw user bundles);
    it powers ``train_view``.
    """

    forget: DataLoader
    retain: DataLoader
    test: DataLoader
    forget_test: DataLoader | None = None
    retain_test: DataLoader | None = None
    ids: dict[str, np.ndarray] | None = None
    dataset_id: str | None = None
    mode: str = "single_class"
    spec: DatasetSpec | None = None
    _train_views: dict = field(default_factory=dict, repr=False)

    _SPLITS = ("forget", "retain", "test", "forget_test", "retain_test")

    def available(self, split: str) -> bool:
        """True iff ``split`` carries a loader in this bundle."""
        if split not in self._SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {self._SPLITS}")
        return getattr(self, split) is not None

    def get(self, split: str) -> DataLoader:
        """Return the loader for ``split``; raise ``SplitNotAvailable`` if the
        split is empty by design in this forgetting mode."""
        if split not in self._SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {self._SPLITS}")
        loader = getattr(self, split)
        if loader is None:
            raise SplitNotAvailable(split, self.mode)
        return loader

    def train_view(self, split: str) -> Dataset:
        """Train-transform (augmented) dataset view of a TRAIN-side split.

        For attack-side fine-tuning only (relearning M11/M12) —
        evaluation paths stay on the canonical augmentation-stripped loaders.
        For spec-resolved bundles this
        delegates to ``train_view_dataset(self.spec, self.ids[split])``
        (memoized per split), so attacks fine-tune under the legacy
        RandomCrop/RandomHorizontalFlip train recipe.

        Documented degradation: a raw user bundle without an originating
        ``DatasetSpec`` has no augmented source of truth, so the canonical
        (augmentation-stripped) dataset is returned with a logged warning —
        the attack then trains without train-time augmentation, disclosed,
        never silent.

        Raises:
            SplitNotAvailable: for non-train splits (only ``forget`` /
                ``retain`` have train views) or splits empty by design.
        """
        if split not in ("forget", "retain"):
            raise SplitNotAvailable(split, self.mode)
        loader = self.get(split)  # SplitNotAvailable when empty by design
        if self.spec is not None and self.ids is not None and split in self.ids:
            if split not in self._train_views:
                self._train_views[split] = train_view_dataset(
                    self.spec, self.ids[split])
            return self._train_views[split]
        logger.warning(
            "SplitBundle.train_view(%r): bundle has no originating "
            "DatasetSpec; returning the canonical (augmentation-stripped) "
            "dataset — attack fine-tuning runs without train-time "
            "augmentation", split)
        return loader.dataset

    @classmethod
    def from_npz(cls, paths: dict[str, str], *, seed: int,
                 mode: str = "single_class", batch_size: int = 256,
                 num_workers: int = 0) -> "SplitBundle":
        """Build a bundle from per-split ``.npz`` tensor dumps (Tier-A data).

        Each ``.npz`` carries arrays ``X`` (features ``[N, ...]``) and ``y``
        (int labels ``[N]``). ``paths`` maps split name -> file:
        ``forget``/``retain``/``test`` required; ``forget_test``/``retain_test``
        optional (single-class). Loaders are deterministic
        (``shuffle=False`` + seeded generator), matching ``resolve_dataset_spec``.

        ``ids`` are assigned as a single running offset across splits
        (forget ``0..n_f-1``, retain ``n_f..``, ...) so the forget/retain
        disjointness check (and fingerprints) are meaningful for genuinely
        disjoint user data. ``spec`` is None (no augmented source -> attack
        fine-tuning uses the supplied tensors as-is).
        """
        import torch
        from torch.utils.data import TensorDataset
        from trail.core.hashing import sha256_bytes

        loaders: dict[str, DataLoader | None] = {}
        ids: dict[str, np.ndarray] = {}
        digest = []  # content-stable dataset_id across runs
        offset = 0
        for split in cls._SPLITS:
            path = paths.get(split)
            if path is None:
                loaders[split] = None
                continue
            with np.load(path) as data:
                if "X" not in data or "y" not in data:
                    raise RequestError(
                        f"{path}: .npz must contain 'X' and 'y'; "
                        f"found {list(data.files)}")
                X = np.asarray(data["X"], dtype=np.float32)
                y = np.asarray(data["y"], dtype=np.int64)
            if len(X) != len(y):
                raise RequestError(f"{path}: len(X)={len(X)} != len(y)={len(y)}")
            ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
            ids[split] = np.arange(offset, offset + len(ds), dtype=np.int64)
            offset += len(ds)
            digest.append(f"{split}:{len(ds)}:"
                          f"{sha256_bytes(y.tobytes())[:16]}")
            kwargs: dict = {}
            if num_workers > 0:
                kwargs["worker_init_fn"] = make_worker_init_fn(seed, "loader")
            loaders[split] = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers,
                generator=torch_generator(seed, "loader"), **kwargs)

        for req in ("forget", "retain", "test"):
            if loaders.get(req) is None:
                raise RequestError(f"from_npz: required split {req!r} missing "
                                   f"from paths {sorted(paths)}")
        dataset_id = "npz@" + sha256_bytes("|".join(digest).encode())[:16]
        return cls(
            forget=loaders["forget"], retain=loaders["retain"],
            test=loaders["test"], forget_test=loaders.get("forget_test"),
            retain_test=loaders.get("retain_test"),
            ids=ids, dataset_id=dataset_id, mode=mode, spec=None)

    @classmethod
    def from_h5(cls, path: str, *, seed: int, mode: str = "single_class",
                batch_size: int = 256, num_workers: int = 0) -> "SplitBundle":
        """Build a bundle from a single HDF5 file with per-split groups.

        Layout: ``/<split>/X`` + ``/<split>/y`` for each split (same split
        names and semantics as :meth:`from_npz`). ``h5py`` is an optional
        dependency, imported lazily; a clear error is raised if absent.
        """
        try:
            import h5py  # optional dependency
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RequestError(
                "SplitBundle.from_h5 requires h5py (`pip install h5py`)") from exc
        import torch
        from torch.utils.data import TensorDataset
        from trail.core.hashing import sha256_bytes

        loaders: dict[str, DataLoader | None] = {}
        ids: dict[str, np.ndarray] = {}
        digest = []
        offset = 0
        with h5py.File(path, "r") as h5:
            for split in cls._SPLITS:
                if split not in h5:
                    loaders[split] = None
                    continue
                grp = h5[split]
                if "X" not in grp or "y" not in grp:
                    raise RequestError(
                        f"{path}:/{split} must contain 'X' and 'y'")
                X = np.asarray(grp["X"], dtype=np.float32)
                y = np.asarray(grp["y"], dtype=np.int64)
                ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
                ids[split] = np.arange(offset, offset + len(ds), dtype=np.int64)
                offset += len(ds)
                digest.append(f"{split}:{len(ds)}:"
                              f"{sha256_bytes(y.tobytes())[:16]}")
                kwargs: dict = {}
                if num_workers > 0:
                    kwargs["worker_init_fn"] = make_worker_init_fn(seed, "loader")
                loaders[split] = DataLoader(
                    ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers,
                    generator=torch_generator(seed, "loader"), **kwargs)
        for req in ("forget", "retain", "test"):
            if loaders.get(req) is None:
                raise RequestError(f"from_h5: required split {req!r} missing")
        dataset_id = "h5@" + sha256_bytes("|".join(digest).encode())[:16]
        return cls(
            forget=loaders["forget"], retain=loaders["retain"],
            test=loaders["test"], forget_test=loaders.get("forget_test"),
            retain_test=loaders.get("retain_test"),
            ids=ids, dataset_id=dataset_id, mode=mode, spec=None)


# ---------------------------------------------------------------------------
# Split builders (byte-exact ports)
# ---------------------------------------------------------------------------

def _carveout_train_indices(train_targets: np.ndarray, carveout_seed: int,
                            val_fraction: float, num_classes: int) -> list[int]:
    """Per-class validation carve-out; returns surviving train indices, sorted.

    BYTE-EXACT port of the original loader: same RandomState consumption order (classes
    ``0..num_classes-1`` in order, ``rng.choice(class_idx,
    int(val_fraction*len), replace=False)``), same ``sorted(set - set)``
    reconstruction. F2: ``num_classes`` was the hardcoded ``len(CIFAR10_CLASSES)``
    — passing ``num_classes_for(spec.name)`` keeps ``range(10)`` for cifar10
    (byte-identical RNG order); a different dataset iterates its own class count.
    """
    rng = np.random.RandomState(carveout_seed)  # sanctioned byte-exact exception
    valid_idx = []
    for i in range(num_classes):
        class_idx = np.where(train_targets == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(val_fraction * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_idx = sorted(set(range(len(train_targets))) - set(valid_idx))
    return [int(i) for i in train_idx]


# ---------------------------------------------------------------------------
# Built-in forgetting-mode resolvers.
#
# Each resolver is the byte-exact split body ported from
# the original data pipeline, lifted VERBATIM out of the old
# ``_split_positions`` if/elif chain — same statements, same order, same
# ``RandomState`` construction/consumption — and registered into the F1
# ``MODE_REGISTRY`` (data/modes.py). ``_split_positions`` is now a thin
# dispatcher over the registry; the G10 split fixtures
# (trail/fixtures/splits/*.json) are unaffected by the indirection.
#
# Resolver signature: ``(spec, train_targets, train_idx, targets_carved) ->
# (forget_pos, retain_pos)`` where ``targets_carved == train_targets[train_idx]``
# is precomputed once by the dispatcher. Returned positions index into the
# carved trainset; the dispatcher normalizes both to ``int64``.
# ---------------------------------------------------------------------------

@register_mode(name="single_class", requires=("forget_class",),
               yields_forget_test=True, id_params=("forget_class",),
               builtin=True)
def _single_class_split(spec: "DatasetSpec", train_targets: np.ndarray,
                        train_idx: list[int],
                        targets_carved: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Port of split_by_class — the original data pipeline.
    forget_pos = np.where(targets_carved == spec.forget_class)[0]
    retain_pos = np.where(targets_carved != spec.forget_class)[0]
    return forget_pos, retain_pos


@register_mode(name="random", requires=("forget_fraction",),
               yields_forget_test=False, id_params=("forget_fraction", "split_seed"),
               builtin=True)
def _random_split(spec: "DatasetSpec", train_targets: np.ndarray,
                  train_idx: list[int],
                  targets_carved: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Port of split_randomly — the original data pipeline.
    n_total = len(train_idx)
    rng = np.random.RandomState(int(spec.split_seed))  # sanctioned exception
    permuted = rng.permutation(n_total)
    n_forget = int(n_total * spec.forget_fraction)
    forget_pos = permuted[:n_forget]
    retain_pos = permuted[n_forget:]
    return forget_pos, retain_pos


@register_mode(name="sub_class_atypical",
               requires=("forget_class", "forget_fraction"),
               yields_forget_test=False,
               id_params=("forget_class", "forget_fraction"), builtin=True)
def _sub_class_atypical_split(spec: "DatasetSpec", train_targets: np.ndarray,
                              train_idx: list[int],
                              targets_carved: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Port of split_atypical_subclass — the original data pipeline.
    # DEVIATION (deliberate, user-approved; see module docstring): the
    # C-Score array is indexed by ORIGINAL-CIFAR indices (corrected
    # semantics). Legacy callers always hit the re-indexed else-branch
    # (the original data pipeline) and indexed the original-order array
    # with carved-trainset positions — an indexing artifact that
    # misranked atypicality (~9% forget-set overlap with this path).
    # No regression fixture covers this mode.
    memorization = load_or_download_cscores(
        spec.name, Path(spec.data_dir) / "cscores")
    trainset_orig_indices = np.asarray(train_idx)
    class_orig = np.where(train_targets == int(spec.forget_class))[0]
    in_train_mask = np.isin(class_orig, trainset_orig_indices)
    class_in_train_orig = class_orig[in_train_mask]
    # Sort by memorization descending: most atypical first.
    class_mem = memorization[class_in_train_orig]
    order = np.argsort(class_mem)[::-1]
    n_forget = int(len(class_in_train_orig) * spec.forget_fraction)
    forget_orig = class_in_train_orig[order[:n_forget]]
    orig_to_pos = {int(o): int(p) for p, o in enumerate(trainset_orig_indices)}
    forget_pos = np.asarray([orig_to_pos[int(o)] for o in forget_orig], dtype=np.int64)
    forget_set = set(forget_pos.tolist())
    retain_pos = np.asarray(
        [p for p in range(len(train_idx)) if p not in forget_set], dtype=np.int64)
    return forget_pos, retain_pos


def _split_positions(spec: DatasetSpec, train_targets: np.ndarray,
                     train_idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Resolve (forget, retain) positions *within the carved trainset*, in
    protocol-native order (ascending for single_class, permutation order for
    random, memorization-descending for sub_class_atypical).

    Dispatches on the F1 mode registry: ``targets_carved`` is computed once
    (exactly as in the pre-registry implementation), then
    ``MODE_REGISTRY[spec.mode].split_fn`` resolves the raw positions. The
    built-in resolvers hold the byte-exact split bodies verbatim, so the G10
    fixtures are unaffected by the registry indirection.
    """
    targets_carved = train_targets[np.asarray(train_idx)]
    mode_spec = MODE_REGISTRY.get(spec.mode)
    if mode_spec is None:  # unregistered mode (DatasetSpec Literal also guards)
        raise RequestError(f"unknown forgetting mode {spec.mode!r}")
    forget_pos, retain_pos = mode_spec.split_fn(
        spec, train_targets, train_idx, targets_carved)
    return (np.asarray(forget_pos, dtype=np.int64),
            np.asarray(retain_pos, dtype=np.int64))


def _dataset_id(spec: DatasetSpec) -> str:
    """Stable split-identity string (no data_dir — location is not identity)."""
    return (
        f"{spec.name}@mode={spec.mode},forget_class={spec.forget_class!r},"
        f"forget_fraction={spec.forget_fraction!r},split_seed={spec.split_seed!r},"
        f"carveout_seed={spec.carveout_seed!r},val_fraction={spec.val_fraction!r}"
    )


def resolve_dataset_spec(spec: DatasetSpec, *, batch_size: int = 256,
                         num_workers: int = 0, base_seed: int = 0) -> SplitBundle:
    """Resolve a ``DatasetSpec`` into canonical evaluation loaders.

    The trainset is a ``Subset`` of a torchvision CIFAR10 built with the
    *test* transform — canonical (augmentation-stripped) by default, because
    this is an evaluation library. Attack fine-tuning gets the
    augmented view via ``SplitBundle.train_view`` (backed by
    ``train_view_dataset``; the resolved spec is attached to the bundle for
    this purpose). All loaders are shuffle=False with framework-seeded
    generators (deterministic loader order).

    Regression caveat (documented parity note): legacy panels evaluated
    train-side splits through the AUGMENTED train transform with
    shuffle=True. Train-side accuracies from legacy panel outputs therefore
    carry a systematic augmentation delta vs the canonical views built here
    — recompute legacy numbers on
    canonical (aug-stripped) views before any ±1 pp comparison; never
    compare against augmented-view legacy panel numbers directly.

    Mode dispatch ported from get_forget_retain_loaders
    (the original data pipeline): single_class derives
    forget/retain test partitions; random / sub_class_atypical have no
    forget-test counterpart by design (forget_test/retain_test are None and
    callers use the full ``test`` split).
    """
    test_transform = get_test_transform(spec.name)
    built = build_datasets(spec.name, spec.data_dir,
                           train_transform=test_transform,
                           test_transform=test_transform)
    base_train, base_test = built["train"], built["test"]
    train_targets = _targets_of(base_train)
    test_targets = _targets_of(base_test)

    train_idx = _carveout_train_indices(train_targets, spec.carveout_seed,
                                        spec.val_fraction,
                                        num_classes_for(spec.name))
    trainset = Subset(base_train, train_idx)

    forget_pos, retain_pos = _split_positions(spec, train_targets, train_idx)
    forget_ds = Subset(trainset, [int(p) for p in forget_pos])
    retain_ds = Subset(trainset, [int(p) for p in retain_pos])

    orig = np.asarray(train_idx, dtype=np.int64)
    ids: dict[str, np.ndarray] = {
        "forget": orig[forget_pos],
        "retain": orig[retain_pos],
        "test": np.arange(len(base_test), dtype=np.int64),
    }

    forget_test_ds: Dataset | None = None
    retain_test_ds: Dataset | None = None
    if spec.mode == "single_class":
        ft_idx = np.where(test_targets == spec.forget_class)[0].astype(np.int64)
        rt_idx = np.where(test_targets != spec.forget_class)[0].astype(np.int64)
        forget_test_ds = Subset(base_test, [int(i) for i in ft_idx])
        retain_test_ds = Subset(base_test, [int(i) for i in rt_idx])
        ids["forget_test"] = ft_idx
        ids["retain_test"] = rt_idx
        logger.info(
            "single_class split (class %r): forget %d train / %d test; "
            "retain %d train / %d test",
            class_names_for(spec.name)[spec.forget_class], len(forget_ds),
            len(ft_idx), len(retain_ds), len(rt_idx))
    else:
        logger.info(
            "%s split: forget %d train / 0 test (no test counterpart by design); "
            "retain %d train; test %d (full)",
            spec.mode, len(forget_ds), len(retain_ds), len(base_test))

    def _loader(ds: Dataset) -> DataLoader:
        kwargs: dict = {}
        if num_workers > 0:
            kwargs["worker_init_fn"] = make_worker_init_fn(base_seed, "loader")
        return DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            generator=torch_generator(base_seed, "loader"), **kwargs)

    return SplitBundle(
        forget=_loader(forget_ds),
        retain=_loader(retain_ds),
        test=_loader(base_test),
        forget_test=_loader(forget_test_ds) if forget_test_ds is not None else None,
        retain_test=_loader(retain_test_ds) if retain_test_ds is not None else None,
        ids=ids,
        dataset_id=_dataset_id(spec),
        mode=spec.mode,
        spec=spec,
    )


def forget_indices(spec: DatasetSpec) -> np.ndarray:
    """Resolved forget split as sorted positions within the carved trainset.

    This is the regression-fixture view (trail/fixtures/splits/*.json store
    sorted carved-trainset positions, 0..44999 for the default 90/10
    carve-out), NOT the original-CIFAR ids exposed in ``SplitBundle.ids``.
    """
    base_train = build_datasets(spec.name, spec.data_dir,
                                splits=("train",))["train"]
    train_targets = _targets_of(base_train)
    train_idx = _carveout_train_indices(train_targets, spec.carveout_seed,
                                        spec.val_fraction,
                                        num_classes_for(spec.name))
    forget_pos, _ = _split_positions(spec, train_targets, train_idx)
    return np.sort(forget_pos).astype(np.int64)


def train_view_dataset(spec: DatasetSpec, indices: Sequence[int] | np.ndarray) -> Subset:
    """Train-transform (augmented) view over ORIGINAL-CIFAR-train ``indices``.

    For attack-side fine-tuning (relearning recovery) only;
    evaluation paths must stay on the canonical aug-stripped view. Pass
    original-CIFAR-train ids, e.g. ``bundle.ids['forget']``.
    """
    base = build_datasets(spec.name, spec.data_dir, splits=("train",),
                          train_transform=get_train_transform(spec.name))["train"]
    return Subset(base, [int(i) for i in indices])
