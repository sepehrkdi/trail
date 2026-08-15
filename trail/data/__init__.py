"""trail.data — declarative dataset specs, split builders, fingerprints."""
from trail.data.fingerprint import (
    check_disjoint,
    loader_indices,
    split_fingerprint,
    transform_repr,
    transform_sha,
)
from trail.data.datasets import (
    CLASS_NAMES,
    DATASET_REGISTRY,
    build_datasets,
    class_names_for,
    num_classes_for,
    register_dataset,
)
from trail.data.modes import (
    MODE_REGISTRY,
    ModeSpec,
    register_mode,
)
from trail.data.specs import (
    CIFAR10_CLASSES,
    DATASET_STATS,
    DatasetSpec,
    SplitBundle,
    forget_indices,
    get_normalization_stats,
    get_test_transform,
    get_train_transform,
    load_or_download_cscores,
    resolve_dataset_spec,
    train_view_dataset,
)

__all__ = [
    "CIFAR10_CLASSES",
    "CLASS_NAMES",
    "DATASET_REGISTRY",
    "DATASET_STATS",
    "MODE_REGISTRY",
    "DatasetSpec",
    "ModeSpec",
    "SplitBundle",
    "build_datasets",
    "check_disjoint",
    "class_names_for",
    "forget_indices",
    "get_normalization_stats",
    "get_test_transform",
    "get_train_transform",
    "load_or_download_cscores",
    "loader_indices",
    "num_classes_for",
    "register_dataset",
    "register_mode",
    "resolve_dataset_spec",
    "split_fingerprint",
    "train_view_dataset",
    "transform_repr",
    "transform_sha",
]
