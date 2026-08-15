"""Request schema: the pydantic models ARE the configuration language.

YAML + dotted CLI overrides resolve through OmegaConf and validate against
these models with ``extra="forbid"`` — unknown keys are errors, not surprises.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Literal, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trail.core.errors import DisclosureError
from trail.data.specs import DatasetSpec, SplitBundle

logger = logging.getLogger("trail.request")


class _Strict(BaseModel):
    """Base for every config model: unknown keys are validation errors."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

class CheckpointSet(_Strict):
    """Role-named checkpoint paths.

    Also accepts the documented positional 3-sequence
    ``[original, gold, unlearned]`` with ``None`` placeholders, and the alias
    field name ``retained`` for ``gold`` (accepted on input, never emitted —
    "retained" collides with the retain *split* and is banned from reports).
    """

    unlearned: str
    original: str | None = None
    gold: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if isinstance(data, (list, tuple)):
            if len(data) != 3:
                raise ValueError(
                    "checkpoint sequence must have exactly 3 entries in the "
                    "order [original, gold, unlearned] (None for absent roles)")
            return {"original": data[0], "gold": data[1], "unlearned": data[2]}
        if isinstance(data, dict) and "retained" in data:
            data = dict(data)
            retained = data.pop("retained")
            if data.get("gold") is not None and data["gold"] != retained:
                raise ValueError(
                    "both 'gold' and its alias 'retained' were supplied with "
                    "different values; use 'gold' only")
            if retained is not None:
                data["gold"] = retained
        return data


# ---------------------------------------------------------------------------
# Hyperparameters (every default is part of the versioned protocol)
# ---------------------------------------------------------------------------

class ReferencesHP(_Strict):
    """``gold``: "off" | "build" | <path to a gold checkpoint>; ``shadow``: 0 | 8.

    ``gold_ensemble`` / ``unlearned_ensemble`` are the KLoM (M68) inputs: lists
    of checkpoint paths. The gold ensemble (retrained-from-scratch oracles) is
    method-INDEPENDENT and content-addressed, so it amortizes across methods;
    the unlearned ensemble is the METHOD's own checkpoints across seeds. Both
    empty (the default) means KLoM skips (``reference_disabled:gold``).
    """

    gold: str = "off"
    shadow: int = 0
    gold_ensemble: list[str] = Field(default_factory=list)
    unlearned_ensemble: list[str] = Field(default_factory=list)


class ShadowHP(_Strict):
    """Shadow-ensemble training recipe for the opt-in LiRA tier (M9).

    The ensemble SIZE is ``hp.references.shadow`` (0 = LiRA disabled; 8 = the
    pinned budget). This recipe is a frozen protocol for cross-method
    comparability, but it is overridable AND fully participates in the L3 cache
    key, so a changed recipe can never serve a stale ensemble. Defaults are
    CIFAR/ResNet scale; the per-shadow training set is the full retain pool
    (always-in filler) plus an ``in_fraction`` random slice of the audit pool
    (forget+test), trained on the canonical (aug-stripped) views for
    determinism. See references/shadow.py.
    """

    epochs: int = 30
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    batch_size: int = 128
    #: fraction of the audit pool (forget+test) that is IN each shadow's
    #: training set; the complement is OUT (gives per-example in/out coverage).
    in_fraction: float = 0.5


class MiaHP(_Strict):
    default: str = "threshold_population"
    signals: list[str] = Field(default_factory=lambda: [
        "correctness", "confidence", "entropy", "m_entropy"])
    #: Operating points for strong MIA reporting: TPR is reported at each
    #: of these false-positive rates, alongside AUC (principles.md §C).
    fpr_targets: list[float] = Field(default_factory=lambda: [0.001, 0.01, 0.1])
    #: LiRA (M9) non-member basis. ``"test"`` (default, protocol pin): the full
    #: held-out test set. ``"class_matched"``: only test examples whose class is
    #: in the forget set — for single-class forgetting this is the held-out
    #: forget-class test set (forget_test), isolating *membership* from
    #: class-identity. In random/uniform forgetting (forget spans all classes)
    #: the two coincide. Selection is post-hoc over the SAME shadow ensemble
    #: (no rebuild — the L3 key is unchanged).
    lira_nonmembers: str = "test"


class RelearnHP(_Strict):
    budgets: list[int | str] = Field(default_factory=lambda: [0, 10, 100, "full"])
    epochs: int = 10
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    batch_size: int = 128
    #: fraction of the retain pool mixed into retain-sourced attack sets
    #: (frozen-protocol knob read by metrics/relearning.py).
    retain_fraction: float = 1.0
    #: D2D (M13): fraction of the forget set used as the attack/relearn slice;
    #: the disjoint remainder is the eval slice. 0.1 = the canonical 1%/9% of a
    #: 10% forget set (Fan et al. 2025).
    d2d_relearn_fraction: float = 0.1

    @model_validator(mode="before")
    @classmethod
    def _reject_source_key(cls, values: Any) -> Any:
        """The attack source is selected by METRIC NAME, never by hp.

        A ``source`` knob here validated, was disclosed, and did nothing —
        requesting ``relearn_forget`` with ``source: retain_mix`` silently
        re-ran the forget-only attack. Fail fast
        with the mapping instead.
        """
        if isinstance(values, dict) and "source" in values:
            raise ValueError(
                "hp.relearn.source does not exist: the attack source is "
                "selected by metric name — request 'relearn_forget' for "
                "forget_only (M11), 'relearn_retain_mix' for retain_mix "
                "(M12), or 'relearn_d2d' for the D2D 1%/9% split (M13) — "
                "and remove the 'source' key from the relearn block."
            )
        return values


class WhiteboxHP(_Strict):
    """White-box MIA recipe (Tier-2 attack family): the gradient-norm signal
    differentiates a deepcopy of the model on the audit examples. A frozen but
    legitimately-varying protocol knob — disclose it or acknowledge the defaults."""

    grad_layer: str = "last"   # "last" (classifier head) or a named module
    loss: str = "ce"


class PoisonHP(_Strict):
    """Poisoning / backdoor / witch recipe (Tier-2). End-to-end is BLOCKED on
    externally-produced poisoned checkpoints (the repo never trains — scope
    firewall); these knobs configure the synthetic/feasibility path and the
    artifact-input contract."""

    trigger_value: float = 1.0
    target_class: int = 0


class AdvHP(_Strict):
    """Adversarial-robustness recipe (Tier-3): FGSM/PGD perturbation budget. A
    frozen-but-varying protocol knob — disclosed via the ``adv`` family before
    any adversarial metric runs."""

    eps: float = 0.03
    alpha: float = 0.008
    steps: int = 7


class DownstreamSpec(_Strict):
    """Downstream transfer dataset for ``knn_transfer`` (Phase 5). ``name`` is a
    registered dataset (data/datasets.py); the kNN probe caps at ``n_samples``,
    does a seeded ``test_fraction`` split, and votes over ``k`` neighbors under
    the ``metric`` ('cosine' | 'l2')."""

    name: str = "cifar100"
    data_dir: str = "./data"
    n_samples: int = 2000
    k: int = 5
    test_fraction: float = 0.2
    metric: str = "cosine"


class BootstrapHP(_Strict):
    n: int = 1000
    alpha: float = 0.05


class KlomHP(_Strict):
    """KLoM (M68) histogram recipe — the Data-Unlearn-Bench defaults (Rinberg
    et al. 2026, arXiv 2602.16400): ``bins`` fixed-width bins over the clipped
    ``[clip_low, clip_high]`` margin range, with Laplace smoothing ``eps`` so
    empty bins stay finite (this caps KLoM at ~12). ``splits`` selects which
    partitions to score — the headline is the forget split; the rest are
    reported as ``klom_<split>`` components. The ensemble SIZES are the lengths
    of ``hp.references.gold_ensemble`` / ``unlearned_ensemble`` (the paper uses
    100 each; a reduced multi-seed budget is a documented, non-comparable proxy).
    """

    bins: int = 20
    clip_low: float = -100.0
    clip_high: float = 100.0
    eps: float = 1e-5
    splits: list[str] = Field(default_factory=lambda: ["forget", "retain", "test"])


class Hyperparams(_Strict):
    """The versioned protocol hyperparameter block."""

    references: ReferencesHP = Field(default_factory=ReferencesHP)
    mia: MiaHP = Field(default_factory=MiaHP)
    relearn: RelearnHP = Field(default_factory=RelearnHP)
    whitebox: WhiteboxHP = Field(default_factory=WhiteboxHP)
    poison: PoisonHP = Field(default_factory=PoisonHP)
    adv: AdvHP = Field(default_factory=AdvHP)
    downstream: DownstreamSpec = Field(default_factory=DownstreamSpec)
    shadow: ShadowHP = Field(default_factory=ShadowHP)
    bootstrap: BootstrapHP = Field(default_factory=BootstrapHP)
    klom: KlomHP = Field(default_factory=KlomHP)
    metric_overrides: dict[str, dict] = Field(default_factory=dict)
    #: Explicit acknowledgment that the frozen-protocol defaults are accepted
    #: for legitimately-varying knobs (the relearning recipe). Required — in
    #: lieu of setting the knobs explicitly — when an attack metric actually
    #: runs; enforced by :func:`check_disclosure` (controlled-freedom).
    accept_protocol_defaults: bool = False
    #: The unlearning guarantee the user's METHOD targets, if any
    #: (exact | approximate | certified | empirical). Recorded in the report's
    #: disclosure block (principles.md §E5). The FRAMEWORK itself only certifies
    #: empirical evaluation — see EvalReport.evaluation_guarantee.
    method_guarantee: str | None = None


# ---------------------------------------------------------------------------
# Disclosure contract (controlled freedom): a user must
# consciously disclose the legitimately-varying knobs rather than silently
# inheriting them. Enforced for attack metrics that actually run (model-in).
# ---------------------------------------------------------------------------

#: Attack metric -> the hp recipe knob it must disclose. Each attack FAMILY
#: keys on its OWN recipe (Tier-2 must-fix: extend the disclosure map BEFORE
#: adding new attack metrics, else a whitebox/poison panel keys on the wrong
#: ``relearn`` knob). The poison family's end-to-end run is blocked on external
#: poisoned checkpoints, but its disclosure contract is wired now.
_ATTACK_RECIPES: dict[str, str] = {
    "relearn_forget": "relearn", "relearn_retain_mix": "relearn",
    "relearn_d2d": "relearn", "efficacy_vs_compute": "relearn",
    "mia_whitebox_gradient": "whitebox", "mia_worst_case": "whitebox",
    "gus_gaussian": "poison", "backdoor_trigger_asr": "poison",
    "witchbrew_asr": "poison",
    "adv_robustness_fgsm": "adv", "adv_robustness_pgd": "adv",
}

#: Backward-compatible alias: the set of attack metrics requiring disclosure.
_ATTACK_METRICS: frozenset[str] = frozenset(_ATTACK_RECIPES)

#: recipe family -> the Hyperparams field that, when user-set, discloses it.
_RECIPE_HP_FIELD: dict[str, str] = {
    "relearn": "relearn", "whitebox": "whitebox", "poison": "poison",
    "adv": "adv"}


def _recipes_in_panel(metrics: "list[str] | str", task: str) -> set[str]:
    """The attack-recipe FAMILIES a resolved panel will run.

    For ``"default"``, the classification default panel includes the relearning
    attacks (M11/M13) while the LLM panel does not (encoded here to avoid
    importing the adapter at validation time); the white-box/poison families are
    external (never in the default panel). For an explicit list, by membership.
    """
    if isinstance(metrics, str):  # "default"
        return {"relearn"} if task == "classification" else set()
    return {_ATTACK_RECIPES[m] for m in metrics if m in _ATTACK_RECIPES}


def _panel_has_attack(metrics: "list[str] | str", task: str) -> bool:
    """Whether the resolved panel runs any disclosure-requiring attack."""
    return bool(_recipes_in_panel(metrics, task))


def disclosure_block(hp: "Hyperparams") -> dict[str, str]:
    """Per-knob disclosure record: "explicit" (user-set) vs "protocol_default".

    Uses pydantic ``model_fields_set`` to distinguish a user-supplied recipe
    block from an inherited default, per attack family. A relearning
    ``metric_overrides`` entry also counts as an explicit relearn disclosure.
    """
    fs = hp.model_fields_set
    block: dict[str, str] = {}
    for recipe, field in _RECIPE_HP_FIELD.items():
        explicit = field in fs
        if recipe == "relearn":
            explicit = explicit or any(
                k in (hp.metric_overrides or {})
                for k, r in _ATTACK_RECIPES.items() if r == "relearn")
        block[f"{recipe}_recipe"] = "explicit" if explicit else "protocol_default"
    block["mia.default"] = "explicit" if "mia" in fs else "protocol_default"
    block["references"] = "explicit" if "references" in fs else "protocol_default"
    block["accept_protocol_defaults"] = str(bool(hp.accept_protocol_defaults))
    block["method_guarantee"] = hp.method_guarantee or "unstated"
    return block


def check_disclosure(request: "EvalRequest") -> dict[str, str]:
    """Enforce the disclosure contract and return the disclosure block.

    Raises :class:`DisclosureError` when a model-in panel runs an attack whose
    recipe FAMILY (relearn / whitebox / poison) was neither set explicitly nor
    acknowledged via ``hp.accept_protocol_defaults=True`` — keyed per family, so
    disclosing the relevant recipe clears exactly that family. Outputs-in
    requests never raise (attack metrics skip with ``requires_model_in``), but
    the block is still recorded. Idempotent besides the raise.
    """
    block = disclosure_block(request.hp)
    if request.input_mode == "model" and not request.hp.accept_protocol_defaults:
        undisclosed = sorted(
            r for r in _recipes_in_panel(request.metrics, request.task)
            if block.get(f"{r}_recipe") != "explicit")
        if undisclosed:
            raise DisclosureError(
                f"this evaluation runs attack metrics whose recipe(s) "
                f"{undisclosed} are frozen but legitimately-varying protocol "
                "knobs; disclose each before running (set hp.<recipe> "
                "explicitly, e.g. hp.relearn / hp.whitebox / hp.poison), or set "
                "hp.accept_protocol_defaults=True to run the frozen defaults "
                "knowingly.")
    return block


# ---------------------------------------------------------------------------
# Runtime / cache / logging configuration
# ---------------------------------------------------------------------------

class RuntimeConfig(_Strict):
    device: str = "auto"
    batch_size: int | None = None
    num_workers: int = 4
    amp: bool | None = None
    #: Adapter construction. None = the adapter's own defaults
    #: (classification: arch="resnet18", dataset/num_classes from the data's
    #: dataset name via the registry). ``arch`` keys ARCH_REGISTRY; ``dataset``
    #: keys data/datasets.DATASET_REGISTRY; ``num_classes`` is cross-checked
    #: against ``num_classes_for(dataset)`` at request validation.
    arch: str | None = None
    num_classes: int | None = None
    dataset: str | None = None
    #: F5: enable the out-of-band artifact subsystem (npz/png/svg emission).
    #: Default off — artifacts are a heavy, opt-in side output; the scored panel
    #: never depends on them. Requires the ``trail[plots]`` extra for figures.
    plots: bool = False


class CacheConfig(_Strict):
    dir: str = ".trail_cache"
    readonly: bool = False
    disable: set[str] = Field(default_factory=set)


class LogConfig(_Strict):
    wandb: bool = False
    wandb_project: str = "trail"
    wandb_entity: str | None = None
    level: str = "INFO"


def _validate_adapter_config(runtime: "RuntimeConfig",
                             data_name: str | None) -> None:
    """Parse-time adapter-config check, shared by model-in and outputs-in.

    A head width that contradicts the dataset must fail FAST and clearly here,
    not as a late cryptic strict-load key/size-mismatch dump in
    ``ClassificationAdapter.load_checkpoint``. Two rules:

    * an explicit ``runtime.dataset`` must be a registered dataset;
    * an explicit ``runtime.num_classes`` must equal ``num_classes_for`` the
      resolved dataset (``runtime.dataset`` else ``data_name``).

    ``data_name`` is the ``DatasetSpec`` name for a model-in request, or None
    (outputs-in / raw user bundle) — there the dataset name can only come from
    ``runtime.dataset``. Every field is optional, so a default request (no
    adapter overrides) is a no-op. Raises ``RequestError``.
    """
    from trail.core.errors import RequestError
    from trail.data.datasets import DATASET_REGISTRY, num_classes_for

    if runtime.dataset is not None and runtime.dataset not in DATASET_REGISTRY:
        raise RequestError(
            f"runtime.dataset={runtime.dataset!r} is not a registered dataset; "
            f"registered: {sorted(DATASET_REGISTRY)}")
    nc = runtime.num_classes
    if nc is None:
        return
    name = runtime.dataset or data_name
    if name is None:
        return  # outputs-in / raw bundle without a dataset: nothing to check
    expected = num_classes_for(name)  # name is registry-validated (above / DatasetSpec)
    if nc != expected:
        raise RequestError(
            f"runtime.num_classes={nc} contradicts num_classes_for({name!r})="
            f"{expected}; omit num_classes (the adapter derives it from the "
            "dataset) or set it consistently")


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class ModelRequest(_Strict):
    """Canonical model-in request: trail loads checkpoints and runs probes."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    task: Literal["classification", "llm"] = "classification"
    mode: str = "single_class"
    seed: int
    checkpoints: CheckpointSet
    data: DatasetSpec | SplitBundle
    metrics: list[str] | Literal["default"] = "default"
    hp: Hyperparams = Field(default_factory=Hyperparams)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    self_reported_cost: dict | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_data(cls, data: Any) -> Any:
        # YAML configs carry `data:` as a plain mapping -> DatasetSpec.
        if isinstance(data, dict) and isinstance(data.get("data"), dict):
            data = dict(data)
            data["data"] = DatasetSpec(**data["data"])
        return data

    @model_validator(mode="after")
    def _check_mode_consistency(self) -> "ModelRequest":
        # Exact-protocol guard: request.mode and the data's own forgetting
        # mode are two sources of truth; a silent mismatch mis-bases the
        # mode-aware test metrics (ta/ra_test) and mislabels the report.
        data_mode = getattr(self.data, "mode", None)
        if data_mode is not None and data_mode != self.mode:
            raise ValueError(
                f"request mode={self.mode!r} does not match the data's "
                f"mode={data_mode!r}; set both to the same forgetting mode")
        return self

    @model_validator(mode="after")
    def _check_adapter_config(self) -> "ModelRequest":
        # F2: validate runtime.dataset/num_classes against the registry, keyed
        # on the DatasetSpec's own dataset name when present (a SplitBundle has
        # none -> only runtime.dataset can supply it).
        _validate_adapter_config(self.runtime, getattr(self.data, "name", None))
        return self

    @property
    def input_mode(self) -> str:
        """Always "model" for this request type."""
        return "model"

    def has_role(self, role: str) -> bool:
        """Whether a checkpoint was supplied for ``role``."""
        if role not in ("unlearned", "original", "gold"):
            raise ValueError(f"unknown checkpoint role {role!r}")
        return getattr(self.checkpoints, role) is not None


class OutputsRequest(_Strict):
    """Outputs-in request: the user ran their own model and submits per-example
    outputs in the documented payload format; trail only scores."""

    task: Literal["classification", "llm"] = "classification"
    mode: str = "single_class"
    seed: int
    forget_outputs: dict
    retain_outputs: dict
    test_outputs: dict
    forget_test_outputs: dict | None = None
    retain_test_outputs: dict | None = None
    gold_outputs: dict[str, dict] | None = None       # split -> payload
    original_outputs: dict[str, dict] | None = None   # split -> payload
    metrics: list[str] | Literal["default"] = "default"
    hp: Hyperparams = Field(default_factory=Hyperparams)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    self_reported_cost: dict | None = None

    @model_validator(mode="after")
    def _check_adapter_config(self) -> "OutputsRequest":
        # Symmetry with ModelRequest: an explicit runtime.dataset must be
        # registered and an explicit runtime.num_classes must match it. There is
        # no DatasetSpec here, so the dataset name comes only from runtime; with
        # neither set this is a no-op (num_classes is inferred from logits width
        # at scoring and the adapter's head width is unused in the outputs path).
        _validate_adapter_config(self.runtime, None)
        return self

    @property
    def input_mode(self) -> str:
        """Always "outputs" for this request type."""
        return "outputs"

    def has_role(self, role: str) -> bool:
        """Whether outputs were supplied for ``role``."""
        if role == "unlearned":
            return True
        if role == "gold":
            return bool(self.gold_outputs)
        if role == "original":
            return bool(self.original_outputs)
        raise ValueError(f"unknown checkpoint role {role!r}")

    def payload_for(self, role: str, split: str) -> dict | None:
        """Raw payload dict for (role, split), or None when absent."""
        if role == "unlearned":
            mapping: dict[str, dict | None] = {
                "forget": self.forget_outputs,
                "retain": self.retain_outputs,
                "test": self.test_outputs,
                "forget_test": self.forget_test_outputs,
                "retain_test": self.retain_test_outputs,
            }
            if split not in mapping:
                raise ValueError(f"unknown split {split!r}")
            return mapping[split]
        if role == "gold":
            return (self.gold_outputs or {}).get(split)
        if role == "original":
            return (self.original_outputs or {}).get(split)
        raise ValueError(f"unknown checkpoint role {role!r}")


EvalRequest = Union[ModelRequest, OutputsRequest]


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------

# Ported from the original research pipeline (_ENV_PATTERN /
# _substitute_env_vars) — ${VAR} and ${VAR:-default} substitution in strings.
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _substitute_env_vars(value: Any) -> Any:
    """Replace ``${VAR}`` and ``${VAR:-default}`` tokens inside YAML strings."""
    if isinstance(value, str):
        def repl(match: re.Match) -> str:
            name, default = match.group(1), match.group(2) or ""
            return os.environ.get(name, default)
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env_vars(v) for v in value]
    return value


def build_request_from_yaml(config_path: str,
                            overrides: Sequence[str] | None = None,
                            ) -> ModelRequest:
    """Resolve a YAML config + dotted overrides into a validated ModelRequest.

    Ported from the original research pipeline (build_config): OmegaConf
    load + ``from_dotlist`` merge + ``to_container`` + env-var substitution,
    then strict pydantic validation (the request models are the schema).
    """
    from omegaconf import OmegaConf  # heavyweight; imported on demand

    base = OmegaConf.load(config_path)
    dot_overrides = list(overrides or [])
    if dot_overrides:
        merged = OmegaConf.merge(base, OmegaConf.from_dotlist(dot_overrides))
    else:
        merged = base
    plain = OmegaConf.to_container(merged, resolve=False)
    if not isinstance(plain, dict):
        from trail.core.errors import RequestError
        raise RequestError(f"config {config_path} must be a mapping at the top level")
    plain = _substitute_env_vars(plain)
    logger.debug("config %s resolved with %d overrides", config_path,
                 len(dot_overrides))
    return ModelRequest.model_validate(plain)
