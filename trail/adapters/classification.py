"""Classification adapter — trail v0.1.

CIFAR-scale image classification: ResNet architectures with the CIFAR stem,
strict checkpoint loading over the format zoo (flat / nested / pruned
weight-mask pairs), the deterministic fp32 forward probe, and constructive
augmentation stripping for canonical evaluation views.

Provenance (re-implemented here, not imported from any training codebase):
- CIFAR ResNet stem, Kaiming/constant init, architecture factories.
- Subset unwrapping + augmentation stripping, rebuilt constructively: rather
  than mutating a shared dataset in place, we build a fresh shallow-copied
  view and never touch user state.
- validate() semantics: eval mode, 0-100 accuracy scale, and a loud
  empty-loader guard instead of a silent 0.0.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader, Dataset, Subset

from trail.adapters.base import ModalityAdapter
from trail.core.errors import CheckpointError, RequestError
from trail.core.seeding import make_worker_init_fn, torch_generator
from trail.core.types import SplitOutputs

logger = logging.getLogger("trail.adapters.classification")

#: Tolerance for the outputs-in cross-check between user-supplied per-example
#: losses and losses re-derived from the supplied logits.
_LOSS_XCHECK_TOL = 1e-3

_DEFAULT_BATCH_SIZE = 256

#: F4: the canonical name of the single penultimate feature layer. ``features``
#: (the activation_distance adjunct) mirrors ``features_multi[_PENULTIMATE_KEY]``.
_PENULTIMATE_KEY = "penultimate"


@dataclass(frozen=True)
class _FeatureHook:
    """A feature-capture point for the multi-hook probe (F4/F3).

    ``module`` is the module to hook. ``pre=False`` captures the module OUTPUT
    (a forward hook — e.g. an AdaptiveAvgPool2d, CNNs/Swin/ConvNeXt). ``pre=True``
    captures the module INPUT (a forward PRE-hook) — used for ViT, where the
    penultimate CLS token is the tensor entering ``model.heads`` (torchvision
    extracts ``x[:, 0]`` inline, so no module emits it as an output). The
    captured tensor is flattened to ``[N, D]`` in the probe.
    """

    module: "nn.Module"
    pre: bool = False


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

#: name -> factory(num_classes) -> nn.Module
ARCH_REGISTRY: dict[str, Callable[[int], "nn.Module"]] = {}

#: F4 feature resolvers: arch name -> ``resolver(model) -> {layer_name: module}``.
#: The probe registers a forward hook on each returned module and captures its
#: output into ``SplitOutputs.features_multi``. Each arch ships a resolver so
#: that transformers (no AdaptiveAvgPool2d) still surface a penultimate feature
#: and ``activation_distance`` does not silently self-skip. An arch WITHOUT a
#: registered resolver falls back to ``_find_penultimate_pool`` (avgpool).
ARCH_FEATURE_RESOLVERS: dict[str, "Callable[[nn.Module], dict]"] = {}


def _avgpool_feature_resolver(model: "nn.Module") -> dict[str, "nn.Module"]:
    """Penultimate-pool resolver (avgpool output): ResNets (CIFAR & ImageNet
    stem), torchvision Swin, and ConvNeXt all place an AdaptiveAvgPool2d before
    the head. Identical capture target to the pre-F4 single-hook probe, so the
    ResNet feature blob is byte-unchanged."""
    pool = _find_penultimate_pool(model)
    return {_PENULTIMATE_KEY: pool} if pool is not None else {}


def _vit_feature_resolver(model: "nn.Module") -> dict:
    """ViT penultimate = the CLS token entering ``model.heads`` (torchvision
    does ``x = x[:, 0]; x = self.heads(x)``), captured via a pre-hook -> [N, D].
    LayerNorm-mean is the documented alternative; CLS is the canonical choice."""
    head = getattr(model, "heads", None)
    if head is None:
        return {}
    return {_PENULTIMATE_KEY: _FeatureHook(module=head, pre=True)}


def register_arch(name: str, factory: Callable[[int], "nn.Module"]) -> None:
    """Register an architecture factory under ``name``.

    Args:
        name: registry key, e.g. ``"resnet18"``.
        factory: callable ``factory(num_classes) -> nn.Module`` returning a
            freshly initialized model (weights are always overwritten by
            ``load_checkpoint`` via a strict state-dict load).

    Raises:
        ValueError: on duplicate registration.
    """
    if name in ARCH_REGISTRY:
        raise ValueError(f"duplicate arch registration: {name!r}")
    ARCH_REGISTRY[name] = factory


def _apply_cifar_tweaks(model: "nn.Module") -> "nn.Module":
    """CIFAR-adapt a torchvision ResNet.

    Ported from the original model factory: replace the ImageNet
    stem with a 3x3/stride-1/pad-1 conv and an identity maxpool, then
    Kaiming-init Conv2d weights and constant-init BN/GroupNorm scale/bias.
    (Init values are cosmetic for trail — every loaded checkpoint replaces
    them via a strict state-dict load — but the stem shape is load-bearing.)
    """
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return model


def _make_resnet18(num_classes: int) -> "nn.Module":
    """CIFAR-adapted torchvision resnet18 (3x3 stem, no maxpool)."""
    from torchvision.models import resnet18  # lazy: keep adapter import light

    return _apply_cifar_tweaks(resnet18(weights=None, num_classes=num_classes))


def _make_resnet34(num_classes: int) -> "nn.Module":
    """CIFAR-adapted torchvision resnet34 (3x3 stem, no maxpool)."""
    from torchvision.models import resnet34  # lazy: keep adapter import light

    return _apply_cifar_tweaks(resnet34(weights=None, num_classes=num_classes))


# NOTE: a module-global model-defaults mechanism is deliberately avoided —
# adapter construction carries arch/num_classes explicitly, so there is no
# hidden module state to get out of sync.
def _make_resnet50(num_classes: int) -> "nn.Module":
    """CIFAR-adapted torchvision resnet50 (F3; reuses the CIFAR stem tweaks).
    Inference-only at eval scale via the probe's OOM batch-halving."""
    from torchvision.models import resnet50  # lazy

    return _apply_cifar_tweaks(resnet50(weights=None, num_classes=num_classes))


def _make_swin_t(num_classes: int) -> "nn.Module":
    """torchvision Swin-T with its NATIVE ImageNet stem — NO CIFAR tweaks.
    CIFAR-stem vs native-stem are distinct arch names, so ``factory(num_classes)``
    stays 1-arg. Inference-only."""
    from torchvision.models import swin_t  # lazy

    return swin_t(weights=None, num_classes=num_classes)


def _make_convnext_t(num_classes: int) -> "nn.Module":
    """torchvision ConvNeXt-T, native stem, no CIFAR tweaks. Inference-only."""
    from torchvision.models import convnext_tiny  # lazy

    return convnext_tiny(weights=None, num_classes=num_classes)


def _make_vit_b_16(num_classes: int) -> "nn.Module":
    """torchvision ViT-B/16, native stem, no CIFAR tweaks. Inference-only.
    Penultimate feature is the CLS token (see _vit_feature_resolver)."""
    from torchvision.models import vit_b_16  # lazy

    return vit_b_16(weights=None, num_classes=num_classes)


class _AllCNN(nn.Module):
    """All-convolutional 32x32 net (Springenberg et al. 2015, AllCNN-C), with a
    GAP + Linear head so the penultimate feature (``avgpool`` output, [N, 2c]) and
    the classifier (``fc``) are cleanly addressable by the feature resolver and
    the grad scaffold (Phase 5 bespoke arch)."""

    def __init__(self, num_classes: int = 10, c: int = 96) -> None:
        super().__init__()

        def block(i, o, k=3, s=1, p=1):
            return nn.Sequential(nn.Conv2d(i, o, k, stride=s, padding=p),
                                 nn.ReLU(inplace=True))

        self.features = nn.Sequential(
            block(3, c), block(c, c), block(c, c, s=2),
            block(c, 2 * c), block(2 * c, 2 * c), block(2 * c, 2 * c, s=2),
            block(2 * c, 2 * c, p=0), block(2 * c, 2 * c, k=1, p=0),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2 * c, num_classes)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.features(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


def _make_allcnn(num_classes: int) -> "nn.Module":
    """Bespoke AllCNN-C 32x32 factory (Phase 5). CIFAR-scale; inference-only."""
    return _AllCNN(num_classes=num_classes)


register_arch("resnet18", _make_resnet18)
register_arch("resnet34", _make_resnet34)
register_arch("resnet50", _make_resnet50)        # CIFAR stem
register_arch("swin_t", _make_swin_t)            # native ImageNet stem
register_arch("convnext_t", _make_convnext_t)    # native ImageNet stem
register_arch("vit_b_16", _make_vit_b_16)        # native ImageNet stem
register_arch("allcnn", _make_allcnn)            # bespoke 32x32 (Phase 5)
ARCH_FEATURE_RESOLVERS["resnet18"] = _avgpool_feature_resolver
ARCH_FEATURE_RESOLVERS["resnet34"] = _avgpool_feature_resolver
ARCH_FEATURE_RESOLVERS["resnet50"] = _avgpool_feature_resolver
ARCH_FEATURE_RESOLVERS["swin_t"] = _avgpool_feature_resolver
ARCH_FEATURE_RESOLVERS["convnext_t"] = _avgpool_feature_resolver
ARCH_FEATURE_RESOLVERS["vit_b_16"] = _vit_feature_resolver
ARCH_FEATURE_RESOLVERS["allcnn"] = _avgpool_feature_resolver


# ---------------------------------------------------------------------------
# Checkpoint unwrapping
# ---------------------------------------------------------------------------

def unwrap_state_dict(raw: Any) -> tuple[dict[str, "torch.Tensor"], str]:
    """Normalize the checkpoint format zoo into one flat state dict.

    Handles, composably:
      (a) a raw flat ``{param_name: tensor}`` dict;
      (b) training-state nesting under ``"model_state"`` or ``"state_dict"``;
      (c) ``torch.nn.utils.prune``-style ``*_orig`` / ``*_mask`` pairs,
          multiplied out into the effective dense parameter;
      plus stripping of a uniform ``"module."`` DataParallel prefix and
      legacy full-module pickles (``state_dict()`` extracted).

    Args:
        raw: the object returned by ``torch.load``.

    Returns:
        ``(flat_state_dict, fmt)`` where ``fmt`` is the manifest string
        describing every unwrapping step applied (``ckpt_format`` row of the
        preprocessing manifest).

    Raises:
        ValueError: if ``raw`` is not a recognizable checkpoint payload.
    """
    fmt_parts: list[str] = []

    if isinstance(raw, nn.Module):  # legacy whole-module pickle
        raw = raw.state_dict()
        fmt_parts.append("module_pickle")

    if not isinstance(raw, dict):
        raise ValueError(
            f"unrecognized checkpoint payload of type {type(raw).__name__}; "
            "expected a state dict, a nested training state, or an nn.Module")

    sd: dict[str, Any] = dict(raw)

    # (b) unnest training states (loop: some trainers nest twice).
    nested = True
    while nested:
        nested = False
        for key in ("model_state", "state_dict"):
            if key in sd and isinstance(sd[key], dict):
                sd = dict(sd[key])
                fmt_parts.append(f"nested:{key}")
                nested = True
                break
    if not fmt_parts:
        fmt_parts.append("flat")
    elif not any(p.startswith(("nested:", "module_pickle")) for p in fmt_parts):
        fmt_parts.insert(0, "flat")

    if not sd:
        raise ValueError("checkpoint unwrapped to an empty state dict")

    # DataParallel/DDP prefix strip (only when uniform across all keys).
    if all(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}
        fmt_parts.append("module_stripped")

    # (c) multiply out prune-style *_orig / *_mask pairs.
    masks_applied = False
    for mask_key in [k for k in list(sd) if k.endswith("_mask")]:
        stem = mask_key[: -len("_mask")]
        orig_key = stem + "_orig"
        if orig_key in sd:
            sd[stem] = sd[orig_key] * sd[mask_key]
            del sd[orig_key]
            del sd[mask_key]
            masks_applied = True
    if masks_applied:
        fmt_parts.append("pruned_masks_applied")

    return sd, "+".join(fmt_parts)


# ---------------------------------------------------------------------------
# Outputs-in helpers
# ---------------------------------------------------------------------------

def _to_numpy(x: Any) -> np.ndarray:
    """Coerce payload values (tensor / list / ndarray) to a numpy array."""
    if hasattr(x, "detach"):  # torch tensor without importing torch types here
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _ce_from_logits(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Numerically stable per-example cross-entropy from logits (float64)."""
    z = logits.astype(np.float64)
    m = z.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
    return lse - z[np.arange(len(z)), targets]


def _extract_normalization(transform: Any) -> list | None:
    """Pull ``[mean, std]`` out of a (possibly composed) torchvision transform."""
    try:
        from torchvision import transforms as T  # lazy
    except ImportError:  # pragma: no cover
        return None
    stack = [transform]
    while stack:
        t = stack.pop()
        if t is None:
            continue
        if isinstance(t, T.Normalize):
            return [list(map(float, t.mean)), list(map(float, t.std))]
        inner = getattr(t, "transforms", None)
        if inner is not None:
            stack.extend(inner)
    return None


def _find_penultimate_pool(model: "nn.Module") -> "nn.Module | None":
    """Locate the penultimate pooling layer for feature capture.

    Prefers a module named ``avgpool`` (torchvision ResNet convention), falls
    back to the LAST ``AdaptiveAvgPool2d`` in module order; returns None when
    neither exists (features are then omitted from SplitOutputs).
    """
    for name, mod in model.named_modules():
        if name == "avgpool" or name.endswith(".avgpool"):
            return mod
    pool = None
    for mod in model.modules():
        if isinstance(mod, nn.AdaptiveAvgPool2d):
            pool = mod
    return pool


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

class ClassificationAdapter(ModalityAdapter):
    """v0.1 image-classification adapter (CIFAR-10/ResNet-18 MVP)."""

    name = "classification"

    def __init__(self, arch: str = "resnet18", num_classes: int = 10,
                 dataset: str = "cifar10") -> None:
        """Args:
            arch: key into ARCH_REGISTRY (validated at load time so users may
                register custom archs after adapter construction).
            num_classes: classifier head width.
            dataset: dataset-registry key used to resolve the canonical test
                transform (normalization constants live in the data layer,
                never here).
        """
        self.arch = arch
        self.num_classes = num_classes
        self.dataset = dataset

    @classmethod
    def from_request(cls, request: Any) -> "ClassificationAdapter":
        """Build a classification adapter from a request (F2 runner entry point).

        Resolution (each independently overridable via ``request.runtime``):
          * ``arch``        — ``runtime.arch`` else ``"resnet18"`` (the MVP default).
          * ``dataset``     — ``runtime.dataset`` else the data's ``name`` (a
            ``DatasetSpec``) else ``"cifar10"``.
          * ``num_classes`` — ``runtime.num_classes`` else
            ``num_classes_for(dataset)``. An unregistered ``dataset`` with no
            explicit ``num_classes`` raises ``RequestError`` (via
            ``num_classes_for``) rather than silently guessing — the head width
            must be derivable, not assumed.
        For the default CIFAR-10 request this yields ``(resnet18, 10, cifar10)``
        — identical to the old zero-arg ``ClassificationAdapter()`` the runner
        constructed, so the reproduction path is byte-unchanged.
        """
        from trail.data.datasets import num_classes_for

        rt = getattr(request, "runtime", None)
        arch = getattr(rt, "arch", None) or "resnet18"
        dataset = (getattr(rt, "dataset", None)
                   or getattr(getattr(request, "data", None), "name", None)
                   or "cifar10")
        num_classes = getattr(rt, "num_classes", None)
        if num_classes is None:
            num_classes = num_classes_for(dataset)  # raises on an unknown dataset
        return cls(arch=arch, num_classes=num_classes, dataset=dataset)

    # -- checkpoints --------------------------------------------------------

    def load_checkpoint(self, path: str, device: "torch.device") -> "nn.Module":
        """Load + strictly validate a classification checkpoint.

        Unwraps the format zoo (see ``unwrap_state_dict``), builds the
        configured architecture, and performs a strict state-dict load.
        The applied unwrapping recipe is attached as
        ``model._trail_ckpt_format`` for the preprocessing manifest
        (``ckpt_format``). The caller hashes the file.

        Raises:
            CheckpointError: with role ``"unknown"`` (the adapter does not
                know which role this path serves; the runner re-attributes).
        """
        try:
            raw = torch.load(path, map_location="cpu", weights_only=True)
        except FileNotFoundError as e:
            raise CheckpointError("unknown", path, f"file not found: {e}") from e
        except Exception as e:  # legacy pickle fallback
            logger.warning(
                "weights_only=True load failed for %s (%s); retrying with "
                "weights_only=False for legacy pickle", path, e)
            try:
                raw = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as e2:
                raise CheckpointError(
                    "unknown", path, f"torch.load failed: {e2}") from e2

        try:
            flat_sd, fmt = unwrap_state_dict(raw)
        except ValueError as e:
            raise CheckpointError("unknown", path, str(e)) from e

        if self.arch not in ARCH_REGISTRY:
            raise CheckpointError(
                "unknown", path,
                f"unknown arch {self.arch!r}; registered: {sorted(ARCH_REGISTRY)}")
        model = ARCH_REGISTRY[self.arch](self.num_classes)
        try:
            model.load_state_dict(flat_sd, strict=True)
        except RuntimeError as e:
            raise CheckpointError(
                "unknown", path,
                f"strict load_state_dict failed for arch {self.arch!r}: {e}") from e

        model._trail_ckpt_format = fmt  # manifest: ckpt_format
        return model.to(device).eval()

    def build_fresh_model(self, device: "torch.device") -> "nn.Module":
        """A fresh, untrained model of this adapter's architecture.

        Used by the shadow-ensemble builder (references/shadow.py) for the
        opt-in LiRA tier (M9): shadow models are reference artifacts, not
        unlearning methods, so building them here is consistent with the
        scope rule (no methods/, but metric-internal reference models are
        allowed). Architecture init RNG
        is the global torch state; callers seed it via a named substream
        before each shadow so the ensemble is reproducible (G2).
        """
        if self.arch not in ARCH_REGISTRY:
            raise CheckpointError(
                "unknown", "<fresh>",
                f"unknown arch {self.arch!r}; registered: {sorted(ARCH_REGISTRY)}")
        return ARCH_REGISTRY[self.arch](self.num_classes).to(device)

    # -- the leaf probe -----------------------------------------------------

    def forward_stats(self, model: "nn.Module", loader: "DataLoader", *,
                      seed: int, device: "torch.device") -> SplitOutputs:
        """ONE deterministic fp32 pass producing SplitOutputs.

        Semantics ported from the original training pipeline
        (``model.eval()``, loud empty-loader guard), extended with:
        BatchNorm-in-train detection forces eval mode and records the fact on
        the model as ``_trail_bn_forced_eval`` (the runner copies it into
        the ``bn_forced_eval`` manifest field); per-sample CE losses; the
        penultimate-pool features are captured via a forward hook; a single
        CUDA-OOM retry re-runs the pass at half batch size.

        ``seed`` parameterizes only the retry loader's generator — the
        classification probe itself is deterministic by construction.

        Raises:
            RequestError: on an empty split (never a silent 0-row output).
        """
        ds = getattr(loader, "dataset", None)
        try:
            if ds is not None and len(ds) == 0:
                raise RequestError("empty split: probe loader has 0 examples")
        except TypeError:  # iterable dataset without __len__
            pass

        # BN-in-train detection BEFORE forcing eval (batch-size invariance).
        bn_forced = any(isinstance(m, _BatchNorm) and m.training
                        for m in model.modules())
        if bn_forced:
            logger.warning(
                "forward_stats: BatchNorm modules were in train mode; forcing "
                "eval() for batch-size invariance (manifest: bn_forced_eval)")
        model.eval()
        # Recorded on the model: the probe has no manifest handle, so the
        # runner reads this attribute to fill the bn_forced_eval field.
        model._trail_bn_forced_eval = bn_forced

        layers = self._resolve_feature_layers(model)
        if not layers:
            logger.warning(
                "forward_stats: arch %r exposes no feature layer (no resolver "
                "and no 'avgpool'/AdaptiveAvgPool2d); features omitted from "
                "SplitOutputs", self.arch)

        try:
            losses, targets, logits, features_multi = self._probe_pass(
                model, loader, device, layers)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            half = max(1, (getattr(loader, "batch_size", None)
                           or _DEFAULT_BATCH_SIZE) // 2)
            logger.warning(
                "forward_stats: CUDA OOM; retrying once with halved "
                "batch_size=%d (recorded as _trail_oom_batch_halved)", half)
            model._trail_oom_batch_halved = True
            retry_loader = DataLoader(
                loader.dataset, batch_size=half, shuffle=False, num_workers=0,
                drop_last=False,
                generator=torch_generator(seed, "probe_oom_retry"))
            losses, targets, logits, features_multi = self._probe_pass(
                model, retry_loader, device, layers)

        n = int(len(losses))
        if n == 0:
            raise RequestError("empty split: probe produced 0 examples")
        # ``features`` mirrors the penultimate layer for activation_distance
        # back-compat; ``features_multi`` carries every captured layer.
        features = (features_multi.get(_PENULTIMATE_KEY)
                    if features_multi else None)
        return SplitOutputs(losses=losses, targets=targets, logits=logits,
                            features=features, features_multi=features_multi,
                            n=n)

    def _resolve_feature_layers(self, model: "nn.Module") -> dict[str, "_FeatureHook"]:
        """Capture points to hook for feature export (F4/F3). The arch's
        registered ARCH_FEATURE_RESOLVERS entry wins; otherwise fall back to the
        penultimate AdaptiveAvgPool2d (pre-F4 behavior). Resolver values may be a
        bare ``nn.Module`` (output captured) or a ``_FeatureHook`` (e.g. ViT's
        pre-hook); both are normalized here. Empty dict = no feature layer."""
        resolver = ARCH_FEATURE_RESOLVERS.get(self.arch)
        if resolver is not None:
            raw = resolver(model)
        else:
            pool = _find_penultimate_pool(model)
            raw = {_PENULTIMATE_KEY: pool} if pool is not None else {}
        out: dict[str, _FeatureHook] = {}
        for name, spec in raw.items():
            if spec is None:
                continue
            hook = spec if isinstance(spec, _FeatureHook) else _FeatureHook(module=spec)
            if hook.module is not None:
                out[name] = hook
        return out

    def probe_layer_names(self) -> tuple[str, ...]:
        """Feature-layer names this adapter's probe intends to capture.
        Folded into the L1 cache key via ``probe_cfg`` so a different layer set
        cannot serve a stale entry. v0.1 captures the single penultimate."""
        return (_PENULTIMATE_KEY,)

    def probe_cfg(self) -> dict:
        """Probe-configuration descriptor folded into the L1 key (F4/G7)."""
        return {"probe_layers": list(self.probe_layer_names())}

    def _probe_pass(self, model: "nn.Module", loader: "DataLoader",
                    device: "torch.device",
                    layers: dict[str, "_FeatureHook"],
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                               "dict[str, np.ndarray] | None"]:
        """Single inference-mode sweep accumulating per-example arrays.

        Multi-hook (F4/F3): registers one forward (or forward-pre) hook per
        capture point in ``layers`` and accumulates each layer's per-example
        feature. The loss/logit/target accumulation is byte-identical to the
        pre-F4 single-hook probe."""
        losses_l: list[np.ndarray] = []
        targets_l: list[np.ndarray] = []
        logits_l: list[np.ndarray] = []
        names = list(layers)
        feats_ll: dict[str, list[np.ndarray]] = {name: [] for name in names}
        caps: dict[str, list[torch.Tensor]] = {name: [] for name in names}

        handles: list[Any] = []
        for name, hook in layers.items():
            if hook.pre:
                def _pre(_m, args, _nm=name):
                    caps[_nm].append(args[0])  # module INPUT (e.g. ViT CLS token)
                handles.append(hook.module.register_forward_pre_hook(_pre))
            else:
                def _post(_m, _i, out, _nm=name):
                    caps[_nm].append(out)       # module OUTPUT (e.g. avgpool)
                handles.append(hook.module.register_forward_hook(_post))
        try:
            with torch.inference_mode():
                for inputs, targets in loader:
                    for c in caps.values():
                        c.clear()
                    inputs = inputs.to(device, non_blocking=True).float()
                    targets = targets.to(device, non_blocking=True)
                    logits = model(inputs)
                    loss = F.cross_entropy(logits, targets, reduction="none")
                    losses_l.append(
                        loss.detach().float().cpu().numpy().astype(np.float32))
                    targets_l.append(
                        targets.detach().cpu().numpy().astype(np.int64))
                    logits_l.append(
                        logits.detach().float().cpu().numpy().astype(np.float32))
                    for name in names:
                        if caps[name]:
                            feats_ll[name].append(
                                caps[name][-1].detach().flatten(1).float()
                                .cpu().numpy().astype(np.float32))
        finally:
            for handle in handles:
                handle.remove()

        if not losses_l:
            raise RequestError("empty split: probe consumed 0 batches")
        # Drop-all-or-nothing per layer: keep only layers captured on EVERY
        # batch (a partially-hooked layer is dropped, never emitted ragged).
        n_batches = len(losses_l)
        features_multi: dict[str, np.ndarray] = {
            name: np.concatenate(feats_ll[name])
            for name in names
            if feats_ll[name] and len(feats_ll[name]) == n_batches}
        return (np.concatenate(losses_l), np.concatenate(targets_l),
                np.concatenate(logits_l), features_multi or None)

    # -- canonical evaluation view -------------------------------------------

    def canonical_eval_view(self, loader: "DataLoader", *, seed: int,
                            num_workers: int = 0,
                            ) -> tuple["DataLoader", dict]:
        """Augmentation-stripped, deterministically ordered view of a loader.

        Constructive re-port of the original evaluation pipeline: the
        legacy ``_augmentation_disabled`` context manager swapped transforms
        on the SHARED underlying dataset in place and restored them on exit;
        here we never mutate user state. We descend the Subset chain
        composing indices (the original evaluation pipeline ``_unwrap_dataset``,
        extended with index composition), shallow-copy the base dataset with
        the registry test transform (ToTensor + dataset Normalize — NOT a
        bare ToTensor; the original evaluation pipelinerationale), preserve the
        ``train`` flag, and rewrap a fresh ``Subset``.

        Returns the new sequential DataLoader and its preprocessing manifest
        entry. ``bn_forced_eval`` is initialized False and filled
        by the probe via ``model._trail_bn_forced_eval``.
        """
        # Lazy imports: the data layer is a sibling subpackage; importing at
        # call time avoids import-order coupling between subpackages.
        from trail.data import fingerprint
        from trail.data.specs import get_test_transform

        # Walk Subset wrappers, composing indices outermost-first:
        # outer[j] == base[inner_idx[outer_idx[j]]].
        base: Any = loader.dataset
        composed: list[int] | None = None
        while isinstance(base, Subset) or (
                hasattr(base, "dataset") and hasattr(base, "indices")):
            layer = list(base.indices)
            composed = layer if composed is None else [layer[i] for i in composed]
            base = base.dataset

        transform = None
        if hasattr(base, "transform"):
            transform = get_test_transform(self.dataset)
            view = copy.copy(base)        # fresh view; user dataset untouched
            view.transform = transform    # train flag preserved by the copy
            ds: Dataset = (Subset(view, composed)
                           if composed is not None else view)
            aug_stripped = True
        else:
            logger.warning(
                "canonical_eval_view: dataset %r exposes no .transform "
                "attribute; cannot strip augmentation constructively — using "
                "the user view as-is (manifest: aug_stripped=False)",
                type(base).__name__)
            ds = loader.dataset
            aug_stripped = False

        canonical = DataLoader(
            ds,
            batch_size=getattr(loader, "batch_size", None) or _DEFAULT_BATCH_SIZE,
            shuffle=False,                # manifest: loader_order=sequential
            num_workers=num_workers,
            drop_last=False,              # no example dropping
            worker_init_fn=make_worker_init_fn(seed, "canonical_loader"),
            generator=torch_generator(seed, "canonical_loader"),
        )
        manifest = {
            "aug_stripped": aug_stripped,
            "normalization": _extract_normalization(transform),
            "loader_order": "sequential",
            "transform_sha": (fingerprint.transform_sha(transform)
                              if transform is not None else None),
            "bn_forced_eval": False,  # filled by the probe (forward_stats)
            "dtype": "fp32",
        }
        return canonical, manifest

    # -- outputs-in ----------------------------------------------------------

    def validate_outputs_payload(self, payload: dict) -> SplitOutputs:
        """Validate an outputs-in payload for the classification task.

        Required: ``logits`` [N, C] float, ``targets`` [N] int.
        Optional: ``losses`` [N] (derived from logits via CE when absent;
        when both present, mean |supplied - derived| must be < 1e-3 over the
        jointly finite entries — non-finite losses are NOT rejected here, per
        they skip at the consuming metric), ``features`` [N, D].

        Raises:
            RequestError: on missing keys, shape/dtype/semantic violations,
                or an empty (N == 0) payload.
        """
        if not isinstance(payload, dict):
            raise RequestError(
                f"outputs payload must be a dict, got {type(payload).__name__}")
        for key in ("logits", "targets"):
            if payload.get(key) is None:
                raise RequestError(
                    "classification outputs payload requires 'logits' [N, C] "
                    f"and 'targets' [N]; missing {key!r}")

        logits = _to_numpy(payload["logits"])
        targets = _to_numpy(payload["targets"])
        if logits.ndim != 2:
            raise RequestError(f"'logits' must be [N, C]; got shape {logits.shape}")
        if targets.ndim != 1:
            raise RequestError(f"'targets' must be [N]; got shape {targets.shape}")
        n, c = logits.shape
        if n == 0:
            raise RequestError("empty split: outputs payload has 0 examples")
        if len(targets) != n:
            raise RequestError(
                f"'targets' length {len(targets)} != logits rows {n}")
        if not np.issubdtype(targets.dtype, np.integer):
            if not np.all(np.isfinite(targets)) or \
                    not np.all(targets == np.round(targets)):
                raise RequestError("'targets' must be integer class ids")
        targets = targets.astype(np.int64)
        if targets.min() < 0 or targets.max() >= c:
            raise RequestError(
                f"'targets' out of range [0, {c}): "
                f"min={int(targets.min())}, max={int(targets.max())}")
        logits = logits.astype(np.float32)

        derived = _ce_from_logits(logits, targets)
        if payload.get("losses") is not None:
            losses = _to_numpy(payload["losses"]).astype(np.float32)
            if losses.shape != (n,):
                raise RequestError(
                    f"'losses' must be [N]={n}; got shape {losses.shape}")
            both = np.isfinite(losses) & np.isfinite(derived)
            if both.any():
                diff = float(np.mean(np.abs(
                    losses[both].astype(np.float64) - derived[both])))
                if not diff < _LOSS_XCHECK_TOL:
                    raise RequestError(
                        "'losses' inconsistent with CE(logits, targets): "
                        f"mean |delta| = {diff:.3e} >= {_LOSS_XCHECK_TOL}")
        else:
            losses = derived.astype(np.float32)

        features = None
        if payload.get("features") is not None:
            features = _to_numpy(payload["features"]).astype(np.float32)
            if features.ndim != 2 or len(features) != n:
                raise RequestError(
                    f"'features' must be [N, D] with N={n}; "
                    f"got shape {features.shape}")

        return SplitOutputs(losses=losses, targets=targets, logits=logits,
                            features=features, n=n)

    # -- capability hooks ----------------------------------------------------

    def derived_test_splits(self, test_loader: "DataLoader",
                            mode: str) -> dict[str, "DataLoader"] | None:
        """Return None: derived test partitions are owned by the data layer.

        Forget-class/retain-class test partitions for class-forgetting modes
        are produced by ``trail.data.specs.resolve_dataset_spec``, which
        owns split identity, the forget-class id, and fingerprinting —
        deriving them here too would create two sources of truth. Returning
        None triggers the documented fallback for raw user loaders the data
        layer cannot partition: full-test accuracy with a warning, never
        silently.
        """
        return None

    def default_metrics(self) -> list[str]:
        """The v0.1 classification default panel (resolved vs METRIC_REGISTRY).

        ``mia_lira`` (M9) is deliberately NOT in the default panel: it is the
        opt-in rigorous tier and training the shadow ensemble is expensive
        (``hp.references.shadow`` models). Opt in by adding ``"mia_lira"`` to
        ``metrics`` AND setting ``hp.references.shadow > 0`` — otherwise default
        runs never pay the shadow-training cost (and a default run that DID set
        ``shadow>0`` for some other reason would not silently trigger it).
        """
        return [
            "fa_train", "fa_test", "ra_train", "ra_test", "ua", "ta",
            "forget_gap_to_gold", "sum_delta_to_gold",
            "mia_threshold_population", "mia_loss_logreg",
            "relearn_forget", "relearn_d2d",
            "collapse_resistance", "activation_distance", "fde",
            "wall_clock",
        ]
