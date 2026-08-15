"""The single RNG authority (guarantee G2).

Every random draw in trail derives from the request seed through a *named*
substream: ``seed_for(base_seed, name)``. Adding a metric to a panel can never
perturb another metric's randomness, because derivation is pure and name-keyed
(no counters, no global state).

This is the ONLY module in the package allowed to call ``manual_seed`` /
``np.random.seed`` / ``random.seed``.

Note on data-identity seeds: the ported split builders (data/specs.py)
construct ``np.random.RandomState(raw_seed)`` directly from user-supplied
``carveout_seed``/``split_seed`` arguments. That is deliberate byte-exact
reproduction of the fixture-era data recipe, not framework randomness, and is
out of scope for the substream rule (RandomState construction from an explicit
argument mutates no global RNG).
"""
from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np
import torch

_MASK63 = (1 << 63) - 1
_MASK32 = (1 << 32) - 1


def seed_for(base_seed: int, name: str) -> int:
    """Derive a stable 63-bit substream seed from (base_seed, component name)."""
    digest = hashlib.blake2b(f"{base_seed}:{name}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _MASK63


def torch_generator(base_seed: int, name: str, device: str = "cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed_for(base_seed, name))
    return g


def numpy_rng(base_seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng(seed_for(base_seed, name))


def numpy_random_state(base_seed: int, name: str) -> np.random.RandomState:
    """Legacy-port shim for ported code written against RandomState semantics."""
    return np.random.RandomState(seed_for(base_seed, name) & _MASK32)


def make_worker_init_fn(base_seed: int, name: str) -> Callable[[int], None]:
    """Deterministic DataLoader worker seeding."""
    base = seed_for(base_seed, name)

    def _init(worker_id: int) -> None:
        np.random.seed((base + worker_id) & _MASK32)
        torch.manual_seed((base + worker_id) & _MASK63)

    return _init
