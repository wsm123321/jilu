"""Common-random-number utilities for Gate 3."""
from __future__ import annotations
import hashlib
import numpy as np


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def standard_noise(max_n: int, split: str, design: str, seed: int) -> np.ndarray:
    """One nested standard-normal stream shared across landscape conditions."""
    return np.random.default_rng(stable_seed("noise", split, design, seed)).normal(size=max_n)


def scaled_noise(base_noise, eta: float, q: float, n: int) -> np.ndarray:
    return np.asarray(base_noise, dtype=float)[:n] * eta * q
