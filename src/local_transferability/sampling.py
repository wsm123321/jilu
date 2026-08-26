"""Frozen two-dimensional sampling designs for the identifiability pilot."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc


FloatArray = NDArray[np.float64]


def sample_sobol(n: int, seed: int) -> FloatArray:
    if n <= 0:
        raise ValueError("n must be positive")
    engine = qmc.Sobol(d=2, scramble=True, seed=seed)
    exponent = int(np.ceil(np.log2(n)))
    # Non-power-of-two sample sizes use a prefix of a generated 2^m net.
    return 2.0 * engine.random_base2(exponent)[:n] - 1.0


def sample_uniform(n: int, seed: int) -> FloatArray:
    if n <= 0:
        raise ValueError("n must be positive")
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n, 2))


def sample_trajectory(n: int, seed: int, curvature: NDArray[np.float64]) -> FloatArray:
    """Generate a biased shrinking trajectory without implementing BO."""

    if n <= 0:
        raise ValueError("n must be positive")
    matrix = np.asarray(curvature, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError("curvature must have shape (2, 2)")
    rng = np.random.default_rng(seed)
    points: list[FloatArray] = []
    values: list[float] = []
    for index in range(n):
        if index < 2:
            candidate = rng.uniform(-1.0, 1.0, size=2)
        else:
            incumbent = points[int(np.argmin(values))]
            radius = max(0.08, 0.85 * (0.78 ** (index - 2)))
            candidate = np.clip(incumbent + rng.normal(0.0, radius, size=2), -1.0, 1.0)
        points.append(candidate)
        values.append(float(0.5 * candidate @ matrix @ candidate))
    return np.asarray(points)


def sample_observed_trajectory(
    max_n: int,
    seed: int,
    curvature: NDArray[np.float64],
    gradient: NDArray[np.float64],
    observation_noise: NDArray[np.float64],
) -> FloatArray:
    """Trajectory driven by observed values with shared latent innovations."""
    if max_n <= 0:
        raise ValueError("max_n must be positive")
    noise = np.asarray(observation_noise, dtype=float)
    if noise.shape[0] < max_n:
        raise ValueError("observation_noise is shorter than max_n")
    rng = np.random.default_rng(seed)
    initial = rng.uniform(-1.0, 1.0, size=(2, 2))
    innovations = rng.normal(size=(max(0, max_n - 2), 2))
    points: list[FloatArray] = []
    observed: list[float] = []
    for index in range(max_n):
        if index < 2:
            candidate = initial[index]
        else:
            incumbent = points[int(np.argmin(observed))]
            radius = max(0.08, 0.85 * (0.78 ** (index - 2)))
            candidate = np.clip(incumbent + radius * innovations[index - 2], -1.0, 1.0)
        value = float(
            gradient @ candidate
            + 0.5 * candidate @ curvature @ candidate
            + noise[index]
        )
        points.append(candidate)
        observed.append(value)
    return np.asarray(points)


def sample_design(name: str, n: int, seed: int, curvature: NDArray[np.float64]) -> FloatArray:
    if name == "sobol":
        return sample_sobol(n, seed)
    if name == "uniform":
        return sample_uniform(n, seed)
    if name == "trajectory":
        return sample_trajectory(n, seed, curvature)
    raise ValueError(f"unknown sampling design: {name}")
