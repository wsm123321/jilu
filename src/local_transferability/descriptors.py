"""Ground-truth descriptors for the first-step local landscape study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .local_patch import LocalPatch


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class CurvatureDescriptor:
    """Coordinate-normalized curvature summary."""

    magnitude: float
    normalized_spectrum: FloatArray
    condition_number: float


def curvature_descriptor(patch: LocalPatch) -> CurvatureDescriptor:
    """Compute descriptors from the recovered canonical curvature matrix."""

    curvature = patch.recovered_canonical_curvature()
    eigenvalues = np.linalg.eigvalsh(curvature)
    absolute = np.abs(eigenvalues)
    total = float(np.sum(absolute))
    minimum = float(np.min(absolute))
    if total <= 0.0 or minimum <= 0.0:
        raise ValueError("curvature spectrum must be nonzero for this descriptor")

    normalized_spectrum = np.sort(absolute / total)[::-1]
    return CurvatureDescriptor(
        magnitude=float(np.linalg.norm(curvature, ord="fro") / np.sqrt(patch.dimension)),
        normalized_spectrum=normalized_spectrum,
        condition_number=float(np.max(absolute) / minimum),
    )


def stable_rank_encoding(values: ArrayLike) -> IntArray:
    """Return a stable best-to-worst index ordering for minimization."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must contain only finite values")
    return np.argsort(array, kind="stable").astype(np.int64)


def rank_descriptor(patch: LocalPatch, canonical_probes: ArrayLike) -> IntArray:
    """Rank canonical probes by their physical-space function values."""

    probes = np.asarray(canonical_probes, dtype=float)
    physical_points = patch.to_physical(probes)
    return stable_rank_encoding(patch.evaluate_physical(physical_points))
