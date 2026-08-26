"""SVD-based complete quadratic estimator with explicit abstention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class QuadraticEstimate:
    intercept: float | None
    gradient: FloatArray | None
    hessian: FloatArray | None
    rank: int
    singular_values: FloatArray
    condition_number: float
    identifiable: bool
    abstain_reason: str | None


def quadratic_design_matrix(points: ArrayLike) -> FloatArray:
    u = np.asarray(points, dtype=float)
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if not np.all(np.isfinite(u)):
        raise ValueError("points must be finite")
    u1, u2 = u[:, 0], u[:, 1]
    return np.column_stack(
        [np.ones(len(u)), u1, u2, 0.5 * u1**2, u1 * u2, 0.5 * u2**2]
    )


def standardized_design_condition(points: ArrayLike) -> tuple[int, float, FloatArray]:
    """Rank/condition after unit-RMS scaling nonconstant design columns."""
    design = quadratic_design_matrix(points)
    scales = np.sqrt(np.mean(design**2, axis=0))
    scales[0] = 1.0
    if np.any(scales <= 0.0):
        return 0, float("inf"), np.zeros(min(design.shape))
    standardized = design / scales
    singular = np.linalg.svd(standardized, compute_uv=False)
    tolerance = np.finfo(float).eps * max(standardized.shape) * singular[0]
    rank = int(np.sum(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    return rank, condition, singular


def fit_complete_quadratic(points: ArrayLike, values: ArrayLike) -> QuadraticEstimate:
    u = np.asarray(points, dtype=float)
    y = np.asarray(values, dtype=float)
    design = quadratic_design_matrix(u)
    if y.shape != (len(u),) or not np.all(np.isfinite(y)):
        raise ValueError("values must be a finite vector with one entry per point")

    singular_values = np.linalg.svd(design, compute_uv=False)
    tolerance = np.finfo(float).eps * max(design.shape) * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0.0
        else float("inf")
    )
    if len(u) < 6:
        return QuadraticEstimate(None, None, None, rank, singular_values, condition, False, "insufficient_samples")
    if rank < 6:
        return QuadraticEstimate(None, None, None, rank, singular_values, condition, False, "rank_deficient")

    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    hessian = np.array(
        [[coefficients[3], coefficients[4]], [coefficients[4], coefficients[5]]],
        dtype=float,
    )
    return QuadraticEstimate(
        intercept=float(coefficients[0]),
        gradient=coefficients[1:3].copy(),
        hessian=hessian,
        rank=rank,
        singular_values=singular_values,
        condition_number=condition,
        identifiable=True,
        abstain_reason=None,
    )
