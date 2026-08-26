"""Representation-layer normalization utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def recover_canonical_curvature(
    physical_hessian: ArrayLike, basis: ArrayLike, output_scale: float
) -> FloatArray:
    """Recover K=(1/s) B^T H_x B from externally supplied quantities."""

    hessian = np.asarray(physical_hessian, dtype=float)
    chart = np.asarray(basis, dtype=float)
    scale = float(output_scale)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("physical_hessian must be square")
    if chart.shape != hessian.shape:
        raise ValueError("basis and physical_hessian must have the same square shape")
    if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(chart)):
        raise ValueError("physical_hessian and basis must be finite")
    if not np.allclose(hessian, hessian.T, rtol=0.0, atol=1e-10):
        raise ValueError("physical_hessian must be symmetric")
    if np.linalg.matrix_rank(chart) != chart.shape[0]:
        raise ValueError("basis must be invertible")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("output_scale must be finite and strictly positive")
    recovered = chart.T @ hessian @ chart / scale
    return 0.5 * (recovered + recovered.T)


def finite_difference_hessian(function, point: ArrayLike, step: float = 1e-4) -> FloatArray:
    """Estimate a Hessian from function values using central differences."""

    x = np.asarray(point, dtype=float)
    h = float(step)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("point must be a finite vector")
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("step must be finite and strictly positive")
    dimension = x.size
    result = np.zeros((dimension, dimension), dtype=float)
    f0 = float(function(x))
    for i in range(dimension):
        ei = np.zeros(dimension)
        ei[i] = h
        result[i, i] = (float(function(x + ei)) - 2.0 * f0 + float(function(x - ei))) / h**2
        for j in range(i + 1, dimension):
            ej = np.zeros(dimension)
            ej[j] = h
            mixed = (
                float(function(x + ei + ej))
                - float(function(x + ei - ej))
                - float(function(x - ei + ej))
                + float(function(x - ei - ej))
            ) / (4.0 * h**2)
            result[i, j] = mixed
            result[j, i] = mixed
    return result
