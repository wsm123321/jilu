"""Orthogonalized quadratic and quartic landscape families for Gate 3."""
from __future__ import annotations
import numpy as np


def rotation(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def orthogonal_curvature(q: float, r: float, theta: float) -> np.ndarray:
    if q <= 0 or r < 1:
        raise ValueError("q must be positive and r must be at least one")
    Q = rotation(theta)
    spectrum = q * np.sqrt(2.0) * np.array([1.0, r]) / np.sqrt(1.0 + r * r)
    return Q @ np.diag(spectrum) @ Q.T


def gradient_state(q: float, nonstationary: bool) -> np.ndarray:
    return q * np.array([0.6, -0.4]) if nonstationary else np.zeros(2)


def evaluate_landscape(points, curvature, gradient=None, intercept=0.0, beta=0.0):
    u = np.asarray(points, dtype=float)
    K = np.asarray(curvature, dtype=float)
    b = np.zeros(2) if gradient is None else np.asarray(gradient, dtype=float)
    return intercept + u @ b + 0.5 * np.einsum("ni,ij,nj->n", u, K, u) + beta * np.sum(u**4, axis=1)
