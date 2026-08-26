"""OLS uncertainty calculations without enforcing positive definiteness."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .quadratic_estimator import quadratic_design_matrix

@dataclass(frozen=True)
class QuadraticUncertainty:
    residual_df: int
    sse: float
    sigma2: float | None
    coefficient_covariance: np.ndarray | None
    magnitude_se: float | None
    relative_magnitude_se: float | None
    source: str


def _magnitude_gradient(hessian: np.ndarray) -> np.ndarray:
    # coefficient order H11,H12,H22; q=||H||F/sqrt(2)
    H = np.asarray(hessian, float)
    norm = np.linalg.norm(H, "fro")
    if norm == 0:
        return np.full(3, np.nan)
    return np.array([H[0,0], 2*H[0,1], H[1,1]]) / (np.sqrt(2.0) * norm)


def estimate_uncertainty(points, values, estimate, *, known_sigma: float | None = None) -> QuadraticUncertainty:
    X = quadratic_design_matrix(points)
    if not estimate.identifiable or estimate.hessian is None:
        return QuadraticUncertainty(len(values)-6, float("nan"), None, None, None, None, "unavailable")
    coef = np.r_[estimate.intercept, estimate.gradient, estimate.hessian[0,0], estimate.hessian[0,1], estimate.hessian[1,1]]
    residual = np.asarray(values) - X @ coef
    sse = float(residual @ residual)
    df = len(values) - 6
    if known_sigma is not None:
        sigma2, source = float(known_sigma)**2, "oracle"
    elif df > 0:
        sigma2, source = sse / df, "residual"
    else:
        return QuadraticUncertainty(df, sse, None, None, None, None, "residual_unavailable")
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    hcov = cov[np.ix_([3,4,5],[3,4,5])]
    grad = _magnitude_gradient(estimate.hessian)
    variance = float(grad @ hcov @ grad)
    se = float(np.sqrt(max(0.0, variance)))
    magnitude = float(np.linalg.norm(estimate.hessian, "fro") / np.sqrt(2.0))
    return QuadraticUncertainty(df, sse, sigma2, cov, se, se / magnitude if magnitude > 0 else float("inf"), source)
