"""Projection-target utilities for misspecified local quadratics."""
from __future__ import annotations
import numpy as np
from .quadratic_estimator import fit_complete_quadratic


def dense_quadratic_projection(points, values) -> np.ndarray:
    estimate = fit_complete_quadratic(points, values)
    if not estimate.identifiable or estimate.hessian is None:
        raise ValueError("dense projection design is not identifiable")
    return estimate.hessian


def decompose_hessian_error(estimate, projection, center_truth):
    return {
        "finite_sample_to_projection": float(np.linalg.norm(estimate-projection,"fro") / np.linalg.norm(center_truth,"fro")),
        "projection_to_center": float(np.linalg.norm(projection-center_truth,"fro") / np.linalg.norm(center_truth,"fro")),
        "total_to_center": float(np.linalg.norm(estimate-center_truth,"fro") / np.linalg.norm(center_truth,"fro")),
    }
