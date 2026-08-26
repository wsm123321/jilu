"""Metrics for few-shot local-curvature identifiability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class IdentifiabilityMetrics:
    hessian_relative_error: float
    magnitude_relative_error: float
    spectrum_l1_error: float
    signed_spectrum_shape_error: float
    log_condition_error: float
    pairwise_order_accuracy: float
    pairwise_coverage: float
    tie_fraction: float
    estimated_min_eigenvalue: float
    is_spd: bool
    inertia_mismatch: bool
    negative_eigenvalue_fraction: float


def signed_spectrum_shape_error(truth: ArrayLike, estimate: ArrayLike) -> float:
    """Scale-free signed eigenvalue-shape error; detects inertia changes."""
    truth_eigenvalues = np.sort(np.linalg.eigvalsh(np.asarray(truth, dtype=float)))
    estimate_eigenvalues = np.sort(np.linalg.eigvalsh(np.asarray(estimate, dtype=float)))
    truth_norm = float(np.linalg.norm(truth_eigenvalues))
    estimate_norm = float(np.linalg.norm(estimate_eigenvalues))
    if truth_norm <= 0.0 or estimate_norm <= 0.0:
        return float("inf")
    return float(np.linalg.norm(estimate_eigenvalues / estimate_norm - truth_eigenvalues / truth_norm))


def normalized_spectrum(matrix: ArrayLike) -> np.ndarray:
    eigenvalues = np.abs(np.linalg.eigvalsh(np.asarray(matrix, dtype=float)))
    total = float(np.sum(eigenvalues))
    if total <= 0.0:
        return np.full_like(eigenvalues, np.nan)
    return np.sort(eigenvalues / total)[::-1]


def pairwise_order_accuracy(truth: ArrayLike, prediction: ArrayLike, tie_tolerance: float = 1e-10) -> tuple[float, float, float]:
    true_values = np.asarray(truth, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    if true_values.shape != predicted.shape or true_values.ndim != 1:
        raise ValueError("truth and prediction must be equally shaped vectors")
    scale = max(1.0, float(np.ptp(true_values)))
    tolerance = tie_tolerance * scale
    correct = 0
    compared = 0
    ties = 0
    total = 0
    for i in range(len(true_values)):
        for j in range(i + 1, len(true_values)):
            total += 1
            true_diff = true_values[i] - true_values[j]
            pred_diff = predicted[i] - predicted[j]
            if abs(true_diff) <= tolerance or abs(pred_diff) <= tolerance:
                ties += 1
                continue
            compared += 1
            correct += int(np.sign(true_diff) == np.sign(pred_diff))
    accuracy = float(correct / compared) if compared else float("nan")
    coverage = float(compared / total) if total else float("nan")
    return accuracy, coverage, float(ties / total) if total else float("nan")


def compute_identifiability_metrics(
    truth_hessian: ArrayLike,
    estimated_hessian: ArrayLike,
    truth_probe_values: ArrayLike,
    estimated_probe_values: ArrayLike,
) -> IdentifiabilityMetrics:
    truth = np.asarray(truth_hessian, dtype=float)
    estimate = np.asarray(estimated_hessian, dtype=float)
    truth_norm = float(np.linalg.norm(truth, ord="fro"))
    estimated_norm = float(np.linalg.norm(estimate, ord="fro"))
    truth_abs_eigenvalues = np.abs(np.linalg.eigvalsh(truth))
    estimated_abs_eigenvalues = np.abs(np.linalg.eigvalsh(estimate))
    truth_condition = float(np.max(truth_abs_eigenvalues) / np.min(truth_abs_eigenvalues))
    estimated_minimum = float(np.min(estimated_abs_eigenvalues))
    estimated_condition = (
        float(np.max(estimated_abs_eigenvalues) / estimated_minimum)
        if estimated_minimum > 0.0
        else float("inf")
    )
    order_accuracy, pair_coverage, tie_fraction = pairwise_order_accuracy(
        truth_probe_values, estimated_probe_values
    )
    signed_truth = np.linalg.eigvalsh(truth)
    signed_estimate = np.linalg.eigvalsh(estimate)
    spd_tolerance = 1e-12 * max(1.0, estimated_norm)
    is_spd = bool(np.min(signed_estimate) > spd_tolerance)
    truth_inertia = tuple(np.sign(signed_truth).astype(int))
    estimate_inertia = tuple(np.where(np.abs(signed_estimate) <= spd_tolerance, 0, np.sign(signed_estimate)).astype(int))
    return IdentifiabilityMetrics(
        hessian_relative_error=float(np.linalg.norm(estimate - truth, ord="fro") / truth_norm),
        magnitude_relative_error=abs(estimated_norm - truth_norm) / truth_norm,
        spectrum_l1_error=float(np.sum(np.abs(normalized_spectrum(estimate) - normalized_spectrum(truth)))),
        signed_spectrum_shape_error=signed_spectrum_shape_error(truth, estimate),
        log_condition_error=abs(float(np.log(estimated_condition)) - float(np.log(truth_condition))),
        pairwise_order_accuracy=order_accuracy,
        pairwise_coverage=pair_coverage,
        tie_fraction=tie_fraction,
        estimated_min_eigenvalue=float(np.min(signed_estimate)),
        is_spd=is_spd,
        inertia_mismatch=estimate_inertia != truth_inertia,
        negative_eigenvalue_fraction=float(np.mean(signed_estimate < -spd_tolerance)),
    )
