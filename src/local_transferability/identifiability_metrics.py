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
    log_condition_error: float
    pairwise_order_accuracy: float
    tie_fraction: float


def normalized_spectrum(matrix: ArrayLike) -> np.ndarray:
    eigenvalues = np.abs(np.linalg.eigvalsh(np.asarray(matrix, dtype=float)))
    total = float(np.sum(eigenvalues))
    if total <= 0.0:
        return np.full_like(eigenvalues, np.nan)
    return np.sort(eigenvalues / total)[::-1]


def pairwise_order_accuracy(truth: ArrayLike, prediction: ArrayLike, tie_tolerance: float = 1e-10) -> tuple[float, float]:
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
    return accuracy, float(ties / total) if total else float("nan")


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
    order_accuracy, tie_fraction = pairwise_order_accuracy(
        truth_probe_values, estimated_probe_values
    )
    return IdentifiabilityMetrics(
        hessian_relative_error=float(np.linalg.norm(estimate - truth, ord="fro") / truth_norm),
        magnitude_relative_error=abs(estimated_norm - truth_norm) / truth_norm,
        spectrum_l1_error=float(np.sum(np.abs(normalized_spectrum(estimate) - normalized_spectrum(truth)))),
        log_condition_error=abs(float(np.log(estimated_condition)) - float(np.log(truth_condition))),
        pairwise_order_accuracy=order_accuracy,
        tie_fraction=tie_fraction,
    )
