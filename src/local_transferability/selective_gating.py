"""Frozen selective-acceptance utilities."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

CONDITION_THRESHOLDS = (10.0,20.0,50.0,100.0,200.0,500.0,1000.0,float("inf"))
UNCERTAINTY_THRESHOLDS = (0.05,0.10,0.20,0.30,0.50,float("inf"))

@dataclass(frozen=True)
class GateResult:
    coverage: float
    selective_risk: float
    accepted: int
    total: int


def reliable(magnitude_error: float, spectrum_error: float, is_spd: bool) -> bool:
    return magnitude_error <= 0.2 and spectrum_error <= 0.15 and is_spd


def summarize_gate(accepted, reliable_flags) -> GateResult:
    accept = np.asarray(accepted, bool); good = np.asarray(reliable_flags, bool)
    count = int(accept.sum()); total = len(accept)
    risk = float((~good[accept]).mean()) if count else float("nan")
    return GateResult(count/total if total else float("nan"), risk, count, total)


def freeze_threshold(values, reliable_flags, candidates, max_risk=0.10):
    """Choose maximum-coverage threshold meeting risk, stricter on ties."""
    vals = np.asarray(values, float); good = np.asarray(reliable_flags, bool)
    feasible = []
    for threshold in candidates:
        result = summarize_gate(np.isfinite(vals) & (vals <= threshold), good)
        if result.accepted and result.selective_risk <= max_risk:
            feasible.append((result.coverage, -threshold, threshold, result))
    if not feasible:
        return None, None
    _, _, threshold, result = max(feasible)
    return threshold, result
