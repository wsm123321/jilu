import math
import numpy as np
from local_transferability.selective_gating import summarize_gate, freeze_threshold


def test_coverage_risk_and_failure_mass_components():
    result = summarize_gate([True, True, False, False], [True, False, False, True])
    assert result.accepted == 2 and result.total == 4
    assert result.coverage == 0.5
    assert result.selective_risk == 0.5
    assert result.coverage * result.selective_risk == 0.25


def test_zero_acceptance_has_undefined_risk():
    result = summarize_gate([False, False], [True, False])
    assert result.coverage == 0.0 and math.isnan(result.selective_risk)


def test_threshold_freezing_uses_only_supplied_development_arrays():
    values = np.array([0.01, 0.03, 0.2, 0.4])
    reliable = np.array([True, True, False, False])
    threshold, result = freeze_threshold(values, reliable, (0.02, 0.05, 0.5), max_risk=0.10)
    assert threshold == 0.05
    assert result.coverage == 0.5 and result.selective_risk == 0.0


def test_no_safe_threshold_returns_none():
    threshold, result = freeze_threshold([1, 2], [False, False], (1, 2), max_risk=0.10)
    assert threshold is None and result is None
