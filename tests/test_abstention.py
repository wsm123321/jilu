import numpy as np
import pytest

from local_transferability.quadratic_estimator import fit_complete_quadratic


@pytest.mark.parametrize("n", [1, 3, 5])
def test_insufficient_samples_always_abstain(n: int) -> None:
    rng = np.random.default_rng(n)
    points = rng.uniform(-1.0, 1.0, size=(n, 2))
    values = np.sum(points**2, axis=1)
    estimate = fit_complete_quadratic(points, values)
    assert not estimate.identifiable
    assert estimate.abstain_reason == "insufficient_samples"
    assert estimate.hessian is None


def test_rank_deficient_six_point_design_abstains() -> None:
    x = np.linspace(-1.0, 1.0, 6)
    points = np.column_stack([x, x])
    estimate = fit_complete_quadratic(points, np.sum(points**2, axis=1))
    assert not estimate.identifiable
    assert estimate.abstain_reason == "rank_deficient"
    assert estimate.rank < 6
