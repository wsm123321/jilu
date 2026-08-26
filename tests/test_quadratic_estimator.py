import numpy as np
import pytest

from local_transferability.quadratic_estimator import (
    fit_complete_quadratic,
    quadratic_design_matrix,
)


def test_design_matrix_coefficient_convention() -> None:
    point = np.array([[2.0, 3.0]])
    np.testing.assert_allclose(
        quadratic_design_matrix(point),
        [[1.0, 2.0, 3.0, 2.0, 6.0, 4.5]],
    )


@pytest.mark.parametrize("seed", range(20))
def test_noiseless_full_rank_fit_recovers_rotated_hessian(seed: int) -> None:
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, size=(12, 2))
    angle = 0.37
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    truth = rotation @ np.diag([1.0, 7.0]) @ rotation.T
    gradient = np.array([0.4, -0.7])
    values = 1.3 + points @ gradient + 0.5 * np.einsum(
        "ni,ij,nj->n", points, truth, points
    )
    estimate = fit_complete_quadratic(points, values)
    assert estimate.identifiable
    np.testing.assert_allclose(estimate.hessian, truth, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(estimate.gradient, gradient, rtol=1e-10, atol=1e-11)
    assert estimate.intercept == pytest.approx(1.3, rel=1e-10, abs=1e-11)


def test_invalid_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        fit_complete_quadratic(np.ones((6, 3)), np.ones(6))
    with pytest.raises(ValueError, match="one entry"):
        fit_complete_quadratic(np.ones((6, 2)), np.ones(5))
