import numpy as np
import pytest

from local_transferability.local_patch import LocalPatch
from local_transferability.normalization import (
    finite_difference_hessian,
    recover_canonical_curvature,
)


@pytest.mark.parametrize("seed", range(30))
def test_external_random_spd_recovery(seed: int) -> None:
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-np.pi, np.pi)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    condition = 10.0 ** rng.uniform(0.0, 3.0)
    curvature = rotation @ np.diag([1.0, condition]) @ rotation.T
    basis = rng.normal(size=(2, 2))
    while abs(np.linalg.det(basis)) < 0.2:
        basis = rng.normal(size=(2, 2))
    scale = float(10.0 ** rng.uniform(-1.0, 1.0))
    inverse = np.linalg.inv(basis)
    external_hessian = scale * inverse.T @ curvature @ inverse
    recovered = recover_canonical_curvature(external_hessian, basis, scale)
    np.testing.assert_allclose(recovered, curvature, rtol=1e-10, atol=1e-10)


def test_function_value_finite_difference_recovery() -> None:
    angle = 0.71
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    curvature = rotation @ np.diag([1.0, 12.0]) @ rotation.T
    patch = LocalPatch(
        curvature=curvature,
        center=np.array([0.2, -0.4]),
        basis=np.array([[1.3, 0.4], [-0.2, 0.8]]),
        output_scale=2.7,
        output_shift=-3.0,
    )
    numerical_hessian = finite_difference_hessian(
        patch.evaluate_physical, patch.center, step=2e-4
    )
    recovered = recover_canonical_curvature(
        numerical_hessian, patch.basis, patch.output_scale
    )
    np.testing.assert_allclose(recovered, curvature, rtol=2e-7, atol=2e-7)
