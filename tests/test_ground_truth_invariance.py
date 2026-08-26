from __future__ import annotations

import numpy as np
import pytest

from local_transferability.descriptors import curvature_descriptor, rank_descriptor
from local_transferability.local_patch import LocalPatch


RTOL = 1e-10
ATOL = 1e-12


def rotation(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]])


def canonical_probes() -> np.ndarray:
    # Fixed asymmetric probes avoid systematic ties on symmetric quadratics.
    return np.array(
        [
            [-0.91, -0.37],
            [-0.73, 0.58],
            [-0.44, 0.19],
            [-0.21, -0.82],
            [0.07, 0.31],
            [0.29, -0.54],
            [0.48, 0.77],
            [0.66, -0.13],
            [0.84, 0.42],
            [0.95, -0.69],
        ]
    )


@pytest.fixture(params=[(1.0, 1.0), (1.0, 4.0), (1.0, 16.0)])
def patch(request: pytest.FixtureRequest) -> LocalPatch:
    return LocalPatch(
        curvature=np.diag(request.param),
        center=np.array([0.3, -0.7]),
        basis=rotation(0.31) @ np.diag([1.4, 0.6]),
        output_scale=2.3,
        output_shift=-1.2,
    )


TRANSFORMS = [
    (np.eye(2), np.array([1.7, -0.4]), 1.0, 0.0),
    (2.5 * np.eye(2), np.zeros(2), 1.0, 0.0),
    (np.diag([0.4, 2.2]), np.zeros(2), 1.0, 0.0),
    (rotation(0.83), np.zeros(2), 1.0, 0.0),
    (rotation(-0.61) @ np.diag([1.8, 0.45]), np.array([-0.8, 1.1]), 1.0, 0.0),
    (np.eye(2), np.zeros(2), 3.7, -2.4),
    (rotation(1.17) @ np.diag([0.7, 1.9]), np.array([0.6, -1.3]), 2.1, 0.9),
]


@pytest.mark.parametrize("matrix,shift,scale,offset", TRANSFORMS)
def test_coordinate_round_trip_and_output_law(
    patch: LocalPatch,
    matrix: np.ndarray,
    shift: np.ndarray,
    scale: float,
    offset: float,
) -> None:
    probes = canonical_probes()
    physical = patch.to_physical(probes)
    np.testing.assert_allclose(patch.to_canonical(physical), probes, rtol=RTOL, atol=ATOL)

    transformed = patch.reexpress(
        matrix,
        shift,
        output_scale=scale,
        output_shift=offset,
    )
    transformed_physical = transformed.to_physical(probes)
    np.testing.assert_allclose(
        transformed_physical,
        physical @ matrix.T + shift,
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        transformed.evaluate_physical(transformed_physical),
        scale * patch.evaluate_physical(physical) + offset,
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("matrix,shift,scale,offset", TRANSFORMS)
def test_hessian_equivariance_and_canonical_recovery(
    patch: LocalPatch,
    matrix: np.ndarray,
    shift: np.ndarray,
    scale: float,
    offset: float,
) -> None:
    transformed = patch.reexpress(
        matrix,
        shift,
        output_scale=scale,
        output_shift=offset,
    )
    expected_hessian = (
        scale
        * np.linalg.inv(matrix).T
        @ patch.physical_hessian()
        @ np.linalg.inv(matrix)
    )
    np.testing.assert_allclose(
        transformed.physical_hessian(), expected_hessian, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        patch.recovered_canonical_curvature(), patch.curvature, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        transformed.recovered_canonical_curvature(),
        patch.curvature,
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("matrix,shift,scale,offset", TRANSFORMS)
def test_ground_truth_descriptors_are_invariant(
    patch: LocalPatch,
    matrix: np.ndarray,
    shift: np.ndarray,
    scale: float,
    offset: float,
) -> None:
    transformed = patch.reexpress(
        matrix,
        shift,
        output_scale=scale,
        output_shift=offset,
    )
    baseline = curvature_descriptor(patch)
    changed = curvature_descriptor(transformed)

    assert changed.strength == pytest.approx(baseline.strength, rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(
        changed.normalized_spectrum,
        baseline.normalized_spectrum,
        rtol=RTOL,
        atol=ATOL,
    )
    assert changed.condition_number == pytest.approx(
        baseline.condition_number, rel=RTOL, abs=ATOL
    )
    np.testing.assert_array_equal(
        rank_descriptor(transformed, canonical_probes()),
        rank_descriptor(patch, canonical_probes()),
    )


def test_invalid_coordinate_or_output_transform_is_rejected(patch: LocalPatch) -> None:
    with pytest.raises(ValueError, match="invertible"):
        patch.reexpress(np.array([[1.0, 0.0], [0.0, 0.0]]), np.zeros(2))
    with pytest.raises(ValueError, match="strictly positive"):
        patch.reexpress(np.eye(2), np.zeros(2), output_scale=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        patch.reexpress(np.eye(2), np.zeros(2), output_scale=-1.0)


def test_invalid_patch_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        LocalPatch(np.diag([1.0, 0.0]), np.zeros(2), np.eye(2))
    with pytest.raises(ValueError, match="invertible"):
        LocalPatch(np.eye(2), np.zeros(2), np.zeros((2, 2)))
