"""Ground-truth local quadratic landscape used in the first study step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _vector(value: ArrayLike, *, name: str, dimension: int) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (dimension,):
        raise ValueError(f"{name} must have shape ({dimension},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _invertible_matrix(value: ArrayLike, *, name: str, dimension: int) -> FloatArray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            f"{name} must have shape ({dimension}, {dimension}), got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if np.linalg.matrix_rank(matrix) != dimension:
        raise ValueError(f"{name} must be invertible")
    return matrix.copy()


@dataclass(frozen=True)
class LocalPatch:
    """A quadratic canonical landscape embedded in physical coordinates.

    The represented function is

        x = center + basis @ u
        y = output_shift + output_scale * 0.5 * u.T @ curvature @ u

    `curvature` is expressed in the canonical coordinate chart and is kept
    fixed when the same landscape is re-expressed in another physical chart.
    """

    curvature: FloatArray
    center: FloatArray
    basis: FloatArray
    output_scale: float = 1.0
    output_shift: float = 0.0

    def __post_init__(self) -> None:
        curvature = np.asarray(self.curvature, dtype=float)
        if curvature.ndim != 2 or curvature.shape[0] != curvature.shape[1]:
            raise ValueError("curvature must be a square matrix")
        dimension = curvature.shape[0]
        if not np.all(np.isfinite(curvature)):
            raise ValueError("curvature must contain only finite values")
        if not np.allclose(curvature, curvature.T, rtol=0.0, atol=1e-12):
            raise ValueError("curvature must be symmetric")
        if np.min(np.linalg.eigvalsh(curvature)) <= 0.0:
            raise ValueError("the first-step curvature must be positive definite")

        center = _vector(self.center, name="center", dimension=dimension)
        basis = _invertible_matrix(self.basis, name="basis", dimension=dimension)
        output_scale = float(self.output_scale)
        output_shift = float(self.output_shift)
        if not np.isfinite(output_scale) or output_scale <= 0.0:
            raise ValueError("output_scale must be finite and strictly positive")
        if not np.isfinite(output_shift):
            raise ValueError("output_shift must be finite")

        object.__setattr__(self, "curvature", curvature.copy())
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "output_scale", output_scale)
        object.__setattr__(self, "output_shift", output_shift)

    @property
    def dimension(self) -> int:
        return self.curvature.shape[0]

    def canonical_function(self, u: ArrayLike) -> FloatArray | float:
        points = np.asarray(u, dtype=float)
        if points.shape == (self.dimension,):
            return float(0.5 * points @ self.curvature @ points)
        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError(
                f"u must have shape ({self.dimension},) or (n, {self.dimension})"
            )
        return 0.5 * np.einsum("ni,ij,nj->n", points, self.curvature, points)

    def to_physical(self, u: ArrayLike) -> FloatArray:
        points = np.asarray(u, dtype=float)
        if points.shape == (self.dimension,):
            return self.center + self.basis @ points
        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError(
                f"u must have shape ({self.dimension},) or (n, {self.dimension})"
            )
        return self.center + points @ self.basis.T

    def to_canonical(self, x: ArrayLike) -> FloatArray:
        points = np.asarray(x, dtype=float)
        if points.shape == (self.dimension,):
            return np.linalg.solve(self.basis, points - self.center)
        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError(
                f"x must have shape ({self.dimension},) or (n, {self.dimension})"
            )
        return np.linalg.solve(self.basis, (points - self.center).T).T

    def evaluate_canonical(self, u: ArrayLike) -> FloatArray | float:
        return self.output_shift + self.output_scale * self.canonical_function(u)

    def evaluate_physical(self, x: ArrayLike) -> FloatArray | float:
        return self.evaluate_canonical(self.to_canonical(x))

    def physical_hessian(self) -> FloatArray:
        basis_inverse = np.linalg.inv(self.basis)
        return self.output_scale * basis_inverse.T @ self.curvature @ basis_inverse

    def recovered_canonical_curvature(self) -> FloatArray:
        from .normalization import recover_canonical_curvature

        return recover_canonical_curvature(
            self.physical_hessian(), self.basis, self.output_scale
        )

    def reexpress(
        self,
        input_matrix: ArrayLike,
        input_shift: ArrayLike,
        *,
        output_scale: float = 1.0,
        output_shift: float = 0.0,
    ) -> "LocalPatch":
        """Re-express the same canonical landscape under x'=Ax+b, y'=cy+d."""

        matrix = _invertible_matrix(
            input_matrix, name="input_matrix", dimension=self.dimension
        )
        shift = _vector(input_shift, name="input_shift", dimension=self.dimension)
        scale = float(output_scale)
        offset = float(output_shift)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("output_scale must be finite and strictly positive")
        if not np.isfinite(offset):
            raise ValueError("output_shift must be finite")

        return LocalPatch(
            curvature=self.curvature,
            center=matrix @ self.center + shift,
            basis=matrix @ self.basis,
            output_scale=scale * self.output_scale,
            output_shift=scale * self.output_shift + offset,
        )
