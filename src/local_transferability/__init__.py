"""Core objects for the independent local-transferability study."""

from .descriptors import (
    CurvatureDescriptor,
    curvature_descriptor,
    rank_descriptor,
    stable_rank_encoding,
)
from .local_patch import LocalPatch
from .normalization import finite_difference_hessian, recover_canonical_curvature
from .quadratic_estimator import QuadraticEstimate, fit_complete_quadratic

__all__ = [
    "CurvatureDescriptor",
    "LocalPatch",
    "QuadraticEstimate",
    "curvature_descriptor",
    "finite_difference_hessian",
    "fit_complete_quadratic",
    "recover_canonical_curvature",
    "rank_descriptor",
    "stable_rank_encoding",
]
