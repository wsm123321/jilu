"""Scale-consistency measurements with independent outer/inner data."""
from __future__ import annotations
from dataclasses import dataclass
from .multiscale_descriptors import normalized_scale_drift,covariance_standardized_drift
@dataclass(frozen=True)
class ScaleConsistency:
    available: bool
    normalized_drift: float | None
    standardized_drift: float | None

def compare_scales(outer_estimate,inner_estimate,outer_uncertainty,inner_uncertainty):
 if not outer_estimate.identifiable or not inner_estimate.identifiable or not outer_uncertainty.available or not inner_uncertainty.available:return ScaleConsistency(False,None,None)
 return ScaleConsistency(True,normalized_scale_drift(outer_estimate.hessian,inner_estimate.hessian),covariance_standardized_drift(outer_estimate.hessian,inner_estimate.hessian,outer_uncertainty.covariance_hessian,inner_uncertainty.covariance_hessian))
