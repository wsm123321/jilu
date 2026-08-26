import numpy as np
from local_transferability.symmetric_probes import symmetric_inner_probes
from local_transferability.landscape_families import orthogonal_curvature,evaluate_family
from local_transferability.quadratic_estimator import fit_complete_quadratic
from local_transferability.multiscale_descriptors import normalized_scale_drift

def test_exact_quadratic_has_zero_two_scale_drift():
    K=orthogonal_curvature(1,4,.4);rng=np.random.default_rng(2);outer=rng.uniform(-1,1,(20,2));inner=symmetric_inner_probes(.25)
    Ko=fit_complete_quadratic(outer,evaluate_family(outer,K)).hessian;Ki=fit_complete_quadratic(inner,evaluate_family(inner,K)).hessian
    assert normalized_scale_drift(Ko,Ki)<1e-10
