import numpy as np
from local_transferability.symmetric_probes import symmetric_inner_probes
from local_transferability.landscape_families import orthogonal_curvature,evaluate_family
from local_transferability.quadratic_estimator import fit_complete_quadratic
from local_transferability.multiscale_descriptors import normalized_scale_drift

def test_strong_quartic_creates_detectable_scale_drift():
    K=orthogonal_curvature(1,4,.2);axis=np.linspace(-1,1,21);outer=np.array(np.meshgrid(axis,axis)).reshape(2,-1).T;inner=symmetric_inner_probes(.25)
    Ko=fit_complete_quadratic(outer,evaluate_family(outer,K,family='axis_quartic',strength=.5)).hessian;Ki=fit_complete_quadratic(inner,evaluate_family(inner,K,family='axis_quartic',strength=.5)).hessian
    assert normalized_scale_drift(Ko,Ki)>.2
