import numpy as np
from local_transferability.landscape_families import orthogonal_curvature,evaluate_family
from local_transferability.quadratic_estimator import fit_complete_quadratic
from local_transferability.joint_uncertainty import joint_curvature_uncertainty

def test_joint_uncertainty_reproducible_and_spd_probability_calibrates_direction():
    rng=np.random.default_rng(5);U=rng.uniform(-1,1,(20,2));K=orthogonal_curvature(1,4,.2);y=evaluate_family(U,K)+rng.normal(0,.01,20);est=fit_complete_quadratic(U,y)
    a=joint_curvature_uncertainty(U,y,est,seed=9);b=joint_curvature_uncertainty(U,y,est,seed=9)
    assert a.available and 0<=a.spd_probability<=1
    np.testing.assert_array_equal(a.draws,b.draws)
