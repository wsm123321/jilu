import numpy as np
from local_transferability.landscape_families import evaluate_landscape, orthogonal_curvature
from local_transferability.variance_decomposition import dense_quadratic_projection

def test_zero_quartic_projection_recovers_center_hessian():
    axis=np.linspace(-1,1,81); U=np.array(np.meshgrid(axis,axis)).reshape(2,-1).T; K=orthogonal_curvature(1,4,.3)
    projection=dense_quadratic_projection(U,evaluate_landscape(U,K,beta=0))
    np.testing.assert_allclose(projection,K,rtol=1e-11,atol=1e-11)

def test_quartic_projection_differs_from_center_hessian():
    axis=np.linspace(-1,1,81); U=np.array(np.meshgrid(axis,axis)).reshape(2,-1).T; K=orthogonal_curvature(1,4,.3)
    projection=dense_quadratic_projection(U,evaluate_landscape(U,K,beta=.5))
    assert np.linalg.norm(projection-K,'fro') > .1
