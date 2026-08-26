import numpy as np
import pytest
from local_transferability.landscape_families import center_hessian,evaluate_family,orthogonal_curvature
from local_transferability.normalization import finite_difference_hessian

@pytest.mark.parametrize('family,strength',[('quadratic',0),('axis_quartic',.5),('cross_quartic',.5),('rotated_quartic',.5),('cubic',.2),('oscillatory',.05)])
def test_analytic_center_hessian_matches_function_value_finite_difference(family,strength):
    K=orthogonal_curvature(1,4,.3)
    numerical=finite_difference_hessian(lambda u: evaluate_family(np.asarray([u]),K,family=family,strength=strength)[0],np.zeros(2),step=2e-4)
    np.testing.assert_allclose(numerical,center_hessian(K,family,strength),rtol=2e-5,atol=2e-5)

def test_cubic_and_oscillatory_center_hessian_are_not_base_curvature():
    K=np.eye(2)
    assert center_hessian(K,'cubic',.2)[0,0]==pytest.approx(1+1.38*.2)
    assert center_hessian(K,'oscillatory',.05)[0,0]==pytest.approx(1+(5.3**2)*np.cos(.37)*.05)
