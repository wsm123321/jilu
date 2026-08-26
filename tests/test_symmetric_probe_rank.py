import numpy as np
from local_transferability.symmetric_probes import symmetric_inner_probes
from local_transferability.quadratic_estimator import quadratic_design_matrix

def test_two_shell_symmetric_probes_identify_complete_quadratic():
    for rho in (.25,.5):
        U=symmetric_inner_probes(rho)
        assert U.shape==(8,2)
        assert np.linalg.matrix_rank(quadratic_design_matrix(U))==6
        np.testing.assert_allclose(U.sum(axis=0),0,atol=1e-15)

def test_single_shell_original_proposal_is_rank_deficient():
    d=1/np.sqrt(2);U=np.array([[1,0],[-1,0],[0,1],[0,-1],[d,d],[-d,-d],[d,-d],[-d,d]])
    assert np.linalg.matrix_rank(quadratic_design_matrix(U))<6
