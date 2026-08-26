import numpy as np
from local_transferability.identifiability_metrics import compute_identifiability_metrics

def values(U,K): return .5*np.einsum('ni,ij,nj->n',U,K,U)

def test_absolute_spectrum_cannot_hide_inertia_error():
    truth=np.diag([1.,4.]); estimate=np.diag([-1.,4.]); U=np.array([[.2,.1],[-.4,.3],[.7,-.2]])
    m=compute_identifiability_metrics(truth,estimate,values(U,truth),values(U,estimate))
    assert m.spectrum_l1_error == 0.0
    assert not m.is_spd and m.inertia_mismatch and m.negative_eigenvalue_fraction == .5
