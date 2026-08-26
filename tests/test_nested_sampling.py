import numpy as np
from local_transferability.landscape_families import orthogonal_curvature
from local_transferability.sampling import sample_design

def test_static_designs_are_nested_prefixes():
    K=orthogonal_curvature(1,4,.2)
    for name in ('sobol','uniform'):
        short=sample_design(name,8,91,K); long=sample_design(name,20,91,K)
        np.testing.assert_array_equal(short,long[:8])

def test_orthogonal_curvature_controls_magnitude_and_ratio():
    for q in (.5,1,2):
      for r in (1,4,16):
       K=orthogonal_curvature(q,r,.37); eig=np.linalg.eigvalsh(K)
       assert np.isclose(np.linalg.norm(K,'fro')/np.sqrt(2),q)
       assert np.isclose(eig[-1]/eig[0],r)
