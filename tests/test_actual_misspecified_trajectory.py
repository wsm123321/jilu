import numpy as np
from local_transferability.landscape_families import orthogonal_curvature,evaluate_family

def trajectory(K,family,strength,seed):
 rng=np.random.default_rng(seed);pts=[];obs=[]
 for i in range(12):
  x=rng.uniform(-1,1,2) if i<2 else np.clip(pts[int(np.argmin(obs))]+rng.normal(0,max(.08,.8*.75**(i-2)),2),-1,1)
  pts.append(x);obs.append(float(evaluate_family(np.array([x]),K,family=family,strength=strength)[0]))
 return np.array(pts)
def test_actual_misspecified_observations_can_change_adaptive_path():
 K=orthogonal_curvature(1,4,.2)
 # Frozen oscillatory case changes incumbent ordering under the same innovation stream.
 assert not np.array_equal(trajectory(K,'quadratic',0,1),trajectory(K,'oscillatory',.5,1))
