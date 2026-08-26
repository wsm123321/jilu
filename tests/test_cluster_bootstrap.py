import math
from local_transferability.clustered_risk import cluster_bootstrap_risk_ucb

def test_cluster_bootstrap_resamples_whole_seed_clusters():
    seeds=[1,1,2,2,3,3];accepted=[1,1,1,1,1,1];reliable=[1,1,1,0,0,0]
    point,ucb=cluster_bootstrap_risk_ucb(seeds,accepted,reliable,bootstrap_seed=3,draws=2000)
    assert point==.5 and ucb>=point

def test_no_selection_has_undefined_risk():
    point,ucb=cluster_bootstrap_risk_ucb([1,2],[0,0],[1,0])
    assert math.isnan(point) and math.isnan(ucb)
