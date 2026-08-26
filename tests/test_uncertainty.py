import numpy as np
from local_transferability.landscape_families import evaluate_landscape, orthogonal_curvature
from local_transferability.quadratic_estimator import fit_complete_quadratic
from local_transferability.uncertainty import estimate_uncertainty

def test_residual_uncertainty_unavailable_at_saturation():
    rng=np.random.default_rng(4); U=rng.uniform(-1,1,(6,2)); y=evaluate_landscape(U,orthogonal_curvature(1,4,.3))
    est=fit_complete_quadratic(U,y); unc=estimate_uncertainty(U,y,est)
    assert unc.residual_df == 0 and unc.sigma2 is None and unc.relative_magnitude_se is None

def test_oracle_uncertainty_available_at_saturation():
    rng=np.random.default_rng(4); U=rng.uniform(-1,1,(6,2)); y=evaluate_landscape(U,orthogonal_curvature(1,4,.3))
    est=fit_complete_quadratic(U,y); unc=estimate_uncertainty(U,y,est,known_sigma=.05)
    assert unc.sigma2 == .05**2 and unc.relative_magnitude_se is not None
