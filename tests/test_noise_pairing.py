import numpy as np
from local_transferability.noise import standard_noise, scaled_noise

def test_shared_noise_and_nested_prefixes():
    base = standard_noise(20, "development", "uniform", 7)
    np.testing.assert_array_equal(scaled_noise(base,.05,.5,8)/.5, scaled_noise(base,.05,2,8)/2)
    np.testing.assert_array_equal(scaled_noise(base,.05,1,8), scaled_noise(base,.05,1,20)[:8])

def test_development_and_holdout_streams_differ():
    assert not np.array_equal(standard_noise(20,"development","uniform",1), standard_noise(20,"holdout","uniform",1))
