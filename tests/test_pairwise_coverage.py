import pytest
from local_transferability.identifiability_metrics import pairwise_order_accuracy

def test_accuracy_is_reported_with_coverage():
    accuracy, coverage, tie_fraction = pairwise_order_accuracy([0,1,2],[0,1,1],tie_tolerance=1e-10)
    assert accuracy == 1.0
    assert coverage == pytest.approx(2/3)
    assert tie_fraction == pytest.approx(1/3)
