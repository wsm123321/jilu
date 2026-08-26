from local_transferability.symmetric_probes import symmetric_inner_probes

def test_structured_probe_budget_is_exactly_eight():
    assert len(symmetric_inner_probes(.25))==8
    assert len(symmetric_inner_probes(.5))==8

def test_total_billed_budget_contract():
    for outer in (8,12):
        assert outer+len(symmetric_inner_probes(.5)) in (16,20)
