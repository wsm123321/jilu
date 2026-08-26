from types import SimpleNamespace
from local_transferability.semantic_classification import classify_semantics
T={'relative_se':.2,'spectrum_width':.3,'spd_probability':.9,'drift':.2}
def unc(se=.1,w=.1,p=.99):return SimpleNamespace(available=True,relative_magnitude_se=se,spectrum_width=w,spd_probability=p)
def test_three_way_semantic_output():
 assert classify_semantics(unc(),unc(),SimpleNamespace(available=True,normalized_drift=.1),T)=='Two-scale-stable-SPD'
 assert classify_semantics(unc(),unc(),SimpleNamespace(available=True,normalized_drift=.5),T)=='Scale-dependent'
 assert classify_semantics(unc(se=.5),unc(),SimpleNamespace(available=True,normalized_drift=.1),T)=='Unidentifiable'
