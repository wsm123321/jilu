"""Three-way semantic classification under frozen thresholds."""
from __future__ import annotations

def classify_semantics(outer_unc,inner_unc,scale,thresholds):
 stats_ok=lambda u: u.available and u.relative_magnitude_se<=thresholds['relative_se'] and u.spectrum_width<=thresholds['spectrum_width'] and u.spd_probability>=thresholds['spd_probability']
 if not stats_ok(outer_unc) or not stats_ok(inner_unc) or not scale.available:return 'Unidentifiable'
 consistent=scale.normalized_drift<=thresholds['drift']
 return 'Two-scale-stable-SPD' if consistent else 'Scale-dependent'
