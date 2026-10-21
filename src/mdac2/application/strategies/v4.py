"""V4 processing and its final gate."""
from mdac2.data_preparation.global_postprocess_v4 import process
from mdac2.application.studio_timing_v4 import apply
from mdac2.data_preparation.playability_audit_v4 import audit_playability_v4


def run(events, features, prediction, duration, prior, timing):
    out, report = process(events, features, prediction, duration, prior)
    out, grid = apply(out, duration, timing)
    return out, report, grid, audit_playability_v4(out, duration, timing=timing)
