"""Independent strategy orchestration."""
from .postprocess import process
from .timing import snap
from mdac2.data_preparation.playability_audit import audit_playability


def run(events, features, prediction, duration, prior, timing):
    out, report = process(events, features, prediction, duration, prior)
    out, grid = snap(out, duration, timing)
    return out, report, grid, audit_playability(out, duration)
