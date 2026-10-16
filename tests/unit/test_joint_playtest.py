"""Beat cleanup must retain real timestamps and refuse an incomplete grid."""
from pathlib import Path
import runpy
import numpy as np

deduplicate = runpy.run_path(str(Path(__file__).resolve().parents[2]/'tools/generate_joint_playtest.py'))['deduplicate_beats']


def test_close_extra_candidates_removed_without_snapping_real_beats():
    beats = np.arange(.3, 20, .4)
    observations = np.sort(np.r_[beats,beats[10]-.08,beats[30]-.08])
    retained,report = deduplicate(observations)
    np.testing.assert_array_equal(retained,beats)
    assert report['applied'] and report['retained_count']==len(beats)
    assert len(report['removed_seconds'])==2


def test_missing_pulse_does_not_get_fabricated_or_silently_reindexed():
    observations = np.delete(np.arange(.3, 20, .4),15)
    retained,report = deduplicate(observations)
    np.testing.assert_array_equal(retained,observations)
    assert not report['applied'] and not report['complete_grid']
