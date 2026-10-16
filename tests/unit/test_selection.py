from dataclasses import replace
from fractions import Fraction as F
import pytest
from mdac2.analysis.onset_candidates import OnsetCandidate,OnsetSet
from mdac2.domain.artifact import content_hash
from mdac2.domain.onset_policy import OnsetPolicy
from mdac2.policies.onset import select_onsets


def candidates(times,scores=None):
    scores=scores or [1]*len(times)
    return OnsetSet('a'*64,'b'*64,'c'*64,tuple(OnsetCandidate(content_hash(i),F(str(t)),F(i),F(i),F(0),s,.1,F(1,100)) for i,(t,s) in enumerate(zip(times,scores))))


def test_silence_and_rests_are_preserved():
    assert select_onsets(candidates([])).selected_ids==()
    onset=candidates([.5,8]);selected=select_onsets(onset)
    assert selected.selected_ids==tuple(c.candidate_id for c in onset.candidates)
    assert len(selected.decisions)==2


def test_density_is_rolling_and_strongest_wins():
    onset=candidates([.1,.3,.5,.7,1.1],[.4,.9,.8,.7,.6])
    p=OnsetPolicy('custom',0,.01,3,(1,))
    result=select_onsets(onset,policy=p)
    assert not result.decisions[0].selected and result.decisions[0].reasons==('LOCAL_DENSITY_LIMIT',)
    assert sum(d.selected for d in result.decisions)==3
    assert select_onsets(onset,policy=p)==result


def test_quantization_score_complexity_and_gap_explanations():
    onset=candidates([.1,.11,.5,1,2],[.9,.8,.01,1,1]);cs=list(onset.candidates)
    cs[3]=replace(cs[3],quantized_q=None,quantization_error_seconds=None)
    cs[4]=replace(cs[4],quantized_q=F(1,3))
    result=select_onsets(replace(onset,candidates=tuple(cs)))
    assert [d.reasons[0] for d in result.decisions]==['ACOUSTIC_SUPPORT_WITHIN_BUDGET','MINIMUM_GAP','BELOW_SCORE_THRESHOLD','NEEDS_MANUAL_QUANTIZATION','GRID_COMPLEXITY_LIMIT']
    assert OnsetPolicy.preset('easy').fingerprint!=OnsetPolicy.preset('hard').fingerprint
    with pytest.raises(ValueError):OnsetPolicy(max_notes_per_second=True)
