from dataclasses import replace
from test_structure import fixture
from mdac2.domain.chart import ChartEvent
from mdac2.domain.structure import Span
from mdac2.evaluation.business_metrics import business_metrics,edit_counts


def test_rest_hold_overlap_and_unknown_game_rule():
    t,_,_,c=fixture();c=replace(c,events=(ChartEvent('hold',1,('ground',),'hold',duration_q=4),ChartEvent('gemini',9,('ground','air'),'gemini')))
    p=business_metrics(c,t,rests=(Span(2,4),Span(12,16))).payload
    assert p['rest_preservation']==.5 and p['forbidden']['rate'] is None
    assert p['logical_events']==2 and p['physical_entities']==3 and p['peak_nps']['physical']==2


def test_multiple_creative_lane_references_have_same_onset_coverage():
    t,_,_,c=fixture();other=replace(c,revision_id='alternate',events=tuple(replace(e,lane_mask=('air',)) for e in c.events))
    p=business_metrics(c,t,references=(c,other)).payload
    assert len(p['reference_comparisons'])==2
    assert all(r['onset_coverage']['scores']['baseline']['recall']==1 for r in p['reference_comparisons'])


def test_edit_amount_tracks_ids_and_locked_changes():
    t,_,_,c=fixture();base=replace(c,events=(replace(c.events[0],locked=True),c.events[1]))
    after=replace(base,events=(replace(base.events[0],q=2),ChartEvent('new',4,('air',))))
    counts=edit_counts(base,after)
    assert counts['added']==counts['deleted']==counts['moved']==counts['locked_modified_or_deleted']==1
