from dataclasses import replace
from fractions import Fraction as F
from test_structure import fixture
from test_selection import candidates
from mdac2.analysis.arrangement_baseline import arrange,chart_from_selection
from mdac2.domain.chart import ChartEvent,ChartRevision
from mdac2.domain.structure import Span
from mdac2.policies.onset import select_onsets


def event(identity,q,lane='ground',**kw):return ChartEvent(identity,q,(lane,),source_kind='candidate',source_id=identity,**kw)


def test_fixed_onsets_and_rests_without_mechanical_alternation():
    t,_,_,_=fixture();c=ChartRevision('c','t',tuple(event(str(i),i*4) for i in range(4)))
    r=arrange(c,t)
    assert r.events==c.events # separated notes have no need to switch every time
    assert [(e.event_id,e.q) for e in r.events]==[(e.event_id,e.q) for e in c.events]
    assert arrange(replace(c,events=()),t).events==()


def test_boundary_context_manual_and_hold_occupancy():
    t,_,_,_=fixture();c=ChartRevision('c','t',(ChartEvent('before',F(9,10),('ground',),locked=True),event('inside',1),ChartEvent('after',F(11,10),('ground',))))
    r=arrange(c,t,span=Span(1,2));assert r.events[1].lane_mask==('air',) and r.events[0]==c.events[0] and r.events[2]==c.events[2]
    c=replace(c,events=(ChartEvent('hold',0,('ground','air'),'hold',duration_q=3),event('inside',1)))
    r=arrange(c,t);assert r.conflicts==( ('inside','BOTH_LANES_OCCUPIED'),) and r.events==c.events


def test_selection_bridge_has_exact_ids_and_no_insertions():
    t,_,_,_=fixture();onsets=replace(candidates([.5,1,8]),timing_fingerprint=t.fingerprint);selected=select_onsets(onsets)
    c=chart_from_selection(onsets,selected,t)
    assert tuple(e.event_id for e in c.events)==selected.selected_ids
    assert all(e.q==o.quantized_q for e,o in zip(c.events,onsets.candidates))
