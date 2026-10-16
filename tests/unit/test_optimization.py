from dataclasses import replace
from fractions import Fraction as F
import pytest
from test_structure import fixture
from test_arrange import event
from mdac2.domain.chart import ChartRevision,ChartEvent
from mdac2.analysis.hold_semantics import SustainEvidence
from mdac2.application.optimization import preview_optimization,apply_optimization,undo_optimization,rule_report


def test_fixed_preview_hold_selection_apply_and_safe_undo():
    t,_,_,_=fixture();c=ChartRevision('c','t',(event('tap',1),))
    evidence=SustainEvidence('a'*64,t.fingerprint,((F(0),F(2),1.),(F(2),F(16),0.)),'b'*64)
    kwargs={'arrangement':True,'hold_ids':('tap',),'evidence':evidence}
    a=preview_optimization(c,t,**kwargs);b=preview_optimization(c,t,**kwargs)
    assert a.fingerprint==b.fingerprint and c.events[0].semantic=='tap'
    assert a.events[0].end_q==4 and a.changes==( ('tap',('semantic','duration_q')),)
    result=apply_optimization(c,t,expected_preview_fingerprint=a.fingerprint,**kwargs)
    assert undo_optimization(result,c,a).events==c.events
    with pytest.raises(ValueError,match='Stale'):apply_optimization(result,t,expected_preview_fingerprint=a.fingerprint,**kwargs)
    with pytest.raises(ValueError):undo_optimization(replace(result,events=()),c,a)


def test_conflict_no_silent_delete_and_report_readonly():
    t,_,_,_=fixture();c=ChartRevision('c','t',(ChartEvent('hold',0,('ground','air'),'hold',duration_q=4),event('tap',1)))
    report=rule_report(c,t);assert ('LANE_OCCUPANCY_REVIEW','warning','tap') in report
    p=preview_optimization(c,t,arrangement=True);assert p.events==c.events and p.conflicts
    with pytest.raises(ValueError,match='conflicts'):apply_optimization(c,t,expected_preview_fingerprint=p.fingerprint,arrangement=True)


def test_manual_hold_request_cannot_bypass_protection():
    t,_,_,c=fixture();evidence=SustainEvidence('a'*64,t.fingerprint,((F(0),F(16),1.),),'b'*64)
    p=preview_optimization(c,t,hold_ids=('source',),evidence=evidence)
    assert p.events==c.events and p.conflicts
    with pytest.raises(ValueError):preview_optimization(c,replace(t,review='pending'))
