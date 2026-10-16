from dataclasses import replace
from fractions import Fraction as F
import struct
from test_structure import fixture
from test_arrange import event
from mdac2.domain.chart import ChartRevision,ChartEvent
from mdac2.analysis.hold_semantics import sustain_evidence,propose_holds,SustainEvidence
from mdac2.ports.timing import TimingInput


def test_real_pcm_sustain_and_silence_do_not_create_edge_holds():
    t,_,_,_=fixture();a=TimingInput('a'*64,1000,16000,2)
    pcm=b''.join(struct.pack('<hh',5000,-5000) if 500<=i<1500 else bytes(4) for i in range(16000))
    evidence=sustain_evidence(a,pcm,t);c=ChartRevision('c','t',(event('start',0),event('supported',1),event('end',31)))
    result=propose_holds(c,t,evidence)
    assert [r.end_q for r in result]==[None,F(3),None]
    assert all(e.semantic=='tap' for e in c.events)


def test_tail_conflict_and_gemini_manual_protection():
    t,_,_,_=fixture();evidence=SustainEvidence('a'*64,t.fingerprint,((F(0),F(16),1.),),'b'*64)
    c=ChartRevision('c','t',(event('tap',1),ChartEvent('manual',2,('ground',)),event('gemini',3,semantic='gemini')))
    c=replace(c,events=(*c.events[:2],replace(c.events[2],lane_mask=('ground','air'))))
    r=propose_holds(c,t,evidence)
    assert [x.reason for x in r]==['TAIL_OR_LANE_CONFLICT','MANUAL_OR_LOCKED_PRESERVED','EXISTING_OR_DOUBLE_LANE_SEMANTIC_PRESERVED']


def test_variable_tempo_tail_uses_inverse_mapping():
    from mdac2.domain.timing import TimingMap,Anchor
    t,_,_,_=fixture();t=replace(t,mapping=TimingMap((Anchor(0,0),Anchor(4,2),Anchor(8,6)),6))
    evidence=SustainEvidence('a'*64,t.fingerprint,((F(0),F(3),1.),(F(3),F(6),0.)),'b'*64)
    r=propose_holds(ChartRevision('c','t',(event('tap',3),)),t,evidence)
    assert r[0].end_q==5
