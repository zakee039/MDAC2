from dataclasses import replace
import pytest
from mdac2.domain.structure import Span,SimilarityProposal,ConfirmedTarget,ConfirmedStructureGroup,validate_relations,validate_reuse
from mdac2.domain.chart import ChartEvent,ChartRevision
from mdac2.domain.timing import TimingMap,Anchor
from mdac2.domain.timing_edits import TimingRevision


def fixture():
    timing=TimingRevision('t',TimingMap((Anchor(0,0),Anchor(32,16)),16),review='accepted')
    p=SimilarityProposal('p',Span(0,4),Span(8,12),'a'*64,timing.fingerprint,'b'*64,(('chroma',.95),),.02)
    target=ConfirmedTarget('target',p.target,p.fingerprint,'exact_repeat',('arrangement','mirror','full_pattern'))
    group=ConfirmedStructureGroup('g',p.source,(target,),timing.fingerprint,'tester','confirmed fixture')
    chart=ChartRevision('c','t',(ChartEvent('source',1,('ground',)),ChartEvent('target',9,('air',))))
    return timing,p,group,chart


def test_proposal_does_not_authorize_and_target_permission_is_independent():
    t,p,g,c=fixture()
    with pytest.raises(ValueError):validate_reuse(c,t,p,target_ids=('target',),mode='arrangement')
    source,targets=validate_reuse(c,t,g,target_ids=('target',),mode='mirror')
    assert source[0].event_id=='source' and targets[0].target_id=='target'
    readonly=replace(g,targets=(replace(g.targets[0],allowed_modes=()),))
    with pytest.raises(ValueError):validate_reuse(c,t,readonly,target_ids=('target',),mode='arrangement')
    assert g.fingerprint!=readonly.fingerprint


def test_empty_source_stale_timing_overlap_and_cycles():
    t,p,g,c=fixture()
    with pytest.raises(ValueError,match='Empty'):validate_reuse(replace(c,events=c.events[1:]),t,g,target_ids=('target',),mode='full_pattern')
    with pytest.raises(ValueError,match='Stale'):validate_reuse(c,replace(t,revision_id='new'),g,target_ids=('target',),mode='mirror')
    with pytest.raises(ValueError,match='overlap'):replace(g,primary=Span(9,11))
    reverse=replace(g,group_id='reverse',primary=Span(8,12),targets=(replace(g.targets[0],target_id='back',span=Span(0,4)),))
    with pytest.raises(ValueError,match='Cyclic'):validate_relations((g,reverse))
    validate_relations((g,))


def test_bad_relations_alignment_and_nonempty_spans():
    t,p,g,c=fixture()
    with pytest.raises(ValueError):Span(1,1)
    with pytest.raises(ValueError):replace(g.targets[0],relation='variation')
    with pytest.raises(ValueError):replace(p,similarity_by_source=(('mix',float('nan')),))
    with pytest.raises(ValueError):replace(p,alignment_anchors=((0,8),(4,7)))
    with pytest.raises(ValueError):replace(g,targets=())
