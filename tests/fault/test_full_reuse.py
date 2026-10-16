from dataclasses import replace
import pytest
from mdac2.domain.artifact import ArtifactEnvelope
from mdac2.domain.chart import ChartEvent,ChartRevision
from mdac2.domain.structure import Span,SimilarityProposal,ConfirmedTarget,ConfirmedStructureGroup
from mdac2.domain.timing import TimingMap,Anchor
from mdac2.domain.timing_edits import TimingRevision
from mdac2.analysis.phrase_alignment import AlignmentResult
from mdac2.application.full_pattern_reuse import preview_full_reuse,apply_full_reuse,undo_full_reuse


def fixture():
    t=TimingRevision('t',TimingMap((Anchor(0,0),Anchor(8,4),Anchor(16,12)),12),review='accepted')
    p=SimilarityProposal('p',Span(0,4),Span(8,12),'a'*64,t.fingerprint,'b'*64,(('mix',1.),),.1)
    g=ConfirmedStructureGroup('g',p.source,(ConfirmedTarget('target',p.target,p.fingerprint,'exact_repeat',('full_pattern',)),),t.fingerprint,'tester','fixture')
    c=ChartRevision('c','t',(ChartEvent('s',1,('ground',)),ChartEvent('old',10,('air',))))
    anchors=((p.source.start_q,p.target.start_q),(p.source.end_q,p.target.end_q))
    a=ArtifactEnvelope.create(artifact_type='phrase_alignment',producer={'name':'fixture','version':'1','model_hash':None,'dependencies':{}},inputs=[],options={},payload={'status':'accepted','proposal_fingerprint':p.fingerprint,'timing_fingerprint':t.fingerprint,'anchors':[[str(x),str(y)] for x,y in anchors]})
    return t,g,c,{'target_ids':('target',),'alignments':{'target':AlignmentResult(anchors,a)}}


def test_diff_copy_uses_q_local_timing_and_undo_is_guarded():
    t,g,c,k=fixture();p=preview_full_reuse(c,t,g,**k)
    assert [e.event_id for e in p.deleted]==['old'] and p.added[0].q==9
    assert t.mapping.t(p.added[0].q)==5 # not source seconds + fixed 4 seconds
    result=apply_full_reuse(c,t,g,expected_preview_fingerprint=p.fingerprint,**k)
    assert undo_full_reuse(result,c,p).events==c.events
    assert c.events[1].event_id=='old'
    with pytest.raises(ValueError,match='Stale'):apply_full_reuse(result,t,g,expected_preview_fingerprint=p.fingerprint,**k)
    with pytest.raises(ValueError):undo_full_reuse(replace(result,events=result.events[:-1]),c,p)


def test_empty_primary_and_locked_targets_never_clear_content():
    t,g,c,k=fixture()
    with pytest.raises(ValueError,match='Empty'):preview_full_reuse(replace(c,events=c.events[1:]),t,g,**k)
    c=replace(c,events=(c.events[0],replace(c.events[1],locked=True)));p=preview_full_reuse(c,t,g,**k)
    assert p.reuse.conflicts
    with pytest.raises(ValueError,match='conflicts'):apply_full_reuse(c,t,g,expected_preview_fingerprint=p.fingerprint,**k)
    assert preview_full_reuse(c,t,g,locked='skip_target',**k).reuse.proposed_events==c.events


def test_hold_boundaries_require_explicit_policy_and_effective_nonempty_source():
    t,g,c,k=fixture();c=replace(c,events=(replace(c.events[0],semantic='hold',duration_q=5),c.events[1]))
    assert preview_full_reuse(c,t,g,**k).reuse.conflicts
    assert ('EMPTY_EFFECTIVE_PRIMARY',None) in preview_full_reuse(c,t,g,source_hold='skip',**k).reuse.conflicts
    p=preview_full_reuse(c,t,g,source_hold='truncate',**k);assert p.added[0].q==9 and p.added[0].end_q==12
    c=replace(c,events=(replace(c.events[0],semantic='tap',duration_q=0),ChartEvent('long',7,('ground',),'hold',duration_q=6)))
    assert preview_full_reuse(c,t,g,**k).reuse.conflicts
    p=preview_full_reuse(c,t,g,target_hold='truncate',**k)
    assert p.modified[0][1].end_q==8 and any(e.q==12 and e.end_q==13 for e in p.added)
    assert not p.reuse.conflicts
    assert ('PRESERVED_HOLD_COLLISION','long') in preview_full_reuse(c,t,g,target_hold='preserve',**k).reuse.conflicts
