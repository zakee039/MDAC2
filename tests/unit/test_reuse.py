from dataclasses import replace
import pytest
from test_structure import fixture
from mdac2.domain.artifact import ArtifactEnvelope
from mdac2.domain.chart import ChartEvent
from mdac2.analysis.phrase_alignment import AlignmentResult
from mdac2.application.arrangement_reuse import preview_reuse,apply_reuse


def alignment(t,p,g):
    anchors=((g.primary.start_q,g.targets[0].span.start_q),(g.primary.end_q,g.targets[0].span.end_q))
    artifact=ArtifactEnvelope.create(artifact_type='phrase_alignment',producer={'name':'fixture','version':'1','model_hash':None,'dependencies':{}},inputs=[],options={},
        payload={'status':'accepted','proposal_fingerprint':p.fingerprint,'timing_fingerprint':t.fingerprint,'anchors':[[str(a),str(b)] for a,b in anchors]})
    return {'target':AlignmentResult(anchors,artifact)}


def test_unequal_event_counts_preserve_unmatched_q_ids_locks_and_base():
    t,p,g,c=fixture();c=replace(c,events=(*c.events,ChartEvent('s2',2,('ground',)),ChartEvent('locked',10,('air',),locked=True),ChartEvent('extra',11,('air',))))
    kwargs={'target_ids':('target',),'alignments':alignment(t,p,g)}
    preview=preview_reuse(c,t,g,**kwargs);result=apply_reuse(c,t,g,expected_preview_fingerprint=preview.fingerprint,**kwargs)
    assert c.events[1].lane_mask==('air',) and result.events[1].lane_mask==('ground',)
    assert [(e.event_id,e.q,e.duration_q,e.semantic) for e in result.events]==[(e.event_id,e.q,e.duration_q,e.semantic) for e in c.events]
    assert result.events[3:]==c.events[3:]
    assert ('extra','UNMATCHED_PRESERVED',None) in preview.decisions
    assert ('locked','LOCKED_PRESERVED','s2') in preview.decisions
    with pytest.raises(ValueError,match='Stale'):apply_reuse(result,t,g,expected_preview_fingerprint=preview.fingerprint,**kwargs)


def test_mirror_preserves_gemini_and_simultaneous_events():
    t,p,g,c=fixture();c=replace(c,events=(*c.events,ChartEvent('sg',1,('ground','air'),'gemini'),ChartEvent('tg',9,('ground','air'),'gemini')))
    preview=preview_reuse(c,t,g,target_ids=('target',),alignments=alignment(t,p,g),mode='mirror')
    assert preview.proposed_events==c.events
    assert len(preview.proposed_events)==4
    assert ('tg','UNCHANGED','sg') in preview.decisions


def test_alignment_boundaries_and_hold_crossing_are_not_silently_applied():
    t,p,g,c=fixture();c=replace(c,events=(replace(c.events[0],semantic='hold',duration_q=1),replace(c.events[1],semantic='hold',duration_q=4)))
    kwargs={'target_ids':('target',),'alignments':alignment(t,p,g)}
    preview=preview_reuse(c,t,g,**kwargs)
    assert ('HOLD_CROSSES_TARGET','target') in preview.conflicts
    with pytest.raises(ValueError,match='conflicts'):apply_reuse(c,t,g,expected_preview_fingerprint=preview.fingerprint,**kwargs)
    kwargs['alignments']['target']=replace(kwargs['alignments']['target'],anchors=((0,8),(4,11)))
    with pytest.raises(ValueError,match='boundaries'):preview_reuse(c,t,g,**kwargs)
