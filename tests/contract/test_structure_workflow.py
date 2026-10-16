from dataclasses import replace
from fractions import Fraction as F
import math
import struct
import pytest
from test_onset_workflow import prepared
from mdac2.application.structure_workflow import StructureWorkflow, envelope
from mdac2.domain.chart import ChartEvent
from mdac2.infrastructure.project_repository import RevisionConflict


def ready(tmp_path):
    pcm=b''.join(struct.pack('<hh',*(2*[round((6000+5000*math.exp(-(i%12000)/1000))*math.sin(2*math.pi*440*i/48000))])) for i in range(96000))
    repo,store,project,onsets,chart,timing=prepared(tmp_path,pcm)
    chart2=replace(chart,revision_id='structure-base',events=(*chart.events,ChartEvent('target',F(5),('ground',))))
    onsets.manual.save(project,expected_revision=chart.revision_id,command_id='target',chart=chart2,timing=timing)
    flow=StructureWorkflow(onsets.manual); candidate=flow.propose(project,chart2.revision_id)
    proposals=envelope(candidate.payload['search']).payload['proposals']
    assert proposals, candidate.payload['search']['payload']
    p=proposals[0]
    return repo,store,project,flow,chart2,timing,candidate,p


def confirm(flow,project,candidate,p,**kwargs):
    body={'primary':p['source'],'targets':[{'proposal_id':p['proposal_id'],'span':p['target'],'relation':'exact_repeat',
        'allowed_modes':['arrangement','mirror','full_pattern'],'confirmed':True}],'actor':'fixture','reason':'synthetic repeated phrase, not music gold'}
    return flow.confirm(project,candidate.artifact_hash,**{**body,**kwargs})


def test_actual_features_review_arrangement_apply_undo_and_lineage(tmp_path):
    repo,store,project,flow,chart,timing,candidate,p=ready(tmp_path)
    assert flow.manual.load(project)==(chart,timing)
    assert flow.load(project,candidate.artifact_hash).artifact_hash==candidate.artifact_hash
    review=confirm(flow,project,candidate,p)
    assert not review.payload['alignment_errors']
    assert flow.manual.load(project)==(chart,timing)
    options={'target_ids':(p['proposal_id'],),'mode':'arrangement'}
    preview=flow.preview(project,review.artifact_hash,**options)
    assert not preview['conflicts'] and preview['changes'][0]['event_id']=='target'
    assert preview['changes'][0]['after']['lane_mask']==['air']
    applied,_,operation=flow.apply(project,review.artifact_hash,preview_fingerprint=preview['fingerprint'],command_id='reuse',**options)
    assert [(e.event_id,e.q) for e in applied.events]==[(e.event_id,e.q) for e in chart.events]
    artifact=store.get('chart',repo.get_revision(project,applied.revision_id).artifact_hash)
    assert operation in artifact.to_dict()['inputs']
    with pytest.raises(RevisionConflict): flow.apply(project,review.artifact_hash,preview_fingerprint=preview['fingerprint'],command_id='repeat',**options)
    restored,_=flow.undo(project,operation,expected_revision=applied.revision_id,command_id='undo')
    assert restored.events==chart.events and restored.revision_id!=chart.revision_id


def test_full_replace_explicit_permissions_and_later_edit_blocks_undo(tmp_path):
    repo,store,project,flow,chart,timing,candidate,p=ready(tmp_path)
    review=confirm(flow,project,candidate,p)
    options={'target_ids':(p['proposal_id'],),'mode':'full_pattern'}
    preview=flow.preview(project,review.artifact_hash,**options)
    assert any(c['before'] and c['after'] is None for c in preview['changes'])
    assert any(c['before'] is None and c['after'] for c in preview['changes'])
    with pytest.raises(ValueError,match='Stale structure preview'):
        flow.apply(project,review.artifact_hash,preview_fingerprint='wrong',command_id='bad',**options)
    applied,_,operation=flow.apply(project,review.artifact_hash,preview_fingerprint=preview['fingerprint'],command_id='replace',**options)
    assert chart.events[0] in applied.events and not any(e.event_id=='target' for e in applied.events)
    edited=replace(applied,revision_id='later-edit',events=(*applied.events,ChartEvent('new',F(7),('ground',))))
    flow.manual.save(project,expected_revision=applied.revision_id,command_id='edit',chart=edited,timing=timing)
    with pytest.raises(ValueError,match='later edits'): flow.undo(project,operation,expected_revision=edited.revision_id,command_id='undo')
    assert flow.manual.load(project)==(edited,timing)


def test_changed_nonbar_boundary_is_annotation_only_and_wrong_project_rejected(tmp_path):
    repo,store,project,flow,chart,timing,candidate,p=ready(tmp_path)
    primary={**p['source'],'start_q':'1/2'}
    review=confirm(flow,project,candidate,p,primary=primary)
    assert review.payload['alignment_errors'][p['proposal_id']]
    with pytest.raises(ValueError,match='annotation only'):
        flow.preview(project,review.artifact_hash,target_ids=(p['proposal_id'],),mode='arrangement')
    with pytest.raises(ValueError,match='another project'): flow.load('other',candidate.artifact_hash)
    assert flow.manual.load(project)==(chart,timing)


def test_missing_permission_unconfirmed_target_and_commit_failure(tmp_path):
    repo,store,project,flow,chart,timing,candidate,p=ready(tmp_path)
    target={'proposal_id':p['proposal_id'],'span':p['target'],'relation':'variation','allowed_modes':['arrangement'],'confirmed':False}
    with pytest.raises(ValueError,match='explicit confirmation'): confirm(flow,project,candidate,p,targets=[target])
    review=confirm(flow,project,candidate,p,targets=[{**target,'confirmed':True}])
    with pytest.raises(ValueError,match='authorized'):
        flow.preview(project,review.artifact_hash,target_ids=(p['proposal_id'],),mode='full_pattern')
    options={'target_ids':(p['proposal_id'],),'mode':'arrangement'}
    preview=flow.preview(project,review.artifact_hash,**options)
    with repo._connection() as c:
        c.execute("CREATE TRIGGER refuse_structure BEFORE INSERT ON revision_timing BEGIN SELECT RAISE(ABORT,'injected'); END")
    with pytest.raises(RevisionConflict):
        flow.apply(project,review.artifact_hash,preview_fingerprint=preview['fingerprint'],command_id='failed',**options)
    assert flow.manual.load(project)==(chart,timing)
