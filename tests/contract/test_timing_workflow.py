from dataclasses import replace
from fractions import Fraction
import pytest

from test_manual_workflow import initialized
from mdac2.application.manual_workflow import ManualWorkflow
from mdac2.application.timing_workflow import TimingWorkflow
from mdac2.domain.artifact import canonical_json
from mdac2.infrastructure.project_repository import RevisionConflict
from mdac2.ports.timing import TimingInput, TimingObservation


def prepare(tmp_path,beats=None):
    repo,store,api,project,url,chart,timing=initialized(tmp_path)
    flow=TimingWorkflow(ManualWorkflow(repo,store))
    audio,_=flow.context(project,'c')
    observation=TimingObservation.create(TimingInput(audio.artifact_hash,48000,96000,2),
        producer={'name':'fixture','version':'1','model_hash':None,'dependencies':{}},options={},
        beat_times=beats if beats is not None else [i*.25 for i in range(9)],downbeat_times=[0,1])
    candidate=flow.propose(project,'c',observation)
    return repo,store,project,flow,candidate,chart,timing


def test_candidate_review_persists_provenance_without_changing_head_until_adopt(tmp_path):
    repo,store,project,flow,candidate,chart,timing=prepare(tmp_path)
    assert repo.get_project(project).head_revision=='c'
    assert candidate.payload['timing']['review']=='pending'
    assert candidate.payload['timing']['downbeat_q']==[]
    preview=flow.preview(project,candidate.artifact_hash,'keep_audio_time')
    assert preview['diffs'][0]['new_q']=='2'
    chart,timing=flow.adopt(project,candidate.artifact_hash,position_policy='keep_audio_time',preview_fingerprint=preview['fingerprint'],
        actor='tester',reason='listened locally',command_id='adopt')
    assert timing.review=='accepted' and chart.events[0].q==2
    assert ManualWorkflow(repo,store).load(project)==(chart,timing)
    with repo._connection() as c:
        h=c.execute('SELECT artifact_hash FROM revision_timing WHERE revision_id=?',(chart.revision_id,)).fetchone()[0]
    envelope=store.get('timing',h)
    review=store.get('timing_review',envelope.to_dict()['inputs'][1])
    assert review.to_dict()['inputs']==[candidate.artifact_hash]
    assert review.payload['reviewed_fingerprint']==timing.fingerprint
    new=replace(chart,revision_id='chart-edit')
    flow.manual.save(project,expected_revision=chart.revision_id,command_id='edit',chart=new,timing=timing)
    with repo._connection() as c:
        assert c.execute('SELECT artifact_hash FROM revision_timing WHERE revision_id=?',('chart-edit',)).fetchone()[0]==h
    with pytest.raises(RevisionConflict):flow.adopt(project,candidate.artifact_hash,position_policy='keep_audio_time',preview_fingerprint=preview['fingerprint'],actor='a',reason='r',command_id='again')


def test_bad_fit_wrong_project_audio_and_preview_are_rejected(tmp_path):
    repo,store,project,flow,candidate,chart,timing=prepare(tmp_path,beats=[.5,1])
    assert candidate.payload['fit']['accepted'] is False
    with pytest.raises(ValueError,match='Rejected fit'):flow.preview(project,candidate.artifact_hash,'keep_audio_time')
    with pytest.raises(ValueError,match='another project'):flow.load('other',candidate.artifact_hash)
    bad=TimingObservation.create(TimingInput('a'*64,48000,96000,2),producer={'name':'fixture','version':'1','model_hash':None,'dependencies':{}},options={},beat_times=[.5,1],downbeat_times=[])
    with pytest.raises(ValueError,match='project audio'):flow.propose(project,'c',bad)


def test_stale_diff_locked_notes_and_failed_commit_preserve_authority(tmp_path):
    repo,store,project,flow,candidate,chart,timing=prepare(tmp_path)
    args=dict(position_policy='keep_audio_time',actor='a',reason='r',command_id='adopt')
    with pytest.raises(ValueError,match='migration'):flow.adopt(project,candidate.artifact_hash,preview_fingerprint='bad',**args)
    preview=flow.preview(project,candidate.artifact_hash,'keep_audio_time')
    with repo._connection() as c:
        c.execute("CREATE TRIGGER refuse_auto BEFORE INSERT ON revision_timing BEGIN SELECT RAISE(ABORT,'injected'); END")
    with pytest.raises(RevisionConflict):flow.adopt(project,candidate.artifact_hash,preview_fingerprint=preview['fingerprint'],**args)
    assert flow.manual.load(project)==(chart,timing)
    with repo._connection() as c:c.execute('DROP TRIGGER refuse_auto')
    locked=replace(chart,revision_id='locked',events=(replace(chart.events[0],locked=True),))
    flow.manual.save(project,expected_revision='c',command_id='lock',chart=locked,timing=timing)
    observation=TimingObservation(canonical_json(candidate.payload['observation']))
    candidate=flow.propose(project,'locked',observation)
    assert flow.preview(project,candidate.artifact_hash,'keep_audio_time')['conflicts'][0]['code']=='LOCKED_EVENT_CHANGED'


def test_manual_api_cannot_forge_initial_review(tmp_path):
    from test_upload import setup,audio
    from mdac2.api.manual import attach_manual
    from mdac2.application.manual_workflow import timing_data
    from mdac2.domain.chart import ChartRevision
    from mdac2.domain.timing import TimingMap,Anchor
    from mdac2.domain.timing_edits import TimingRevision
    repo,store,api,project=setup(tmp_path)
    attach_manual(api.app,ManualWorkflow(repo,store))
    api.post(f'/api/v1/projects/{project}/audio',content=audio(96000),headers={'Content-Type':'audio/wav'})
    timing=TimingRevision('t',TimingMap((Anchor(0,0),Anchor(4,2)),2),review='accepted')
    response=api.post(f'/api/v1/projects/{project}/manual',json=dict(expected_revision=None,command_id='forge',chart=ChartRevision('c','t',()).to_dict(),
        timing=timing_data(timing),position_policy=None,preview_fingerprint=None))
    assert response.status_code==422 and repo.get_project(project).head_revision is None
