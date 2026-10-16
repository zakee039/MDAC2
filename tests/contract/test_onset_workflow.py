from dataclasses import replace
from fractions import Fraction as F
import io
import math
import struct
import wave
import pytest

from test_upload import setup
from mdac2.application.manual_workflow import ManualWorkflow
from mdac2.application.onset_workflow import OnsetWorkflow
from mdac2.application.timing_workflow import TimingWorkflow
from mdac2.domain.chart import ChartEvent, ChartRevision
from mdac2.domain.timing import Anchor, TimingMap
from mdac2.domain.timing_edits import TimingRevision
from mdac2.infrastructure.project_repository import RevisionConflict
from mdac2.ports.timing import TimingInput, TimingObservation


def prepared(tmp_path, pcm_override=None):
    repo, store, api, project = setup(tmp_path)
    stream = io.BytesIO()
    pcm = pcm_override if pcm_override is not None else b''.join(struct.pack('<hh', *(2*[round(18000*math.exp(-(i%12000)/600)*math.sin(i*.6))])) for i in range(96000))
    with wave.open(stream, 'wb') as wav:
        wav.setparams((2, 2, 48000, 0, 'NONE', 'none')); wav.writeframes(pcm)
    assert api.post(f'/api/v1/projects/{project}/audio', content=stream.getvalue(), headers={'Content-Type':'audio/wav'}).status_code == 201
    manual = ManualWorkflow(repo, store)
    timing = TimingRevision('t', TimingMap((Anchor(0, 0), Anchor(8, 2)), 2))
    chart = ChartRevision('c', 't', (ChartEvent('manual', F(1), ('air',), locked=True),))
    manual.save(project, expected_revision=None, command_id='initial', chart=chart, timing=timing)
    timeflow = TimingWorkflow(manual); audio, _ = timeflow.context(project, 'c')
    observation = TimingObservation.create(TimingInput(audio.artifact_hash, 48000, 96000, 2),
        producer={'name':'fixture','version':'1','model_hash':None,'dependencies':{}}, options={},
        beat_times=[i*.25 for i in range(9)], downbeat_times=[0, 1, 2])
    candidate = timeflow.propose(project, 'c', observation)
    preview = timeflow.preview(project, candidate.artifact_hash, 'keep_musical_position')
    chart, timing = timeflow.adopt(project, candidate.artifact_hash, position_policy='keep_musical_position',
        preview_fingerprint=preview['fingerprint'], actor='fixture', reason='synthetic fixture only', command_id='review')
    return repo, store, project, OnsetWorkflow(manual), chart, timing


def adopt(flow, project, candidate, preview, command='onsets'):
    return flow.adopt(project, candidate.artifact_hash, selected_ids=tuple(preview['selected_ids']),
        preview_fingerprint=preview['fingerprint'], actor='fixture', reason='reviewed generated diff', command_id=command)


def test_real_detection_persistence_protection_and_evidence_closure(tmp_path):
    repo, store, project, flow, chart, timing = prepared(tmp_path)
    candidate = flow.propose(project, chart.revision_id)
    assert len(candidate.payload['onsets']['candidates']) >= 5
    assert repo.get_project(project).head_revision == chart.revision_id
    preview = flow.preview(project, candidate.artifact_hash)
    assert preview['preserved_ids'] == ['manual'] and len(preview['owned_events']) >= 3
    changed, same_timing = adopt(flow, project, candidate, preview)
    assert same_timing == timing and chart.events[0] in changed.events
    assert OnsetWorkflow(ManualWorkflow(repo, store)).manual.load(project) == (changed, timing)
    chart_artifact = store.get('chart', repo.get_revision(project, changed.revision_id).artifact_hash)
    assert changed.onset_revision in chart_artifact.to_dict()['inputs']
    review = store.get('onset_review', changed.onset_revision)
    assert review.to_dict()['inputs'] == [candidate.artifact_hash]
    with pytest.raises(RevisionConflict): adopt(flow, project, candidate, preview, 'again')


def test_regeneration_preserves_edits_locks_and_deletions_across_repeated_adoptions(tmp_path):
    repo, store, project, flow, chart, timing = prepared(tmp_path)
    candidate = flow.propose(project, chart.revision_id)
    chart, _ = adopt(flow, project, candidate, flow.preview(project, candidate.artifact_hash))
    generated = [e for e in chart.events if e.source_kind == 'candidate']
    removed, moved, locked = generated[:3]
    moved_new = replace(moved, q=moved.q+F(1,8), lane_mask=('air',))
    locked_new = replace(locked, locked=True)
    edited = replace(chart, revision_id='edited', events=tuple(
        moved_new if e==moved else locked_new if e==locked else e for e in chart.events if e!=removed))
    flow.manual.save(project, expected_revision=chart.revision_id, command_id='edit', chart=edited, timing=timing)
    for index in range(2):
        candidate = flow.propose(project, edited.revision_id)
        preview = flow.preview(project, candidate.artifact_hash)
        edited, _ = adopt(flow, project, candidate, preview, f'regenerate-{index}')
        assert moved_new in edited.events and locked_new in edited.events
        assert not any(e.q in (removed.q, moved.q) for e in edited.events)
        assert removed.event_id not in {e.event_id for e in edited.events}
    # Deleting an already protected edited candidate is also remembered.
    deleted = replace(edited, revision_id='delete-edited', events=tuple(e for e in edited.events if e != moved_new))
    flow.manual.save(project, expected_revision=edited.revision_id, command_id='delete-edit', chart=deleted, timing=timing)
    candidate = flow.propose(project, deleted.revision_id)
    preview = flow.preview(project, candidate.artifact_hash)
    assert moved.event_id not in {e['event_id'] for e in preview['events']}


def test_rejects_wrong_scope_bad_selection_unreviewed_preview_and_atomic_commit_failure(tmp_path):
    repo, store, project, flow, chart, timing = prepared(tmp_path)
    candidate = flow.propose(project, chart.revision_id)
    with pytest.raises(ValueError, match='another project'): flow.load('other', candidate.artifact_hash)
    with pytest.raises(ValueError, match='Unknown selected'): flow.preview(project, candidate.artifact_hash, selected_ids=('unknown',))
    preview = flow.preview(project, candidate.artifact_hash)
    with pytest.raises(ValueError, match='current onset preview'):
        adopt(flow, project, candidate, {**preview, 'fingerprint':'wrong'})
    with repo._connection() as c:
        c.execute("CREATE TRIGGER refuse_onsets BEFORE INSERT ON revision_timing BEGIN SELECT RAISE(ABORT,'injected'); END")
    with pytest.raises(RevisionConflict): adopt(flow, project, candidate, preview)
    assert flow.manual.load(project) == (chart, timing)
    with repo._connection() as c: c.execute('DROP TRIGGER refuse_onsets')
    adopt(flow, project, candidate, preview)


def test_empty_selection_does_not_remove_manual_events_and_requires_accepted_timing(tmp_path):
    repo, store, project, flow, chart, timing = prepared(tmp_path)
    candidate = flow.propose(project, chart.revision_id)
    preview = flow.preview(project, candidate.artifact_hash, selected_ids=())
    assert preview['events'] == [chart.events[0].to_dict()] and preview['changes'] == []
    pending = replace(timing, revision_id='pending', review='pending', parent_fingerprint=timing.fingerprint)
    migration = flow.manual.preview(project, chart.revision_id, pending, 'keep_musical_position')
    changed = replace(chart, revision_id='pending-chart', timing_revision='pending')
    flow.manual.save(project, expected_revision=chart.revision_id, command_id='pending', chart=changed, timing=pending,
        position_policy='keep_musical_position', preview_fingerprint=migration.fingerprint)
    with pytest.raises(ValueError, match='Approved timing'): flow.propose(project, changed.revision_id)
