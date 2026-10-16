"""Synthetic engineering chain; no human/music accuracy or game acceptance."""
import io
import zipfile
from test_structure_api import setup
from mdac2.api.onsets import attach_onsets
from mdac2.application.onset_workflow import OnsetWorkflow
from mdac2.application.optimization import preview_optimization, apply_optimization
from mdac2.application.backup import backup_workspace, restore_workspace
from mdac2.application.manual_workflow import ManualWorkflow
from mdac2.application.export_builder import build_export
from mdac2.infrastructure.artifact_store import ArtifactStore
from mdac2.infrastructure.job_backend import JobBackend


def test_mixed_analysis_reuse_optimization_restore_export_chain(tmp_path):
    api, url, worker, structure, base, timing = setup(tmp_path)
    project = url.rsplit('/', 1)[1]
    attach_onsets(api.app, OnsetWorkflow(structure.manual), worker.backend)

    def post(path, body, status=200):
        response = api.post(path, json=body)
        assert response.status_code == status, response.text
        return response.json()

    def analyze(kind, revision, **options):
        job = post(url+'/'+kind+'-analysis', {'expected_revision': revision, **options}, 202)
        worker.tick()
        assert worker.backend.get_job(job['job_id']).state == 'succeeded'
        response = api.get('/api/v1/jobs/'+job['job_id']+'/'+kind+'-analysis')
        assert response.status_code == 200, response.text
        return response.json()

    try:
        onset = analyze('onset', base.revision_id, difficulty='normal')
        path = url+'/onset-analysis/'+onset['candidate_id']
        preview = post(path+'/preview', {'selected_ids': None})
        assert preview['owned_events']
        adopted = post(path+'/adopt', {'selected_ids': preview['selected_ids'],
            'preview_fingerprint': preview['fingerprint'], 'actor': 'synthetic-test',
            'reason': 'engineering diff only', 'command_id': 'chain-onsets'})
        result = analyze('structure', adopted['chart']['revision_id'])
        p = result['proposals'][0]
        review = post(url+'/structure-analysis/'+result['candidate_id']+'/confirm', {
            'primary': p['source'], 'targets': [{'proposal_id': p['proposal_id'],
            'span': p['target'], 'relation': 'exact_repeat', 'allowed_modes': ['arrangement'],
            'confirmed': True}], 'actor': 'synthetic-test', 'reason': 'synthetic periodic input'})
        options = {'target_ids': [p['proposal_id']], 'mode': 'arrangement',
                   'source_hold': 'reject', 'target_hold': 'reject', 'locked': 'reject'}
        path = url+'/structure-reviews/'+review['review_id']
        diff = post(path+'/preview', options)
        assert diff['changes'] and not diff['conflicts']
        applied = post(path+'/apply', {'options': options,
            'preview_fingerprint': diff['fingerprint'], 'command_id': 'chain-reuse'})
        chart, current_timing = structure.manual.load(project)
        assert current_timing == timing and base.events[0] in chart.events
        optimize = preview_optimization(chart, timing, arrangement=True)
        assert not optimize.conflicts
        optimized = apply_optimization(chart, timing, arrangement=True,
                                       expected_preview_fingerprint=optimize.fingerprint)
        structure.manual.save(project, expected_revision=chart.revision_id,
            command_id='chain-optimize', chart=optimized, timing=timing)
        assert base.events[0] in optimized.events
        # A later optimization must not be overwritten by the old reuse undo.
        post(url+'/structure-operations/'+applied['operation_id']+'/undo',
             {'expected_revision': optimized.revision_id, 'command_id': 'unsafe-undo'}, 422)
        job = post(url+'/exports', {'expected_revision': optimized.revision_id}, 202)
        worker.tick()
        assert worker.backend.get_job(job['job_id']).state == 'succeeded'
        exported = api.get('/api/v1/jobs/'+job['job_id']+'/export').json()
        assert api.get('/api/v1/exports/'+exported['export_id']+'/download').status_code == 200
        backup_workspace(worker.backend, worker.store, tmp_path/'backup')
        restored = restore_workspace(tmp_path/'backup', tmp_path/'restored')
        repo = JobBackend(restored/'state.sqlite'); store = ArtifactStore(restored/'store')
        assert ManualWorkflow(repo, store).load(project) == (optimized, timing)
        assert store.get('onset_review', optimized.onset_revision)
        assert store.get('structure_operation', applied['operation_id'])
        # Rebuild from the restored database/files, not from source artifacts.
        kwargs = dict(project_id=project, expected_revision=optimized.revision_id,
            timing=timing.mapping, timing_revision=timing.revision_id, title='chain', artist='fixture')
        before = build_export(worker.backend, worker.store, worker.decoder, **kwargs)
        after = build_export(repo, store, worker.decoder, **kwargs)
        # Ogg stream serials vary across encodes; compare decoded PCM instead.
        def resources(artifact, artifact_store):
            package = artifact_store._destination('export', artifact.artifact_hash)/'chart.mdm'
            with zipfile.ZipFile(io.BytesIO(package.read_bytes())) as archive:
                return {name: archive.read(name) for name in archive.namelist()}
        original = resources(before, worker.store)
        recovered = resources(after, store)
        assert original.keys() == recovered.keys()
        for name in original:
            if name.endswith('.ogg'):
                _, left = worker.decoder.decode(original[name])
                _, right = worker.decoder.decode(recovered[name])
                assert left['audio.wav'] == right['audio.wav']
            else:
                assert original[name] == recovered[name]
        assert before.payload['logical_count'] == after.payload['logical_count'] == len(optimized.events)
    finally:
        worker.close()
