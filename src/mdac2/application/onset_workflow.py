"""Persisted mix candidates and explicit, protected regeneration against a CAS base."""
from dataclasses import asdict, replace
from fractions import Fraction
from uuid import uuid4
import wave

from mdac2.analysis.onset_candidates import CandidatePolicy, detect_candidates
from mdac2.application.timing_workflow import TimingWorkflow
from mdac2.domain.artifact import ArtifactEnvelope, content_hash
from mdac2.domain.chart import ChartEvent
from mdac2.domain.onset_policy import OnsetPolicy
from mdac2.domain.validation import validate_chart
from mdac2.policies.onset import select_onsets
from mdac2.ports.timing import TimingInput

PRODUCER = {'name': 'mdac2.mix-onsets', 'version': '1', 'model_hash': None, 'dependencies': {}}


def request_options(project_id, expected_revision, difficulty, candidate_policy=CandidatePolicy()):
    if type(difficulty) is not str: raise ValueError('Explicit difficulty name required')
    policy = OnsetPolicy.preset(difficulty)
    return {'project_id': project_id, 'base_revision': expected_revision, 'mode': 'mix_only',
        'candidate_policy': {**asdict(candidate_policy), 'grid_quarters': str(candidate_policy.grid_quarters)},
        'selection_policy': {**asdict(policy), 'allowed_denominators': list(policy.allowed_denominators)}}


class OnsetWorkflow:
    def __init__(self, manual):
        self.manual = manual
        self.repository, self.store = manual.repository, manual.store

    def context(self, project_id, expected_revision):
        audio, inputs = TimingWorkflow(self.manual).context(project_id, expected_revision)
        if expected_revision is None: raise ValueError('Save and review timing before onset analysis')
        chart, timing = self.manual.load(project_id, expected_revision)
        if timing.review != 'accepted': raise ValueError('Approved timing required for onset analysis')
        if len(chart.events) > 2048: raise ValueError('Onset workflow event budget exceeded')
        return audio, inputs, chart, timing

    def propose(self, project_id, expected_revision, *, difficulty='normal', candidate_policy=CandidatePolicy()):
        audio, inputs, chart, timing = self.context(project_id, expected_revision)
        policy = OnsetPolicy.preset(difficulty)
        c = audio.payload['canonical']
        with wave.open(str(self.store._destination('audio', audio.artifact_hash) / 'audio.wav'), 'rb') as wav:
            pcm = wav.readframes(wav.getnframes())
        source = TimingInput(audio.artifact_hash, c['sample_rate'], c['sample_count'], len(c['channel_layout']))
        onsets = detect_candidates(source, pcm, timing, policy=candidate_policy)
        selection = select_onsets(onsets, policy=policy)
        options = request_options(project_id, expected_revision, difficulty, candidate_policy)
        artifact = ArtifactEnvelope.create(artifact_type='onset_candidate', producer=PRODUCER, inputs=inputs, options=options,
            payload={'schema_version': 1, 'project_id': project_id, 'base_revision': expected_revision,
                'base_chart_hash': content_hash(chart.to_dict()), 'timing_fingerprint': timing.fingerprint,
                'onsets': onsets.to_dict(), 'selection_hash': selection.fingerprint,
                'decisions': [{**asdict(d), 'reasons': list(d.reasons)} for d in selection.decisions],
                'selected_ids': list(selection.selected_ids), 'warnings': ['MIX_ONLY_RULE_BASELINE_NOT_MUSIC_QUALITY_ACCEPTANCE']})
        self.context(project_id, expected_revision)
        self.store.publish(artifact)
        return artifact

    def load(self, project_id, candidate_id, *, current=True):
        artifact = self.store.get('onset_candidate', candidate_id)
        data = artifact.payload
        if data['project_id'] != project_id: raise ValueError('Candidate belongs to another project')
        if current:
            audio, _, chart, timing = self.context(project_id, data['base_revision'])
            if (audio.artifact_hash, content_hash(chart.to_dict()), timing.fingerprint) != (
                    data['onsets']['audio_hash'], data['base_chart_hash'], data['timing_fingerprint']):
                raise ValueError('Stale onset candidate inputs')
        return artifact

    def _previous(self, project_id, chart, timing):
        if chart.onset_revision is None: return {}, set(), {}
        # Foreign onset lineages cannot authorize destructive regeneration.
        try: prior = self.store.get('onset_review', chart.onset_revision).payload
        except FileNotFoundError: return {}, set(), {}
        if prior['project_id'] != project_id: raise ValueError('Onset lineage belongs to another project')
        if prior['timing_fingerprint'] != timing.fingerprint: return {}, set(), {}
        return ({e['event_id']: ChartEvent.from_dict(e) for e in prior['owned_events']}, set(prior['suppressed_q']),
                {e['event_id']: ChartEvent.from_dict(e) for e in prior['tracked_events']})

    def preview(self, project_id, candidate_id, *, selected_ids=None):
        artifact = self.load(project_id, candidate_id); data = artifact.payload
        _, _, chart, timing = self.context(project_id, data['base_revision'])
        ids = tuple(data['selected_ids']) if selected_ids is None else selected_ids
        if type(ids) is not tuple or len(ids) > 2048 or any(type(i) is not str for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('Explicit unique bounded selected candidate IDs required')
        candidates = {c['candidate_id']: c for c in data['onsets']['candidates']}
        if not set(ids) <= candidates.keys(): raise ValueError('Unknown selected candidate')
        old_owned, suppressed, tracked = self._previous(project_id, chart, timing)
        current = {e.event_id: e for e in chart.events}
        # Deletions made in the editor persist through later regeneration.
        suppressed.update(str(e.q) for key, e in tracked.items() if key not in current or current[key].q != e.q)
        preserved = [e for e in chart.events if e.locked or e.source_kind != 'candidate' or old_owned.get(e.event_id) != e]
        preserved_ids = {e.event_id for e in preserved}
        generated = []; reasons = []
        for key in sorted(ids, key=lambda i: (Fraction(candidates[i]['seconds']), i)):
            candidate = candidates[key]
            if candidate['quantized_q'] is None: raise ValueError('Selected onset requires manual quantization')
            q = Fraction(candidate['quantized_q'])
            if key in preserved_ids:
                reasons.append([key, 'EXISTING_EDIT_OR_LOCK_PRESERVED']); continue
            if str(q) in suppressed:
                reasons.append([key, 'MANUAL_DELETION_PRESERVED']); continue
            # Initial generation never inserts a competing point under a manual
            # event or Hold. This is protection, not inferred game legality.
            if any(e.q == q or e.semantic == 'hold' and e.q <= q < e.end_q for e in preserved):
                reasons.append([key, 'PROTECTED_POSITION_PRESERVED']); continue
            if any(e.q == q for e in generated):
                raise ValueError('Selected candidates collide on the same quantized point')
            generated.append(ChartEvent(key, q, ('ground',), source_kind='candidate', source_id=key))
        events = tuple(sorted((*preserved, *generated), key=lambda e: (e.q, e.event_id)))
        if len(events) > 2048: raise ValueError('Onset workflow event budget exceeded')
        proposed = replace(chart, events=events)
        report = validate_chart(proposed, timing.mapping, timing_revision=timing.revision_id, allow_empty=True)
        if report.has_errors: raise ValueError('Invalid onset preview')
        after = {e.event_id: e for e in events}
        changes = [{'event_id': key, 'before': current[key].to_dict() if key in current else None,
                    'after': after[key].to_dict() if key in after else None}
                   for key in sorted(current.keys() | after.keys()) if current.get(key) != after.get(key)]
        result = {'candidate_id': candidate_id, 'base_revision': data['base_revision'],
            'base_chart_hash': data['base_chart_hash'], 'timing_fingerprint': timing.fingerprint,
            'selected_ids': sorted(ids), 'events': [e.to_dict() for e in events],
            'owned_events': [e.to_dict() for e in generated], 'suppressed_q': sorted(suppressed, key=Fraction),
            'tracked_events': [e.to_dict() for e in events if e.source_kind == 'candidate'],
            'changes': changes, 'preserved_ids': sorted(preserved_ids), 'reasons': reasons,
            'warnings': [i.code for i in report.issues]}
        return {**result, 'fingerprint': content_hash(result)}

    def adopt(self, project_id, candidate_id, *, selected_ids, preview_fingerprint, actor, reason, command_id):
        if any(type(v) is not str or not v.strip() or len(v) > 2000 for v in (actor, reason)):
            raise ValueError('Explicit actor and review reason required')
        preview = self.preview(project_id, candidate_id, selected_ids=selected_ids)
        if preview['fingerprint'] != preview_fingerprint: raise ValueError('Review the current onset preview before adoption')
        candidate = self.load(project_id, candidate_id)
        _, _, chart, timing = self.context(project_id, preview['base_revision'])
        review = ArtifactEnvelope.create(artifact_type='onset_review', producer=PRODUCER, options={}, inputs=[candidate.artifact_hash],
            payload={'project_id': project_id, 'actor': actor, 'reason': reason, **preview})
        self.store.publish(review)
        chart = replace(chart, revision_id=str(uuid4()), events=tuple(ChartEvent.from_dict(e) for e in preview['events']),
            onset_revision=review.artifact_hash, arrangement_policy_hash=None, semantics_policy_hash=None,
            user_patch_ids=(*chart.user_patch_ids, review.artifact_hash))
        return self.manual.save(project_id, expected_revision=preview['base_revision'], command_id=command_id,
            chart=chart, timing=timing, evidence=(review,))
