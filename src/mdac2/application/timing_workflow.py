"""Immutable analysis candidates, explicit review and CAS timing adoption."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from fractions import Fraction
from uuid import uuid4

from mdac2.analysis.timing_fit import fit_timing
from mdac2.application.manual_workflow import timing_data, timing_from
from mdac2.application.timing_migration import preview_migration
from mdac2.domain.artifact import ArtifactEnvelope, canonical_json
from mdac2.domain.chart import ChartRevision
from mdac2.domain.review import TimingQuality, QualityItem, review_timing
from mdac2.domain.timing_edits import TimingRevision
from mdac2.infrastructure.project_repository import RevisionConflict
from mdac2.policies.timing import assess_fit
from mdac2.ports.timing import TimingObservation


def quality_data(quality):
    value=asdict(quality)
    value['items']=[{**asdict(i),'reasons':list(i.reasons)} for i in quality.items]
    return value


def quality_from(value):
    return TimingQuality(value['revision_fingerprint'],value['observation_hash'],value['policy_hash'],
        tuple(QualityItem(**{**i,'reasons':tuple(i['reasons'])}) for i in value['items']))


class TimingWorkflow:
    def __init__(self,manual):
        self.manual=manual
        self.repository,self.store=manual.repository,manual.store

    def context(self,project_id,expected_revision):
        project=self.repository.get_project(project_id)
        if project.head_revision!=expected_revision:raise RevisionConflict('Stale analysis base')
        with self.repository._connection() as c:
            row=c.execute('SELECT artifact_hash FROM project_audio WHERE project_id=?',(project_id,)).fetchone()
        if row is None:raise ValueError('Import audio before analysis')
        audio=self.store.get('audio',row[0])
        inputs=[audio.artifact_hash]
        if expected_revision is not None:inputs.append(self.repository.get_revision(project_id,expected_revision).artifact_hash)
        return audio,inputs

    def propose(self,project_id,expected_revision,observation,*,pulse_unit=Fraction(1,4),first_q=0,meter=(4,4),producer=None,options=None):
        audio,inputs=self.context(project_id,expected_revision)
        if not isinstance(observation,TimingObservation):raise ValueError('Observation required')
        source=observation.payload['audio'];canonical=audio.payload['canonical']
        if (source['audio_hash'],source['sample_rate'],source['sample_count'],source['channels'])!=(audio.artifact_hash,canonical['sample_rate'],canonical['sample_count'],len(canonical['channel_layout'])):
            raise ValueError('Observation does not describe project audio')
        fit=fit_timing(observation,pulse_unit=pulse_unit,first_q=first_q,meter=meter)
        timing=quality=None
        if fit.accepted:
            previous=self.manual.load(project_id,expected_revision) if expected_revision else None
            # Raw downbeats remain observations. Never guess their bar alignment.
            timing=TimingRevision(str(uuid4()),fit.mapping,parent_fingerprint=previous[1].fingerprint if previous else None)
            quality=assess_fit(timing,fit)
        payload={'schema_version':1,'project_id':project_id,'base_revision':expected_revision,
            'observation':observation.artifact.to_dict(),'timing':timing_data(timing) if timing else None,
            'quality':quality_data(quality) if quality else None,
            'fit':{'accepted':fit.accepted,'reason':fit.reason,'model_type':fit.model_type,
                'p95_seconds':fit.p95_seconds,'max_seconds':fit.max_seconds,'coverage_seconds':list(fit.coverage_seconds),
                'warnings':list(fit.warnings),'residual_seconds':list(fit.residual_seconds)}}
        envelope=ArtifactEnvelope.create(artifact_type='timing_candidate',
            producer=producer or observation.artifact.to_dict()['producer'],inputs=inputs,
            options=options or {'project_id':project_id,'base_revision':expected_revision,'pulse_unit':str(pulse_unit),
                'first_q':str(first_q),'meter':list(meter),'fit_policy':fit.policy_hash},payload=payload)
        self.context(project_id,expected_revision)
        self.store.publish(envelope)
        return envelope

    def load(self,project_id,candidate_id,*,current=True):
        envelope=self.store.get('timing_candidate',candidate_id);data=envelope.payload
        if data['project_id']!=project_id:raise ValueError('Candidate belongs to another project')
        if current:self.context(project_id,data['base_revision'])
        return envelope

    def preview(self,project_id,candidate_id,position_policy):
        data=self.load(project_id,candidate_id).payload
        if data['timing'] is None:raise ValueError('Rejected fit has no adoptable timing')
        timing=timing_from(data['timing'])
        previous=self.manual.load(project_id,data['base_revision']) if data['base_revision'] else None
        if previous and previous[0].events:
            preview=preview_migration(*previous,timing,position_policy=position_policy)
            return {'fingerprint':preview.fingerprint,'diffs':[d.to_dict() for d in preview.diffs],
                'conflicts':[asdict(c) for c in preview.conflicts]}
        if position_policy not in ('keep_audio_time','keep_musical_position'):raise ValueError('Explicit position policy required')
        return {'fingerprint':None,'diffs':[],'conflicts':[]}

    def adopt(self,project_id,candidate_id,*,position_policy,preview_fingerprint,actor,reason,command_id):
        envelope=self.load(project_id,candidate_id);data=envelope.payload
        preview=self.preview(project_id,candidate_id,position_policy)
        if preview['fingerprint']!=preview_fingerprint or preview['conflicts']:raise ValueError('Review a conflict-free current migration')
        base=timing_from(data['timing']);quality=quality_from(data['quality'])
        timing,decision=review_timing(base,quality,expected_fingerprint=base.fingerprint,decision='accepted',
            actor=actor,reason=reason,timestamp=datetime.now(timezone.utc).isoformat())
        previous=self.manual.load(project_id,data['base_revision']) if data['base_revision'] else None
        # Review changes the fingerprint: recompute internal migration against the
        # reviewed revision after checking the user saw the pending revision diff.
        internal=None
        if previous:
            chart=replace(previous[0],revision_id=str(uuid4()),timing_revision=timing.revision_id)
            if chart.events:
                migration=preview_migration(*previous,timing,position_policy=position_policy)
                internal=migration.fingerprint
                chart=replace(chart,events=migration.proposed_events)
        else:chart=ChartRevision(str(uuid4()),timing.revision_id,())
        review=ArtifactEnvelope.create(artifact_type='timing_review',producer={'name':'human-review','version':'1','model_hash':None,'dependencies':{}},
            options={},inputs=[envelope.artifact_hash],payload=asdict(decision))
        self.store.publish(review)
        return self.manual.save(project_id,expected_revision=data['base_revision'],command_id=command_id,chart=chart,timing=timing,
            position_policy=position_policy,preview_fingerprint=internal,reviewed=(quality,decision),evidence=(review,))
