"""Persistent mixed-audio proposals, explicit relation review and reversible reuse."""
from dataclasses import asdict, replace
from fractions import Fraction as F
from importlib.metadata import version
from uuid import uuid4
import wave

from mdac2.analysis.similarity_features import extract_features, FeaturePolicy
from mdac2.analysis.similarity_search import search, SearchPolicy
from mdac2.analysis.phrase_alignment import align, AlignmentResult, AlignmentPolicy
from mdac2.application.onset_workflow import OnsetWorkflow
from mdac2.application.arrangement_reuse import preview_reuse, apply_reuse
from mdac2.application.full_pattern_reuse import preview_full_reuse, apply_full_reuse
from mdac2.domain.artifact import ArtifactEnvelope, canonical_json, content_hash
from mdac2.domain.chart import ChartRevision
from mdac2.domain.structure import Span, SimilarityProposal, ConfirmedTarget, ConfirmedStructureGroup
from mdac2.ports.timing import TimingInput


def producer():
    deps = {name: version(name) for name in ('librosa', 'numpy', 'scipy', 'numba')}
    if deps != {'librosa':'0.11.0','numpy':'2.2.6','scipy':'1.15.3','numba':'0.61.2'}:
        raise ValueError('Verified similarity analysis environment required')
    return {'name':'mdac2.structure-workflow','version':'1','model_hash':None,'dependencies':deps}


def request_options(project_id, revision):
    return {'project_id':project_id,'base_revision':revision,'mode':'mix_only',
        'features':asdict(FeaturePolicy()),'search':SearchPolicy().to_dict(),'alignment':asdict(AlignmentPolicy())}


def envelope(data): return ArtifactEnvelope.from_json(canonical_json(data))


def span(data):
    if type(data) is not dict or set(data) != {'start_q','end_q'}: raise ValueError('Explicit span bounds required')
    values = []
    for key in ('start_q','end_q'):
        value = data[key]
        if type(value) is not str or len(value) > 64: raise ValueError('Bounded exact span coordinates required')
        try: q = F(value)
        except (ValueError, ZeroDivisionError) as exc: raise ValueError('Invalid span coordinate') from exc
        if str(q) != value: raise ValueError('Canonical span coordinate required')
        values.append(q)
    return Span(*values)


def proposal(data):
    return SimilarityProposal(data['proposal_id'],span(data['source']),span(data['target']),data['audio_hash'],
        data['timing_fingerprint'],data['feature_hash'],tuple(tuple(v) for v in data['similarity_by_source']),
        data['boundary_uncertainty_seconds'],tuple((F(a),F(b)) for a,b in data['alignment_anchors']),tuple(data['warnings']))


def group_data(group):
    return {'group_id':group.group_id,'primary':group.primary.to_dict(),'timing_fingerprint':group.timing_fingerprint,
        'actor':group.actor,'reason':group.reason,'review':group.review,'targets':[
            {'target_id':t.target_id,'span':t.span.to_dict(),'proposal_fingerprint':t.proposal_fingerprint,
             'relation':t.relation,'allowed_modes':list(t.allowed_modes)} for t in group.targets]}


def group_from(data):
    return ConfirmedStructureGroup(data['group_id'],span(data['primary']),tuple(ConfirmedTarget(t['target_id'],span(t['span']),
        t['proposal_fingerprint'],t['relation'],tuple(t['allowed_modes'])) for t in data['targets']),
        data['timing_fingerprint'],data['actor'],data['reason'],data['review'])


class StructureWorkflow:
    def __init__(self, manual):
        self.manual = manual
        self.repository, self.store = manual.repository, manual.store

    def context(self, project_id, revision): return OnsetWorkflow(self.manual).context(project_id, revision)

    def propose(self, project_id, revision):
        audio, inputs, chart, timing = self.context(project_id, revision)
        identity = producer(); options = request_options(project_id, revision)
        c = audio.payload['canonical']
        with wave.open(str(self.store._destination('audio', audio.artifact_hash)/'audio.wav'),'rb') as wav:
            pcm = wav.readframes(wav.getnframes())
        features = extract_features(TimingInput(audio.artifact_hash,c['sample_rate'],c['sample_count'],len(c['channel_layout'])),pcm,timing)
        result = search(features)
        alignments = {p.proposal_id:align(features,p).artifact.to_dict() for p in result.proposals}
        # Embedded immutable envelopes travel with the result and backup closure;
        # the job's direct inputs remain exactly the enqueued audio/base revision.
        artifact = ArtifactEnvelope.create(artifact_type='structure_candidate',producer=identity,options=options,inputs=inputs,
            payload={'schema_version':1,'project_id':project_id,'base_revision':revision,'base_chart_hash':content_hash(chart.to_dict()),
                'audio_hash':audio.artifact_hash,'timing_fingerprint':timing.fingerprint,'features':features.to_dict(),
                'search':result.artifact.to_dict(),'alignments':alignments,'warnings':['MIX_SPECTRAL_BASELINE_REQUIRES_HUMAN_REVIEW']})
        self.context(project_id,revision); self.store.publish(artifact)
        return artifact

    def load(self, project_id, candidate_id, *, current=True):
        artifact = self.store.get('structure_candidate',candidate_id); data = artifact.payload
        if data['project_id'] != project_id: raise ValueError('Structure candidate belongs to another project')
        if current:
            audio, _, chart, timing = self.context(project_id,data['base_revision'])
            if (audio.artifact_hash,content_hash(chart.to_dict()),timing.fingerprint) != (
                    data['audio_hash'],data['base_chart_hash'],data['timing_fingerprint']): raise ValueError('Stale structure inputs')
        features = envelope(data['features']); proposals = envelope(data['search'])
        if features.payload['audio_hash'] != data['audio_hash'] or features.payload['timing_fingerprint'] != data['timing_fingerprint']:
            raise ValueError('Embedded feature input mismatch')
        if proposals.to_dict()['inputs'] != [features.artifact_hash]: raise ValueError('Embedded search source mismatch')
        for raw in proposals.payload['proposals']:
            p = proposal(raw)
            a = envelope(data['alignments'][p.proposal_id])
            if p.feature_hash != features.artifact_hash or a.payload['proposal_fingerprint'] != p.fingerprint:
                raise ValueError('Embedded alignment source mismatch')
        return artifact

    def confirm(self, project_id, candidate_id, *, primary, targets, actor, reason):
        if any(type(v) is not str or not v.strip() or len(v)>2000 for v in (actor,reason)):
            raise ValueError('Explicit bounded review actor/reason required')
        if type(targets) is not list or not 1<=len(targets)<=20: raise ValueError('One to twenty explicit reviewed targets required')
        candidate = self.load(project_id,candidate_id); data = candidate.payload
        _, _, _, timing = self.context(project_id,data['base_revision'])
        features = envelope(data['features'])
        proposals = {p['proposal_id']:proposal(p) for p in envelope(data['search']).payload['proposals']}
        source = span(primary); timing.mapping.t(source.start_q); timing.mapping.t(source.end_q)
        confirmed=[]; alignments={}; errors={}; provenance=[]; original_source=None
        for target in targets:
            if type(target) is not dict or set(target) != {'proposal_id','span','relation','allowed_modes','confirmed'} or target['confirmed'] is not True:
                raise ValueError('Each target needs explicit confirmation and permissions')
            if type(target['proposal_id']) is not str or target['proposal_id'] not in proposals: raise ValueError('Unknown structure proposal')
            p = proposals[target['proposal_id']]
            if original_source is not None and original_source != p.source: raise ValueError('Group proposals must share the same original primary')
            original_source = p.source
            destination = span(target['span']); timing.mapping.t(destination.start_q); timing.mapping.t(destination.end_q)
            edited = replace(p,source=source,target=destination,alignment_anchors=())
            if type(target['allowed_modes']) is not list: raise ValueError('Explicit mode list required')
            confirmed.append(ConfirmedTarget(p.proposal_id,destination,edited.fingerprint,target['relation'],tuple(target['allowed_modes'])))
            provenance.append({'proposal_id':p.proposal_id,'original_fingerprint':p.fingerprint,'reviewed_fingerprint':edited.fingerprint})
            try:
                aligned = align(features,edited)
                alignments[p.proposal_id] = aligned.artifact.to_dict()
                if not aligned.accepted: errors[p.proposal_id] = aligned.artifact.payload['reasons']
            except ValueError:
                # A user annotation can be stored, but no invented/linear mapping
                # is offered for boundaries unsupported by bar-level features.
                errors[p.proposal_id] = ['REVIEWED_BOUNDARIES_NOT_ALIGNABLE_WITH_CURRENT_FEATURES']
        group = ConfirmedStructureGroup(str(uuid4()),source,tuple(confirmed),timing.fingerprint,actor,reason)
        artifact = ArtifactEnvelope.create(artifact_type='structure_review',producer=producer(),options={},inputs=[candidate.artifact_hash],
            payload={'schema_version':1,'project_id':project_id,'base_revision':data['base_revision'],'candidate_id':candidate_id,
                'group':group_data(group),'provenance':provenance,'alignments':alignments,'alignment_errors':errors})
        self.context(project_id,data['base_revision']); self.store.publish(artifact)
        return artifact

    def reviewed(self, project_id, review_id):
        artifact = self.store.get('structure_review',review_id); data=artifact.payload
        if data['project_id'] != project_id: raise ValueError('Structure review belongs to another project')
        self.load(project_id,data['candidate_id'])
        _, _, chart, timing = self.context(project_id,data['base_revision'])
        group = group_from(data['group']); alignments={}
        for key,raw in data['alignments'].items():
            a = envelope(raw)
            alignments[key] = AlignmentResult(tuple((F(x),F(y)) for x,y in a.payload['anchors']),a)
        return artifact, chart, timing, group, alignments

    def _preview(self, project_id, review_id, *, target_ids, mode, source_hold='reject', target_hold='reject', locked='reject'):
        artifact, chart, timing, group, alignments = self.reviewed(project_id,review_id)
        if type(target_ids) is not tuple or any(type(i) is not str for i in target_ids): raise ValueError('Explicit target IDs required')
        if any(i not in alignments or not alignments[i].accepted for i in target_ids): raise ValueError('Selected target alignment rejected; annotation only')
        options={'target_ids':target_ids,'alignments':alignments}
        if mode=='full_pattern':
            options.update(source_hold=source_hold,target_hold=target_hold,locked=locked)
            result=preview_full_reuse(chart,timing,group,**options); reuse=result.reuse
        else:
            if (source_hold,target_hold,locked)!=('reject','reject','reject'): raise ValueError('Boundary choices apply only to full replacement')
            options['mode']=mode; result=preview_reuse(chart,timing,group,**options); reuse=result
        before={e.event_id:e for e in chart.events}; after={e.event_id:e for e in reuse.proposed_events}
        diff=[{'event_id':key,'before':before[key].to_dict() if key in before else None,'after':after[key].to_dict() if key in after else None}
              for key in sorted(before.keys()|after.keys()) if before.get(key)!=after.get(key)]
        public={'review_id':review_id,'base_revision':chart.revision_id,'fingerprint':result.fingerprint,'mode':mode,
            'target_ids':list(target_ids),'source_hold':source_hold,'target_hold':target_hold,'locked':locked,
            'changes':diff,'decisions':[list(v) for v in reuse.decisions],'conflicts':[list(v) for v in reuse.conflicts],
            'before_issues':[list(v) for v in reuse.before_issues],'after_issues':[list(v) for v in reuse.after_issues]}
        return public,artifact,chart,timing,group,options

    def preview(self, project_id, review_id, **options): return self._preview(project_id,review_id,**options)[0]

    def apply(self, project_id, review_id, *, preview_fingerprint, command_id, **options):
        p,review,base,timing,group,actual = self._preview(project_id,review_id,**options)
        if p['fingerprint']!=preview_fingerprint: raise ValueError('Stale structure preview')
        fn=apply_full_reuse if p['mode']=='full_pattern' else apply_reuse
        chart=fn(base,timing,group,expected_preview_fingerprint=preview_fingerprint,**actual)
        operation=ArtifactEnvelope.create(artifact_type='structure_operation',producer=producer(),options={},inputs=[review.artifact_hash],
            payload={'project_id':project_id,'base_revision':base.revision_id,'timing_fingerprint':timing.fingerprint,
                'base_chart':base.to_dict(),'applied_chart':chart.to_dict(),'preview':p})
        self.store.publish(operation)
        saved=self.manual.save(project_id,expected_revision=base.revision_id,command_id=command_id,chart=chart,timing=timing,evidence=(operation,))
        return *saved,operation.artifact_hash

    def undo(self, project_id, operation_id, *, expected_revision, command_id):
        operation=self.store.get('structure_operation',operation_id); data=operation.payload
        if data['project_id']!=project_id: raise ValueError('Structure operation belongs to another project')
        _,_,chart,timing=self.context(project_id,expected_revision)
        if chart.to_dict()!=data['applied_chart'] or timing.fingerprint!=data['timing_fingerprint']:
            raise ValueError('Undo would overwrite later edits')
        base=ChartRevision.from_json(canonical_json(data['base_chart']))
        restored=replace(base,revision_id=str(uuid4()),user_patch_ids=(*chart.user_patch_ids,content_hash({'undo':operation_id,'current':chart.revision_id})))
        undo=ArtifactEnvelope.create(artifact_type='structure_undo',producer=producer(),options={},inputs=[operation.artifact_hash],
            payload={'project_id':project_id,'undone_revision':chart.revision_id,'restored_chart':restored.to_dict()})
        self.store.publish(undo)
        return self.manual.save(project_id,expected_revision=expected_revision,command_id=command_id,chart=restored,timing=timing,evidence=(undo,))
