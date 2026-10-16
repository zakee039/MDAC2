"""Frozen paired evaluation reports; missing songs stay in every denominator."""
from dataclasses import dataclass
import math
import random
from mdac2.domain.artifact import ArtifactEnvelope,_hash
from mdac2.evaluation.dataset_manifest import DatasetManifest

METRICS=frozenset(('beat_f1','downbeat_f1','onset_f1','rest_preservation','coverage','load_success'))
SLICES=('fixed','variable_tempo','3/4','6/8','high_speed','pickup','long_intro','instrumental','vocal_dense')


def freeze_plan(manifest,*,candidate_hash,configuration_hash,metric_profile_hash,reference_kind='onsets',axis='timing',metrics=('onset_f1',)):
    checked=DatasetManifest.load(manifest.artifact.to_json(),expected_hash=manifest.fingerprint)
    if checked.songs!=manifest.songs:raise ValueError('Manifest object differs from frozen artifact')
    for h in (candidate_hash,configuration_hash,metric_profile_hash):_hash(h)
    if reference_kind not in ('timing','onsets','chart','structure') or axis not in ('timing','onsets') or type(metrics) is not tuple or not metrics or len(set(metrics))!=len(metrics) or not set(metrics)<=METRICS:raise ValueError('Explicit supported evaluation plan required')
    for song in manifest.songs:
        if song.role=='locked_test' and any(not any(r.kind==kind and r.gold_eligible for r in song.references) for kind in (reference_kind,axis)):
            raise ValueError('Every locked song needs qualified gold evaluation and oracle references')
    return ArtifactEnvelope.create(artifact_type='evaluation_plan',producer={'name':'mdac2.evaluation-plan','version':'1','model_hash':None,'dependencies':{}},inputs=[manifest.fingerprint],options={},
        payload={'schema_version':1,'candidate_hash':candidate_hash,'configuration_hash':configuration_hash,'metric_profile_hash':metric_profile_hash,'reference_kind':reference_kind,
            'axis':axis,'modes':['oracle_timing','auto_timing'] if axis=='timing' else ['gold_onsets','auto_onsets'],'metrics':list(metrics),
            'dataset_hash':manifest.fingerprint,'locked_only':True,'bootstrap_seed':42,'bootstrap_samples':200,'small_slice_threshold':20})


@dataclass(frozen=True,slots=True)
class EvaluationRun:
    plan_hash:str
    song_id:str
    mode:str
    reference_hash:str
    input_reference_hash:str|None
    status:str
    metrics:tuple[tuple[str,float|None],...]
    output_hash:str|None=None
    failure_code:str|None=None

    def __post_init__(self):
        _hash(self.plan_hash);_hash(self.reference_hash)
        if self.input_reference_hash is not None:_hash(self.input_reference_hash)
        if self.status not in ('succeeded','failed') or type(self.song_id) is not str or not self.song_id:raise ValueError('Explicit evaluation run identity/status required')
        if type(self.metrics) is not tuple or any(type(pair) is not tuple or len(pair)!=2 for pair in self.metrics) or len({name for name,_ in self.metrics})!=len(self.metrics):raise ValueError('Unique immutable run metrics required')
        for name,value in self.metrics:
            if name not in METRICS or value is not None and (type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1):raise ValueError('Bounded higher-is-better report metrics required')
        if self.status=='succeeded':
            _hash(self.output_hash)
            if self.failure_code is not None:raise ValueError('Successful run cannot carry failure')
        elif self.metrics or self.output_hash is not None or type(self.failure_code) is not str or not self.failure_code:raise ValueError('Failed run requires reason and no success output')


def interval(values,clusters,seed=42,count=200):
    grouped={}
    for value,cluster in zip(values,clusters):grouped.setdefault(cluster,[]).append(value)
    if len(grouped)<2:return None
    rng=random.Random(seed);groups=list(grouped.values());means=[]
    for _ in range(count):
        sample=[v for _ in groups for v in rng.choice(groups)]
        means.append(sum(sample)/len(sample))
    means.sort()
    return [means[int(.025*(count-1))],means[int(.975*(count-1))]]


def build_report(manifest,plan,runs):
    if plan.to_dict()['artifact_type']!='evaluation_plan':raise ValueError('Frozen evaluation plan required')
    p=plan.payload
    rebuilt=freeze_plan(manifest,candidate_hash=p['candidate_hash'],configuration_hash=p['configuration_hash'],metric_profile_hash=p['metric_profile_hash'],reference_kind=p['reference_kind'],axis=p['axis'],metrics=tuple(p['metrics']))
    if rebuilt.artifact_hash!=plan.artifact_hash:raise ValueError('Plan/manifest mismatch or noncanonical plan')
    songs={s.song_id:s for s in manifest.songs if s.role=='locked_test'}
    if not songs or len(songs)>2000 or any(len(s.slices)>32 for s in songs.values()):raise ValueError('Nonempty bounded locked dataset required')
    if type(runs) is not tuple or len(runs)>len(songs)*2 or any(type(r) is not EvaluationRun for r in runs):raise ValueError('Bounded immutable paired runs required')
    indexed={}
    for run in runs:
        if run.plan_hash!=plan.artifact_hash or run.song_id not in songs or run.mode not in p['modes']:raise ValueError('Run differs from frozen comparison plan')
        key=(run.song_id,run.mode)
        if key in indexed:raise ValueError('Duplicate run; choose an explicit frozen run set')
        song=songs[run.song_id];refs={r.label_hash:r for r in song.references};reference=refs.get(run.reference_hash)
        if reference is None or reference.kind!=p['reference_kind'] or not reference.gold_eligible:raise ValueError('Evaluation reference is not independently reviewed human gold')
        if run.mode==p['modes'][0]:
            oracle=refs.get(run.input_reference_hash)
            if oracle is None or oracle.kind!=p['axis'] or not oracle.gold_eligible:raise ValueError('Oracle input is not eligible human reference')
        elif run.input_reference_hash is not None:raise ValueError('Automatic input cannot masquerade as oracle')
        if not set(dict(run.metrics))<=set(p['metrics']):raise ValueError('Metric outside frozen plan')
        indexed[key]=run
    for identity in songs:
        pair=[indexed.get((identity,m)) for m in p['modes']]
        if all(pair) and pair[0].reference_hash!=pair[1].reference_hash:raise ValueError('Paired runs must use the same evaluation reference')
    rows=[]
    for identity,song in songs.items():
        for mode in p['modes']:
            run=indexed.get((identity,mode))
            rows.append({'song_id':identity,'mode':mode,'slices':list(song.slices),'status':'missing' if run is None else run.status,
                'failure_code':'NO_RUN' if run is None else run.failure_code,'metrics':{} if run is None else dict(run.metrics),
                'output_hash':None if run is None else run.output_hash,'reference_hash':None if run is None else run.reference_hash,
                'reference_uncertainty_ms':None if run is None else next(r.uncertainty_ms for r in song.references if r.label_hash==run.reference_hash)})
    summaries=[]
    for label in ('all',*sorted(set(SLICES)|{tag for s in songs.values() for tag in s.slices})):
        selected=[r for r in rows if label=='all' or label in r['slices']]
        for mode in p['modes']:
            group=[r for r in selected if r['mode']==mode];values={}
            for metric in p['metrics']:
                success=[r['metrics'][metric] for r in group if r['status']=='succeeded' and r['metrics'].get(metric) is not None]
                all_values=[r['metrics'].get(metric) if r['status']=='succeeded' and r['metrics'].get(metric) is not None else 0. for r in group]
                values[metric]={'measured_count':len(success),'missing_or_failed_count':len(group)-len(success),
                    'successful_mean':sum(success)/len(success) if success else None,'all_locked_zero_failure_mean':sum(all_values)/len(all_values) if all_values else None,
                    'all_locked_bootstrap_95':interval(all_values,[songs[r['song_id']].work_family for r in group])}
            summaries.append({'slice':label,'mode':mode,'song_count':len(group),'failed_count':sum(r['status']!='succeeded' for r in group),
                'uncertainty':'NO_SAMPLES' if not group else 'SMALL_SAMPLE' if len(group)<20 else 'BOOTSTRAP_NOT_PROOF_OF_GENERALIZATION','metrics':values})
    deltas=[]
    for metric in p['metrics']:
        differences=[];families=[]
        for identity in songs:
            a=indexed.get((identity,p['modes'][0]));b=indexed.get((identity,p['modes'][1]))
            if a and b and a.status==b.status=='succeeded' and dict(a.metrics).get(metric) is not None and dict(b.metrics).get(metric) is not None:
                differences.append(dict(b.metrics)[metric]-dict(a.metrics)[metric]);families.append(songs[identity].work_family)
        deltas.append({'metric':metric,'paired_count':len(differences),'excluded_pair_count':len(songs)-len(differences),'auto_minus_oracle_mean':sum(differences)/len(differences) if differences else None,'bootstrap_95':interval(differences,families)})
    return ArtifactEnvelope.create(artifact_type='evaluation_report',producer={'name':'mdac2.evaluation-report','version':'1','model_hash':None,'dependencies':{}},inputs=[manifest.fingerprint,plan.artifact_hash],options={'plan':plan.artifact_hash},
        payload={'schema_version':1,'rows':rows,'summaries':summaries,'paired_deltas':deltas,'pretrained_overlap_counts':{risk:sum(s.pretrained_overlap==risk for s in songs.values()) for risk in ('known_overlap','known_disjoint','unknown')},
            'bootstrap_unit':'work_family','warnings':['FAILURES_RETAINED_IN_LOCKED_DENOMINATOR','LESS_THAN_TWO_FAMILIES_HAS_NO_INTERVAL','NO_RELEASE_OR_GAME_ACCEPTANCE_INFERRED']})
