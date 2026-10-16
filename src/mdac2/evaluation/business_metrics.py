"""Business evidence distinct from standard accuracy and game acceptance."""
from dataclasses import dataclass
from mdac2.domain.artifact import ArtifactEnvelope,content_hash,_hash
from mdac2.domain.validation import validate_chart
from mdac2.evaluation.standard_metrics import event_scores


@dataclass(frozen=True,slots=True)
class VerifiedForbiddenAssessment:
    chart_hash:str
    profile_hash:str
    game_version:str
    verification_record_hash:str
    violating_event_ids:tuple[str,...]

    def __post_init__(self):
        for h in (self.chart_hash,self.profile_hash,self.verification_record_hash):_hash(h)
        if type(self.game_version) is not str or not self.game_version.strip() or type(self.violating_event_ids) is not tuple or len(set(self.violating_event_ids))!=len(self.violating_event_ids):raise ValueError('Explicit verified game assessment required')


def edit_counts(before,after):
    a={e.event_id:e for e in before.events};b={e.event_id:e for e in after.events};shared=a.keys()&b.keys()
    fields={'moved':'q','lanes_changed':'lane_mask','semantic_changed':'semantic','duration_changed':'duration_q','accent_changed':'accent'}
    counts={name:sum(getattr(a[i],field)!=getattr(b[i],field) for i in shared) for name,field in fields.items()}
    modified=sum(a[i]!=b[i] for i in shared)
    return {**counts,'added':len(b.keys()-a.keys()),'deleted':len(a.keys()-b.keys()),'modified_events':modified,
        'locked_modified_or_deleted':sum(e.locked and b.get(i)!=e for i,e in a.items()),
        'edit_fraction_of_base':(modified+len(a.keys()-b.keys())+len(b.keys()-a.keys()))/len(a) if a else None}


def business_metrics(chart,timing,*,rests=(),references=(),baseline=None,forbidden=None):
    if chart.timing_revision!=timing.revision_id or len(chart.events)>10000:raise ValueError('Matching bounded chart/timing required')
    base_hash=content_hash(chart.to_dict());report=validate_chart(chart,timing.mapping,timing_revision=timing.revision_id,allow_empty=True)
    coordinates_valid=not any(i.code=='TIME_OUT_OF_RANGE' for i in report.issues)
    if len({r.revision_id for r in references})!=len(references):raise ValueError('Unique reference revisions required')
    if any(a.overlaps(b) for i,a in enumerate(rests) for b in rests[i+1:]):raise ValueError('Reference rest intervals must not overlap')
    rest_rows=[]
    for span in rests:
        start,end=timing.mapping.t(span.start_q),timing.mapping.t(span.end_q)
        violating=[e.event_id for e in chart.events if span.contains(e.q) or e.semantic=='hold' and e.q<span.end_q and e.end_q>span.start_q]
        rest_rows.append({'span':span.to_dict(),'seconds':float(end-start),'preserved':not violating,'violating_events':violating})
    peaks={'logical':None,'physical':None}
    if coordinates_valid:
        ordered=sorted(chart.events,key=lambda e:(e.q,e.event_id));left=0;physical=0;peaks={'logical':0,'physical':0}
        for i,e in enumerate(ordered):
            physical+=len(e.lane_mask)
            while timing.mapping.t(e.q)-timing.mapping.t(ordered[left].q)>=1:
                physical-=len(ordered[left].lane_mask);left+=1
            peaks['logical']=max(peaks['logical'],i-left+1);peaks['physical']=max(peaks['physical'],physical)
    comparisons=[]
    for reference in references:
        if reference.timing_revision!=timing.revision_id or validate_chart(reference,timing.mapping,timing_revision=timing.revision_id,allow_empty=True).has_errors:raise ValueError('Comparable valid reference timing required')
        # This metric is onset-time coverage, not event count or lane accuracy.
        reference_times=sorted({float(timing.mapping.t(e.q)) for e in reference.events})
        estimated_times=sorted({float(timing.mapping.t(e.q)) for e in chart.events}) if coordinates_valid else None
        score=event_scores(reference_times,estimated_times,kind='onset')
        comparisons.append({'reference_id':reference.revision_id,'reference_hash':content_hash(reference.to_dict()),'onset_coverage':score.payload,
            'comparison_meaning':'one_valid_creative_reference_not_unique_truth'})
    forbidden_value={'status':'unknown_no_verified_game_profile','rate':None,'count':None}
    if forbidden is not None:
        if type(forbidden) is not VerifiedForbiddenAssessment or forbidden.chart_hash!=base_hash or not set(forbidden.violating_event_ids)<={e.event_id for e in chart.events}:raise ValueError('Stale or invalid game assessment')
        forbidden_value={'status':'verified_profile_assessment','rate':len(forbidden.violating_event_ids)/len(chart.events) if chart.events else None,
            'count':len(forbidden.violating_event_ids),'profile_hash':forbidden.profile_hash,'game_version':forbidden.game_version,'verification_record_hash':forbidden.verification_record_hash}
    return ArtifactEnvelope.create(artifact_type='business_metrics',producer={'name':'mdac2.business-metrics','version':'1','model_hash':None,'dependencies':{}},inputs=[],
        options={'chart':base_hash,'timing':timing.fingerprint,'rests':[s.to_dict() for s in rests],'references':[content_hash(r.to_dict()) for r in references],
            'baseline':None if baseline is None else content_hash(baseline.to_dict()),'forbidden':forbidden_value,'density_window_seconds':1,'onset_preprocessing':'unique_time_only_for_onset_coverage'},
        payload={'schema_version':1,'chart_hash':base_hash,'logical_events':len(chart.events),'physical_entities':sum(len(e.lane_mask) for e in chart.events),
            'hard_errors':[{'code':i.code,'event_id':i.event_id} for i in report.issues if i.severity=='error'],
            'rest_preservation':sum(r['preserved'] for r in rest_rows)/len(rest_rows) if rest_rows else None,'rest_regions':rest_rows,
            'peak_nps':peaks,'forbidden':forbidden_value,'reference_comparisons':comparisons,'edits':None if baseline is None else edit_counts(baseline,chart),
            'warnings':['MULTIPLE_REFERENCES_ARE_NOT_A_UNIQUE_TARGET','NO_GAME_LOAD_SUCCESS_INFERRED']})
