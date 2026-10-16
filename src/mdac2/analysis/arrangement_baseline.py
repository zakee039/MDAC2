"""Small deterministic load-aware lane baseline; no onset filling or shifting."""
from dataclasses import asdict,dataclass,replace
from fractions import Fraction as F
import math
from mdac2.domain.artifact import content_hash
from mdac2.domain.chart import ChartEvent,ChartRevision
from mdac2.domain.validation import validate_chart


@dataclass(frozen=True,slots=True)
class ArrangementPolicy:
    version:str='lane-load-1'
    window_seconds:float=1.
    same_lane_gap:float=.12
    change_cost:float=.15
    load_cost:float=.25

    def __post_init__(self):
        if self.version!='lane-load-1':raise ValueError('Unknown arrangement policy')
        for v in (self.window_seconds,self.same_lane_gap,self.change_cost,self.load_cost):
            if type(v) not in (float,int) or not math.isfinite(v) or not 0<v<=2:raise ValueError('Positive bounded arrangement configuration required')
        if self.same_lane_gap>self.window_seconds:raise ValueError('Gap exceeds context window')

    @property
    def fingerprint(self):return content_hash(asdict(self))


@dataclass(frozen=True,slots=True)
class ArrangementResult:
    events:tuple[ChartEvent,...]
    decisions:tuple[tuple[str,str],...]
    conflicts:tuple[tuple[str,str],...]
    policy_hash:str
    base_hash:str


def chart_from_selection(onsets,selection,timing):
    if selection.onset_fingerprint!=onsets.fingerprint or onsets.timing_fingerprint!=timing.fingerprint:raise ValueError('Stale onset selection/timing')
    ids=set(selection.selected_ids)
    if not ids<={c.candidate_id for c in onsets.candidates}:raise ValueError('Unknown selected candidate')
    events=[]
    for c in onsets.candidates:
        if c.candidate_id not in ids:continue
        if c.quantized_q is None:raise ValueError('Selected onset still requires manual quantization')
        events.append(ChartEvent(c.candidate_id,c.quantized_q,('ground',),source_kind='candidate',source_id=c.candidate_id))
    return ChartRevision(content_hash({'onsets':onsets.fingerprint,'selection':selection.fingerprint,'timing':timing.fingerprint}),timing.revision_id,tuple(events),onset_revision=selection.fingerprint)


def arrange(chart,timing,*,span=None,policy=ArrangementPolicy()):
    if type(policy) is not ArrangementPolicy or timing.review!='accepted':raise ValueError('Approved timing and typed policy required')
    if len(chart.events)>2048:raise ValueError('Arrangement context limit exceeded')
    if validate_chart(chart,timing.mapping,timing_revision=timing.revision_id,allow_empty=True).has_errors:raise ValueError('Invalid arrangement input')
    if span is not None:timing.mapping.t(span.start_q);timing.mapping.t(span.end_q)
    eligible=lambda e:(span is None or span.contains(e.q)) and not e.locked and e.source_kind!='manual' and e.semantic=='tap' and len(e.lane_mask)==1
    fixed=[e for e in chart.events if not eligible(e)]
    placed=[];changed={};decisions=[];conflicts=[]
    for event in sorted(chart.events,key=lambda e:(e.q,e.event_id)):
        if not eligible(event):decisions.append((event.event_id,'MANUAL_LOCK_SEMANTIC_OR_OUTSIDE_PRESERVED'));continue
        seconds=timing.mapping.t(event.q);context=fixed+placed;costs={}
        for lane in ('ground','air'):
            blockers=[e for e in context if lane in e.lane_mask and (e.q==event.q or e.semantic=='hold' and e.q<=event.q<e.end_q)]
            if blockers:continue
            near=[abs(float(timing.mapping.t(e.q)-seconds)) for e in context if lane in e.lane_mask]
            load=sum(max(0,1-d/policy.window_seconds) for d in near)
            rapid=sum(d<policy.same_lane_gap for d in near)
            costs[lane]=load*policy.load_cost+rapid+ (0 if lane in event.lane_mask else policy.change_cost)
        if not costs:
            conflicts.append((event.event_id,'BOTH_LANES_OCCUPIED'));placed.append(event);continue
        lane=min(costs,key=lambda lane:(costs[lane],lane!=event.lane_mask[0],lane))
        new=replace(event,lane_mask=(lane,));changed[event.event_id]=new;placed.append(new)
        decisions.append((event.event_id,'LOCAL_LOAD_AND_BOUNDARY_CONTEXT' if new!=event else 'EXISTING_LANE_WITHIN_BUDGET'))
    return ArrangementResult(tuple(changed.get(e.event_id,e) for e in chart.events),tuple(decisions),tuple(conflicts),policy.fingerprint,content_hash(chart.to_dict()))
