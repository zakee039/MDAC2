"""Bounded robust fixed/two-segment fitting, with explicit residual rejection."""
from dataclasses import dataclass,asdict
from fractions import Fraction
import math
from statistics import median
from mdac2.domain.artifact import content_hash
from mdac2.domain.timing import Anchor,MeterSegment,TimingMap,rational
from mdac2.ports.timing import TimingObservation


@dataclass(frozen=True,slots=True)
class FitPolicy:
    version:str='1'
    tolerance_seconds:float=.035
    hard_limit_seconds:float=.08
    complexity_penalty:float=.0002
    min_segment_beats:int=6
    max_segments:int=2
    max_observations:int=4096
    max_boundaries:int=128
    seconds_decimals:int=9

    def __post_init__(self):
        if self.version!='1':raise ValueError('Unknown fit policy')
        for v in (self.tolerance_seconds,self.hard_limit_seconds,self.complexity_penalty):
            if type(v) not in (int,float) or not math.isfinite(v) or v<=0:raise ValueError('Positive finite fit budgets required')
        if self.hard_limit_seconds<self.tolerance_seconds:raise ValueError('Hard limit below tolerance')
        for v in (self.min_segment_beats,self.max_segments,self.max_observations,self.max_boundaries,self.seconds_decimals):
            if type(v) is not int or v<=0:raise ValueError('Positive integer fit bounds required')
        if self.min_segment_beats<3 or self.max_segments not in (1,2) or self.seconds_decimals>12 or self.max_observations>10000 or self.max_boundaries>512:
            raise ValueError('Unsupported fit complexity')

    @property
    def fingerprint(self):return content_hash(asdict(self))


@dataclass(frozen=True,slots=True)
class FitResult:
    accepted:bool
    model_type:str
    mapping:TimingMap|None
    residual_seconds:tuple[float,...]
    p95_seconds:float
    max_seconds:float
    objective:float
    policy_hash:str
    observation_hash:str
    interpretation_hash:str
    coverage_seconds:tuple[float,float]
    warnings:tuple[str,...]
    reason:str|None


def _line(times,start,end):
    # Long-baseline median slopes resist a few local jittered intervals.
    half=(end-start+1)//2
    slope=median((times[i+half]-times[i])/half for i in range(start,end-half+1))
    intercept=median(times[i]-slope*i for i in range(start,end+1))
    return intercept,slope


def fit_timing(observation:TimingObservation,*,pulse_unit=Fraction(1,4),first_q=0,
               meter=(4,4),policy=FitPolicy()):
    if not isinstance(observation,TimingObservation) or not isinstance(policy,FitPolicy):raise ValueError('Observation and policy required')
    unit=rational(pulse_unit);origin=rational(first_q)
    if unit<=0:raise ValueError('Positive pulse interpretation required')
    signature=MeterSegment(0,*meter)
    interpretation=content_hash({'pulse_unit':str(unit),'first_q':str(origin),'meter':list(meter)})
    p=observation.payload;times=p['beat_times'];n=len(times)
    duration=Fraction(p['audio']['sample_count'],p['audio']['sample_rate'])
    coverage=(times[0],times[-1]) if times else (0.,0.)
    common=dict(policy_hash=policy.fingerprint,observation_hash=observation.artifact.artifact_hash,interpretation_hash=interpretation,coverage_seconds=coverage)
    warnings=['METER_AND_PULSE_ARE_EXPLICIT_CANDIDATES','UNOBSERVED_AUDIO_NOT_EXTRAPOLATED']
    if n<policy.min_segment_beats or n>policy.max_observations:
        return FitResult(False,'rejected',None,(),0,0,0,warnings=tuple(warnings),reason='INSUFFICIENT_BEATS' if n<policy.min_segment_beats else 'OBSERVATION_LIMIT',**common)

    def evaluate(points):
        anchors=tuple(Anchor(origin+unit*4*i,rational(round(max(0,min(float(duration),t)),policy.seconds_decimals))) for i,t in points)
        mapping=TimingMap(anchors,duration,unit,(signature,))
        residual=[]
        for i,t in enumerate(times):
            # q points are inside anchor span even if observed seconds jitter.
            residual.append(float(mapping.t(origin+unit*4*i))-t)
        absolute=sorted(abs(r) for r in residual)
        p95=absolute[max(0,math.ceil(.95*n)-1)];maximum=absolute[-1]
        loss=sum(min(r*r,policy.hard_limit_seconds**2) for r in residual)/n
        objective=loss+policy.complexity_penalty*(len(points)-2)
        accepted=p95<=policy.tolerance_seconds and maximum<=policy.hard_limit_seconds
        return mapping,tuple(residual),p95,maximum,objective,accepted

    intercept,slope=_line(times,0,n-1)
    fixed=evaluate([(0,intercept),(n-1,intercept+slope*(n-1))])
    def result(candidate,kind,accepted,reason=None):
        return FitResult(accepted,kind,candidate[0] if accepted else None,candidate[1],candidate[2],candidate[3],candidate[4],
            warnings=tuple(warnings),reason=reason,**common)
    # Do not introduce tiny tempo changes when one line already fits the budget.
    if fixed[-1]:return result(fixed,'fixed',True)
    best=None
    if policy.max_segments==2 and n>=policy.min_segment_beats*2-1:
        candidates=list(range(policy.min_segment_beats-1,n-policy.min_segment_beats+1))
        if len(candidates)>policy.max_boundaries:
            candidates=sorted({candidates[round(i*(len(candidates)-1)/(policy.max_boundaries-1))] for i in range(policy.max_boundaries)}) if policy.max_boundaries>1 else [candidates[len(candidates)//2]]
            warnings.append('BOUNDARY_SEARCH_BOUNDED')
        for boundary in candidates:
            left_a,left_b=_line(times,0,boundary);right_a,right_b=_line(times,boundary,n-1)
            middle=(left_a+left_b*boundary+right_a+right_b*boundary)/2
            try:candidate=evaluate([(0,left_a),(boundary,middle),(n-1,right_a+right_b*(n-1))])
            except ValueError:continue
            if candidate[-1] and candidate[4]<fixed[4] and (best is None or candidate[4]<best[4]):best=candidate
    if best is not None:return result(best,'piecewise',True)
    return result(fixed,'rejected',False,'RESIDUAL_OR_COMPLEXITY_BUDGET')
