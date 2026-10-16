"""Version-pinned mir_eval wrappers with visible missing/empty semantics."""
from dataclasses import asdict,dataclass
from fractions import Fraction
from importlib.metadata import version
import math
import warnings
from mdac2.domain.artifact import ArtifactEnvelope,content_hash


@dataclass(frozen=True,slots=True)
class MetricProfile:
    name:str='mdac-baseline-1'
    beat_window:float=.07
    onset_window:float=.05
    boundary_window:float=.5
    tempo_tolerance:float=.04
    strict_factor:float=.5

    def __post_init__(self):
        if self.name!='mdac-baseline-1':raise ValueError('Unknown metric profile')
        for value in (self.beat_window,self.onset_window,self.boundary_window,self.tempo_tolerance,self.strict_factor):
            if type(value) not in (float,int) or not math.isfinite(value) or not 0<value<=1:raise ValueError('Positive bounded metric tolerances required')
        if self.strict_factor>=1:raise ValueError('Sensitivity tolerance must be stricter')


def events(values):
    if type(values) not in (list,tuple) or len(values)>100000:raise ValueError('Bounded event sequence required')
    if any(type(v) not in (int,float,Fraction) or not math.isfinite(v) or not 0<=v<=30000 for v in values):raise ValueError('Finite nonnegative event seconds required')
    result=[float(v) for v in values]
    if any(a>=b for a,b in zip(result,result[1:])):raise ValueError('Unique increasing event seconds required; no silent sorting or deduplication')
    return result


def libraries():
    if version('mir_eval')!='0.8.2':raise ValueError('Metric runtime differs from fixed mir_eval 0.8.2')
    import numpy as np
    import mir_eval
    return np,mir_eval


def result(kind,reference,estimate,profile,scores,notices):
    return ArtifactEnvelope.create(artifact_type='standard_metrics',producer={'name':'mdac2.mir-eval-wrapper','version':'1','model_hash':None,
        'dependencies':{n:version(n) for n in ('mir_eval','numpy','scipy')}},inputs=[],
        options={**asdict(profile),'kind':kind,'reference':content_hash(reference),'estimate':content_hash(estimate),'preprocessing':'none-no-trim-no-deduplicate',
            'tempo_factors':[.5,1.,2.],'boundary_trim':False},
        payload={'schema_version':1,'kind':kind,'profile':asdict(profile),'status':'missing' if estimate is None else 'empty' if estimate==[] else 'ok',
            'scores':scores,'warnings':notices,'reference_count':len(reference) if type(reference) is list else 1,'estimate_available':estimate is not None})


def event_scores(reference,estimate,*,kind='beat',profile=MetricProfile()):
    if kind not in ('beat','downbeat','onset') or type(profile) is not MetricProfile:raise ValueError('Supported metric kind/profile required')
    ref=events(reference);est=None if estimate is None else events(estimate);np,mir=libraries();scores={}
    window=profile.onset_window if kind=='onset' else profile.beat_window
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for name,factor in (('baseline',1),('strict',profile.strict_factor)):
            r,e=np.array(ref),np.array(est or [])
            f,p,recall=mir.onset.f_measure(r,e,window=window*factor)
            if kind!='onset':f=mir.beat.f_measure(r,e,f_measure_threshold=window*factor)
            scores[name]={'f1':float(f),'precision':float(p),'recall':float(recall),'window_seconds':window*factor}
    notices=sorted({str(w.message) for w in caught})
    if est is None:notices.append('MISSING_ESTIMATE_COUNTS_AS_FAILURE')
    if not ref:notices.append('EMPTY_REFERENCE_NOT_POSITIVE_ACCURACY_EVIDENCE')
    return result(kind,ref,est,profile,scores,notices)


def tempo_scores(reference,estimate,*,profile=MetricProfile()):
    for v in (reference,estimate):
        if v is not None and (type(v) not in (int,float) or not math.isfinite(v) or not 0<v<=1000):raise ValueError('Positive finite tempo required')
    if reference is None:raise ValueError('Tempo reference required')
    np,mir=libraries();scores={}
    for name,factor in (('baseline',1),('strict',profile.strict_factor)):
        tolerance=profile.tempo_tolerance*factor
        hits=[False]*3 if estimate is None else [mir.tempo.detection(np.array([reference*k,reference*k]),.5,np.array([estimate,estimate]),tol=tolerance)[1] for k in (.5,1.,2.)]
        scores[name]={'acc1':float(hits[1]),'acc2_half_double':float(any(hits)),'relative_tolerance':tolerance}
    return result('tempo',reference,estimate,profile,scores,['MISSING_ESTIMATE_COUNTS_AS_FAILURE'] if estimate is None else [])


def boundary_scores(reference,estimate,*,profile=MetricProfile()):
    ref=events(reference);est=None if estimate is None else events(estimate)
    if len(ref)<2 or est is not None and est and (len(est)<2 or est[0]!=ref[0] or est[-1]!=ref[-1]):raise ValueError('Segmentation requires matching explicit coverage endpoints')
    np,mir=libraries();scores={}
    for name,factor in (('baseline',1),('strict',profile.strict_factor)):
        window=profile.boundary_window*factor
        if not est:p=r=f=0.
        else:p,r,f=mir.segment.detection(np.array(list(zip(ref,ref[1:]))),np.array(list(zip(est,est[1:]))),window=window,trim=False)
        scores[name]={'precision':float(p),'recall':float(r),'f1':float(f),'window_seconds':window}
    return result('segment_boundaries',ref,est,profile,scores,['MISSING_ESTIMATE_COUNTS_AS_FAILURE'] if est is None else [])
