"""Bounded, inspectable recurrence baseline; proposals never authorize reuse."""
from dataclasses import asdict,dataclass
from fractions import Fraction
from importlib.metadata import version
import math
from mdac2.domain.artifact import ArtifactEnvelope,content_hash
from mdac2.domain.structure import Span,SimilarityProposal


@dataclass(frozen=True,slots=True)
class SearchPolicy:
    version:str='recurrence-1'
    scales:tuple[int,...]=(1,2,4)
    weights:tuple[float,...]=(.4,.4,.2)
    min_similarity:float=.82
    min_active_fraction:float=.75
    max_bars:int=512
    max_proposals:int=20

    def __post_init__(self):
        if self.version!='recurrence-1' or type(self.scales) is not tuple or not self.scales or len(set(self.scales))!=len(self.scales) or any(type(s) is not int or s not in (1,2,4) for s in self.scales):raise ValueError('Supported unique search scales required')
        if type(self.weights) is not tuple or len(self.weights)!=3 or any(type(w) not in (int,float) or not math.isfinite(w) or w<0 for w in self.weights) or abs(sum(self.weights)-1)>1e-9:raise ValueError('Three normalized feature weights required')
        for value in (self.min_similarity,self.min_active_fraction):
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=1:raise ValueError('Bounded search threshold required')
        if type(self.max_bars) is not int or not 1<=self.max_bars<=512 or type(self.max_proposals) is not int or not 1<=self.max_proposals<=100:raise ValueError('Search resource limit exceeded')

    def to_dict(self):
        d=asdict(self);d['scales']=list(self.scales);d['weights']=list(self.weights);return d


def feature_rows(features,max_bars=512):
    if not isinstance(features,ArtifactEnvelope) or features.to_dict()['artifact_type']!='similarity_features':raise ValueError('Similarity feature artifact required')
    payload=features.payload;bars=payload.get('bars');cells=payload.get('cells_per_bar');mels=payload.get('n_mels')
    if payload.get('schema_version')!=1 or type(bars) is not list or not 1<=len(bars)<=max_bars or cells not in (8,16,32) or type(mels) is not int or not 12<=mels<=64:raise ValueError('Feature matrix shape/resource limit')
    previous=None
    for bar in bars:
        span=Span(Fraction(bar['span']['start_q']),Fraction(bar['span']['end_q']))
        if previous is not None and span.start_q<previous:raise ValueError('Unordered feature bars')
        previous=span.end_q
        meter=bar.get('meter')
        if type(meter) is not list or len(meter)!=2 or any(type(v) is not int or v<=0 for v in meter) or span.end_q-span.start_q!=Fraction(meter[0]*4,meter[1]):raise ValueError('Feature meter/span mismatch')
        if len(bar['cells'])!=cells:raise ValueError('Inconsistent feature cells')
        for i,cell in enumerate(bar['cells']):
            if Fraction(cell['start_q'])!=span.start_q+(span.end_q-span.start_q)*Fraction(i,cells) or Fraction(cell['end_q'])!=span.start_q+(span.end_q-span.start_q)*Fraction(i+1,cells):raise ValueError('Feature cell coverage mismatch')
            if type(cell['silent']) is not bool or Fraction(cell['end_seconds'])<=Fraction(cell['start_seconds']):raise ValueError('Invalid cell mask/time')
            for key,size in (('chroma',12),('mel',mels)):
                values=cell[key]
                if type(values) is not list or len(values)!=size or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values):raise ValueError('Invalid feature values')
            if type(cell['onset']) not in (int,float) or not math.isfinite(cell['onset']) or cell['onset']<0:raise ValueError('Invalid onset value')
    return bars


def proposal_dict(p):
    return {'proposal_id':p.proposal_id,'source':p.source.to_dict(),'target':p.target.to_dict(),
        'audio_hash':p.audio_hash,'timing_fingerprint':p.timing_fingerprint,'feature_hash':p.feature_hash,
        'similarity_by_source':[[k,v] for k,v in p.similarity_by_source],
        'boundary_uncertainty_seconds':p.boundary_uncertainty_seconds,
        'alignment_anchors':[[str(a),str(b)] for a,b in p.alignment_anchors],'warnings':list(p.warnings)}


@dataclass(frozen=True,slots=True)
class SearchResult:
    proposals:tuple[SimilarityProposal,...]
    artifact:ArtifactEnvelope


def search(features,*,policy=SearchPolicy()):
    if type(policy) is not SearchPolicy:raise ValueError('Typed search policy required')
    bars=feature_rows(features,policy.max_bars);n=len(bars)
    import numpy as np
    import librosa
    matrices=[]
    for key in ('chroma','mel','onset'):
        data=np.array([[v for c in b['cells'] for v in ([c[key]] if key=='onset' else c[key])] for b in bars],dtype=float)
        norms=np.linalg.norm(data,axis=1);active=norms>1e-12
        normalized=data/np.maximum(norms[:,None],1e-12)
        # librosa 0.11 validates width even in full mode and rejects <5 bars.
        if n<5:similarity=np.clip(normalized@normalized.T,0,1)
        else:
            distance=librosa.segment.recurrence_matrix(normalized.T,metric='cosine',mode='distance',full=True,self=True)
            similarity=np.clip(1-distance,0,1)
        similarity[~active,:]=0;similarity[:,~active]=0
        matrices.append(similarity)
    combined=sum(w*m for w,m in zip(policy.weights,matrices));candidates=[]
    def window(start,size):
        segment=bars[start:start+size]
        if len(segment)!=size or any(b['meter']!=segment[0]['meter'] for b in segment):return None
        if any(a['span']['end_q']!=b['span']['start_q'] for a,b in zip(segment,segment[1:])):return None
        active=sum(not c['silent'] for b in segment for c in b['cells'])/(size*len(segment[0]['cells']))
        if active<policy.min_active_fraction:return None
        return Span(Fraction(segment[0]['span']['start_q']),Fraction(segment[-1]['span']['end_q']))
    for scale in sorted(policy.scales,reverse=True):
        if n<scale*2:continue
        enhanced=combined if scale==1 else librosa.segment.path_enhance(combined,n=scale,window='boxcar',min_ratio=1,max_ratio=1,n_filters=1,mode='constant')
        for i in range(n-scale+1):
            source=window(i,scale)
            if source is None:continue
            for j in range(i+scale,n-scale+1):
                target=window(j,scale)
                if target is None or bars[i]['meter']!=bars[j]['meter'] or source.overlaps(target):continue
                scores=[float(np.mean([m[i+k,j+k] for k in range(scale)])) for m in matrices]
                score=sum(w*s for w,s in zip(policy.weights,scores))
                path_score=float(max(enhanced[i+k,j+k] for k in range(scale)))
                if min(score,path_score)<policy.min_similarity:continue
                candidates.append((score,scale,i,j,source,target,scores,path_score))
    # Prefer the longest supported boundary before suppressing nested/sliding pairs.
    candidates.sort(key=lambda v:(-v[1],-v[0],v[2],v[3]));kept=[]
    for item in candidates:
        score,scale,i,j,source,target,scores,path_score=item
        if any(source.overlaps(p.source) and target.overlaps(p.target) for p in kept):continue
        identity=content_hash({'features':features.artifact_hash,'policy':policy.to_dict(),'source':source.to_dict(),'target':target.to_dict()})
        uncertainty=max(float(Fraction(c['end_seconds'])-Fraction(c['start_seconds'])) for b in bars[i:i+scale]+bars[j:j+scale] for c in b['cells'])
        kept.append(SimilarityProposal(identity,source,target,features.payload['audio_hash'],features.payload['timing_fingerprint'],features.artifact_hash,
            tuple(zip(('mix_chroma','mix_mel','mix_onset','combined','diagonal_path'),[*scores,score,min(1.,path_score)])),uncertainty,
            warnings=('SPECTRAL_BASELINE_ONLY','BAR_BOUNDARIES_NOT_REFINED','REQUIRES_HUMAN_CONFIRMATION')))
        if len(kept)>=policy.max_proposals:break
    artifact=ArtifactEnvelope.create(artifact_type='similarity_proposals',producer={'name':'mdac2.librosa-recurrence','version':'1','model_hash':None,'dependencies':{'librosa':version('librosa'),'numpy':version('numpy')}},
        inputs=[features.artifact_hash],options=policy.to_dict(),payload={'schema_version':1,'proposals':[proposal_dict(p) for p in kept],
            'raw_candidate_count':len(candidates),'suppressed_or_limited':len(candidates)-len(kept),'bar_count':n,'matrix_cells':n*n,'mode':'mix_only',
            'matrix_method':'direct_cosine_small_matrix' if n<5 else 'librosa_full_recurrence'})
    return SearchResult(tuple(kept),artifact)
