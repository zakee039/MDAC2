"""Rejectable local DTW with strictly increasing musical-coordinate anchors."""
from dataclasses import asdict,dataclass
from fractions import Fraction as F
from importlib.metadata import version
import math
from mdac2.analysis.similarity_search import feature_rows
from mdac2.domain.artifact import ArtifactEnvelope
from mdac2.domain.structure import SimilarityProposal


@dataclass(frozen=True,slots=True)
class AlignmentPolicy:
    version:str='constrained-dtw-1'
    band_radius:float=.15
    max_mean_cost:float=.2
    min_coverage:float=.75
    min_active_fraction:float=.75
    max_stretch:float=2.
    max_cells:int=256

    def __post_init__(self):
        if self.version!='constrained-dtw-1':raise ValueError('Unknown alignment policy')
        for v in (self.band_radius,self.max_mean_cost,self.min_coverage,self.min_active_fraction):
            if type(v) not in (int,float) or not math.isfinite(v) or not 0<v<=1:raise ValueError('Bounded alignment threshold required')
        if type(self.max_stretch) not in (int,float) or not math.isfinite(self.max_stretch) or not 1<=self.max_stretch<=2 or type(self.max_cells) is not int or not 2<=self.max_cells<=256:raise ValueError('Alignment resource/stretch bound exceeded')


@dataclass(frozen=True,slots=True)
class AlignmentResult:
    anchors:tuple[tuple[F,F],...]
    artifact:ArtifactEnvelope

    @property
    def accepted(self):return self.artifact.payload['status']=='accepted'


def align(features,proposal,*,policy=AlignmentPolicy()):
    if type(proposal) is not SimilarityProposal or type(policy) is not AlignmentPolicy:raise ValueError('Typed alignment inputs required')
    bars=feature_rows(features)
    if proposal.feature_hash!=features.artifact_hash or proposal.audio_hash!=features.payload['audio_hash'] or proposal.timing_fingerprint!=features.payload['timing_fingerprint']:raise ValueError('Stale/mismatched alignment inputs')
    def extract(span):
        selected=[b for b in bars if F(b['span']['start_q'])>=span.start_q and F(b['span']['end_q'])<=span.end_q]
        if not selected or F(selected[0]['span']['start_q'])!=span.start_q or F(selected[-1]['span']['end_q'])!=span.end_q:raise ValueError('Alignment requires complete feature bars')
        if any(F(a['span']['end_q'])!=F(b['span']['start_q']) for a,b in zip(selected,selected[1:])):raise ValueError('Missing feature coverage')
        cells=[c for b in selected for c in b['cells']]
        if len(cells)>policy.max_cells:raise ValueError('Local alignment matrix limit exceeded')
        return selected,cells
    source,x=extract(proposal.source);target,y=extract(proposal.target)
    reasons=[];anchors=();metrics={}
    if any(b['meter']!=source[0]['meter'] for b in source+target):reasons.append('METER_CHANGE_UNSUPPORTED')
    ratio=(proposal.target.end_q-proposal.target.start_q)/(proposal.source.end_q-proposal.source.start_q)
    if not 1/policy.max_stretch<=ratio<=policy.max_stretch:reasons.append('GLOBAL_STRETCH_EXCEEDED')
    if min(sum(not c['silent'] for c in x)/len(x),sum(not c['silent'] for c in y)/len(y))<policy.min_active_fraction:reasons.append('INSUFFICIENT_ACTIVE_AUDIO')
    if not reasons:
        import numpy as np
        import librosa
        def vectors(cells):
            # Onset is bounded separately so loudness cannot dominate pitch/timbre.
            a=np.array([c['chroma']+c['mel']+[c['onset']/(1+c['onset'])] for c in cells],dtype=float)
            return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
        cost=np.clip(1-vectors(x)@vectors(y).T,0,1)
        for i,a in enumerate(x):
            for j,b in enumerate(y):
                if a['silent'] or b['silent']:cost[i,j]=1.
        try:
            _,reverse_path=librosa.sequence.dtw(C=cost,step_sizes_sigma=np.array([[1,1],[1,2],[2,1]]),
                weights_add=np.array([0.,.05,.05]),global_constraints=True,band_rad=policy.band_radius)
        except librosa.util.exceptions.ParameterError:
            reasons.append('NO_CONSTRAINED_PATH')
        else:
            path=reverse_path[::-1]
            coverage=min(len(set(path[:,0]))/len(x),len(set(path[:,1]))/len(y))
            mean_cost=float(np.mean([cost[i,j] for i,j in path]))
            metrics={'coverage':coverage,'mean_cost':mean_cost,'path_cells':len(path),'matrix_cells':len(x)*len(y)}
            if coverage<policy.min_coverage:reasons.append('INSUFFICIENT_PATH_COVERAGE')
            if mean_cost>policy.max_mean_cost:reasons.append('ALIGNMENT_COST_EXCEEDED')
            centers=lambda cells:[(F(c['start_q'])+F(c['end_q']))/2 for c in cells]
            qx,qy=centers(x),centers(y)
            proposed=((proposal.source.start_q,proposal.target.start_q),*[(qx[i],qy[j]) for i,j in path],(proposal.source.end_q,proposal.target.end_q))
            slopes=[(b[1]-a[1])/(b[0]-a[0]) for a,b in zip(proposed,proposed[1:])]
            metrics['max_local_stretch']=float(max(max(slopes),1/min(slopes)))
            if any(not 1/policy.max_stretch<=s<=policy.max_stretch for s in slopes):reasons.append('LOCAL_STRETCH_EXCEEDED')
            if not reasons:anchors=tuple(proposed)
    artifact=ArtifactEnvelope.create(artifact_type='phrase_alignment',producer={'name':'mdac2.librosa-dtw','version':'1','model_hash':None,'dependencies':{'librosa':version('librosa'),'numpy':version('numpy')}},
        inputs=[features.artifact_hash],options={**asdict(policy),'proposal_fingerprint':proposal.fingerprint,'steps':[[1,1],[1,2],[2,1]],'skip_penalty':.05},
        payload={'schema_version':1,'status':'rejected' if reasons else 'accepted','reasons':reasons,'metrics':metrics,
            'proposal_fingerprint':proposal.fingerprint,'timing_fingerprint':proposal.timing_fingerprint,
            'anchors':[[str(a),str(b)] for a,b in anchors],'warnings':['ALIGNMENT_IS_NOT_REUSE_PERMISSION']})
    return AlignmentResult(anchors,artifact)
