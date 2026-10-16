from fractions import Fraction as F
import pytest
from mdac2.domain.artifact import ArtifactEnvelope
from mdac2.analysis.similarity_search import search,SearchPolicy


def features(labels,*,silent=(),meters=None):
    bars=[];q=F(0)
    for i,label in enumerate(labels):
        meter=(meters or [(4,4)]*len(labels))[i];width=F(meter[0]*4,meter[1]);cells=[]
        for k in range(16):
            a=q+width*F(k,16);b=q+width*F(k+1,16);mute=i in silent
            cells.append({'start_q':str(a),'end_q':str(b),'start_seconds':str(a/2),'end_seconds':str(b/2),
                'silent':mute,'rms':0. if mute else .2,'chroma':[0. if mute else float(v==label) for v in range(12)],
                'mel':[0. if mute else float(v==label) for v in range(12)],'onset':0. if mute else float(k==label)})
        bars.append({'span':{'start_q':str(q),'end_q':str(q+width)},'meter':list(meter),'cells':cells,'silent':i in silent});q+=width
    return ArtifactEnvelope.create(artifact_type='similarity_features',producer={'name':'fixture','version':'1','model_hash':None,'dependencies':{}},inputs=['a'*64],options={},
        payload={'schema_version':1,'audio_hash':'a'*64,'timing_fingerprint':'b'*64,'cells_per_bar':16,'n_mels':12,'bars':bars})


def test_repeated_phrase_path_and_nested_suppression():
    f=features([0,1,2,3,0,1,2,3]);result=search(f)
    assert len(result.proposals)==1
    p=result.proposals[0]
    assert (p.source.start_q,p.source.end_q,p.target.start_q,p.target.end_q)==(0,16,16,32)
    assert dict(p.similarity_by_source)['combined']==pytest.approx(1)
    assert p.alignment_anchors==() and 'REQUIRES_HUMAN_CONFIRMATION' in p.warnings
    assert result.artifact.payload['suppressed_or_limited']>0
    assert search(f,policy=SearchPolicy(max_proposals=1)).artifact.request_key!=result.artifact.request_key


def test_nonrepeat_silence_meter_and_single_bar():
    assert not search(features([0,1,2,3])).proposals
    assert not search(features([0,0,0,0],silent=(0,1,2,3))).proposals
    assert not search(features([0])).proposals
    result=search(features([0,1,0,1],meters=[(4,4),(6,8),(4,4),(6,8)]))
    assert all(p.source.end_q-p.source.start_q in (3,4) for p in result.proposals)


def test_limits_and_identity():
    f=features([0]*8)
    with pytest.raises(ValueError,match='resource'):search(f,policy=SearchPolicy(max_bars=4))
    with pytest.raises(ValueError):SearchPolicy(weights=(1.,1.,1.))
    result=search(f,policy=SearchPolicy(scales=(1,),max_proposals=2))
    assert len(result.proposals)==2
    assert all(not p.source.overlaps(p.target) for p in result.proposals)
    assert result.artifact.artifact_hash==search(f,policy=SearchPolicy(scales=(1,),max_proposals=2)).artifact.artifact_hash
