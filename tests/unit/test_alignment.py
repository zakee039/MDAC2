from dataclasses import replace
import pytest
from test_similarity import features
from mdac2.analysis.phrase_alignment import align,AlignmentPolicy
from mdac2.analysis.similarity_search import search
from mdac2.domain.structure import SimilarityProposal,Span


def proposal(f,source=Span(0,8),target=Span(8,16)):
    return SimilarityProposal('fixture',source,target,'a'*64,'b'*64,f.artifact_hash,(('fixture',1.),),.1)


def test_repeat_has_monotone_exact_q_anchors_and_identity():
    f=features([0,1,0,1]);p=proposal(f);r=align(f,p)
    assert r.accepted and r.anchors[0]==(0,8) and r.anchors[-1]==(8,16)
    assert all(b-a==8 for a,b in r.anchors)
    assert r.artifact.payload['metrics']['coverage']==1
    assert r.artifact.artifact_hash!=align(f,p,policy=AlignmentPolicy(max_mean_cost=.1)).artifact.artifact_hash
    with pytest.raises(ValueError,match='Stale'):align(f,replace(p,timing_fingerprint='c'*64))


def test_unrelated_silence_and_meter_are_rejected_without_anchors():
    f=features([0,1,2,3]);r=align(f,proposal(f))
    assert not r.accepted and not r.anchors and 'ALIGNMENT_COST_EXCEEDED' in r.artifact.payload['reasons']
    f=features([0,1,0,1],silent=(2,3));assert not align(f,proposal(f)).accepted
    f=features([0,1],meters=[(4,4),(6,8)])
    assert 'METER_CHANGE_UNSUPPORTED' in align(f,proposal(f,Span(0,4),Span(4,7))).artifact.payload['reasons']


def test_resource_bounds_and_search_chain():
    f=features([0,1,2,3,0,1,2,3]);p=search(f).proposals[0]
    assert align(f,p).accepted
    with pytest.raises(ValueError,match='matrix limit'):align(f,p,policy=AlignmentPolicy(max_cells=16))
    with pytest.raises(ValueError):AlignmentPolicy(max_stretch=3)
