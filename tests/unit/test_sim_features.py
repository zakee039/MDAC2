from dataclasses import replace
from fractions import Fraction as F
import math
import struct
import pytest
from mdac2.analysis.similarity_features import FeaturePolicy,extract_features,bar_spans
from mdac2.domain.timing import TimingMap,Anchor,MeterSegment
from mdac2.domain.timing_edits import TimingRevision
from mdac2.ports.timing import TimingInput


def test_meter_changes_and_partial_pickup_use_real_bar_lengths():
    mapping=TimingMap((Anchor(-1,0),Anchor(15,8)),8,meter_segments=(MeterSegment(0,4,4),MeterSegment(8,6,8)))
    t=TimingRevision('t',mapping,review='accepted')
    assert [(s.start_q,s.end_q,m) for s,m in bar_spans(t,FeaturePolicy())]==[(0,4,(4,4)),(4,8,(4,4)),(8,11,(6,8)),(11,14,(6,8))]
    with pytest.raises(ValueError):bar_spans(replace(t,review='pending'),FeaturePolicy())
    with pytest.raises(ValueError):bar_spans(t,FeaturePolicy(max_bars=1))


def test_real_librosa_silence_mask_antiphase_and_identity():
    rate=8000;values=[]
    for i in range(rate*4):
        v=round(10000*math.sin(2*math.pi*440*i/rate)) if i<rate*2 else 0
        values.extend((v,-v))
    pcm=struct.pack('<'+'h'*len(values),*values);a=TimingInput('a'*64,rate,rate*4,2)
    t=TimingRevision('t',TimingMap((Anchor(0,0),Anchor(8,4)),4),review='accepted')
    result=extract_features(a,pcm,t);bars=result.payload['bars']
    assert not bars[0]['silent'] and bars[1]['silent']
    assert any(v>0 for v in bars[0]['cells'][0]['chroma'])
    assert all(v==0 for c in bars[1]['cells'] for v in c['mel'])
    assert result.to_dict()['producer']['dependencies']['librosa']=='0.11.0'
    changed=extract_features(a,pcm,replace(t,revision_id='new'))
    assert changed.artifact_hash!=result.artifact_hash
    changed_mels=extract_features(a,pcm,t,policy=FeaturePolicy(n_mels=24))
    assert changed_mels.request_key!=result.request_key


def test_local_cell_seconds_follow_variable_tempo():
    t=TimingRevision('t',TimingMap((Anchor(0,0),Anchor(4,2),Anchor(8,3)),3),review='accepted')
    a=TimingInput('a'*64,8000,24000,1)
    rows=extract_features(a,bytes(48000),t).payload['bars']
    widths=[F(r['cells'][0]['end_seconds'])-F(r['cells'][0]['start_seconds']) for r in rows]
    assert widths==[F(1,8),F(1,16)]


def test_spectral_memory_budget_rejects_before_analysis():
    t=TimingRevision('t',TimingMap((Anchor(0,0),Anchor(8,4)),4),review='accepted')
    a=TimingInput('a'*64,8000,32000,1)
    with pytest.raises(ValueError,match='memory bound'):
        extract_features(a,bytes(64000),t,policy=FeaturePolicy(max_spectral_cells=100))
