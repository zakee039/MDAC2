from copy import deepcopy
from fractions import Fraction as F
import pytest
from mdac2.data_preparation.community_labels import convert, SourceTiming
from mdac2.data_preparation.training_chart import TrainingEvent, validate_events


def raw(cells, headers=None):
    return {'headers':headers or {'BPM':['120']}, 'errors':[],
            'cells':[{'measure':m,'channel':ch,'index':i,'divisions':d,'code':code,'line':n+10}
                     for n,(m,ch,i,d,code) in enumerate(cells)]}


def run(source):
    return convert(source, chart_hash='a'*64, duration=60)


def test_user_mapping_and_excluded_roles_keep_same_time_valid_notes():
    source=raw([(1,'13',0,1,'0C'),(2,'14',0,1,'0I'),(3,'14',0,1,'11'),
                (4,'13',0,1,'23'),(4,'14',0,1,'21'),(5,'14',0,1,'22')])
    original=deepcopy(source);result=run(source)
    assert [e['kind'] for e in result['events']]==['light','light','extra_heavy','light']
    assert result['events'][2]['exclusive']
    assert {x['reason'] for x in result['excluded']}=={'ghost','heart'}
    assert result['training_eligible'] and source==original
    assert result==run(source)


def test_masher_double_heavy_does_not_invent_hold_and_retains_independent_hold():
    result=run(raw([(1,'54',0,1,'0G'),(2,'54',0,1,'0G'),
                    (1,'53',0,1,'0F'),(2,'53',0,1,'0F'),(3,'D4',0,1,'0G')]))
    assert len(result['events'])==5
    assert [e['kind'] for e in result['events']].count('hold')==1
    assert len({e['group_id'] for e in result['events'] if e['kind']=='heavy'})==2
    assert result['training_eligible']


def test_boss_hold_real_end_and_tap_form_quarantine():
    result=run(raw([(1,'54',0,1,'16'),(3,'54',0,1,'16'),(4,'D4',0,1,'17')]))
    assert len(result['events'])==1
    event=result['events'][0]
    assert event['kind']=='hold' and event['lanes']==['ground','air']
    assert (event['start'],event['end'])==(2,6)
    assert result['unresolved'][0]['reason']=='boss_masher_without_sustained_interval'
    assert not result['training_eligible']


def test_exact_cross_lane_mashers_expand_once_with_all_provenance():
    result=run(raw([(1,'53',0,1,'0G'),(2,'53',0,1,'0G'),
                    (1,'54',0,1,'0G'),(2,'54',0,1,'0G'),
                    (3,'53',0,1,'16'),(4,'53',0,1,'16'),
                    (3,'54',0,1,'16'),(4,'54',0,1,'16')]))
    assert len(result['events'])==3 and result['training_eligible']
    assert all(len(e['source_ids'])==4 for e in result['events'])
    heavy=[e for e in result['events'] if e['kind']=='heavy']
    assert len({e['group_id'] for e in heavy})==1


def test_same_lane_masher_duplicates_are_not_silently_deduplicated():
    result=run(raw([(1,'D3',0,1,'0G'),(1,'D3',0,1,'0G')]))
    assert len(result['events'])==4 and not result['training_eligible']


def test_gemini_pair_and_singleton_are_not_silently_merged():
    result=run(raw([(1,'13',0,1,'0E'),(1,'14',0,1,'0E'),(2,'13',0,1,'0E')]))
    assert len(result['events'])==1 and result['events'][0]['kind']=='gemini'
    assert result['unresolved'][0]['reason']=='unpaired_or_ambiguous_gemini'


def test_exclusive_cross_lane_and_half_open_hold_boundary():
    def event(key,kind,lanes,start,end):
        return TrainingEvent(key,kind,lanes,start,end,(key,),'11','test',kind=='extra_heavy')
    hold=event('h','hold',('air',),1,3)
    middle=event('x','extra_heavy',('ground',),2,2)
    after=event('y','extra_heavy',('ground',),3,3)
    assert validate_events([hold,middle],60)[0]['code']=='exclusive_overlap'
    assert validate_events([hold,after],60)==[]


def test_music_offset_bpm_and_measure_length_are_integrated():
    source=raw([(2,'08',0,1,'01'),(1,'01',0,1,'10'),(3,'13',0,1,'01')],
               {'BPM':['120'],'BPM01':['240']})
    source['cells'].insert(0,{'measure':1,'channel':'02','raw':'0.5','line':5})
    timing=SourceTiming(source)
    assert timing.quarter(F(3))==10
    assert timing.music_start==2
    assert run(source)['events'][0]['start']==2


def test_late_timing_unknown_channels_and_duplicate_tempo_are_visible():
    for source in [raw([(3,'13',0,1,'01'),(2,'08',0,1,'01')],{'BPM':['120'],'BPM01':['240']}),
                   raw([(1,'09',0,1,'01')]),
                   raw([(1,'08',0,1,'01'),(1,'08',0,1,'01')],{'BPM':['120'],'BPM01':['240']})]:
        result=run(source)
        assert not result['training_eligible']
        assert result['issues'][0]['code']=='timing_unresolved'


def test_unknown_object_and_unclosed_hold_never_become_light_or_empty_training():
    result=run(raw([(1,'13',0,1,'3G'),(2,'53',0,1,'0F')]))
    assert not result['events'] and not result['training_eligible']
    assert {x['reason'] for x in result['unresolved']}=={'unknown_object','unclosed_hold'}


def test_type_duration_and_exclusive_contract_reject_invalid_events():
    with pytest.raises(ValueError):
        TrainingEvent('e','extra_heavy',('air',),1,1,('s',),'11','boss',False)
    with pytest.raises(ValueError):
        TrainingEvent('e','light',('air',),1,2,('s',),'01','small')
