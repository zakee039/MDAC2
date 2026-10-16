import numpy as np
from mdac2.data_preparation.training_tensors import frame_targets,slice_window,timing_features,window_starts


def chart(events,excluded=None):
    return {'duration':1.,'events':events,'excluded':excluded or [],'training_eligible':True,
            'timing':{'tempos':[{'seconds':0,'quarter':'0','bpm':'120'}]}}


def event(kind,start,end=None,lanes=None):
    return {'kind':kind,'start':start,'end':start if end is None else end,'lanes':lanes or ['ground']}


def test_ghost_mask_keeps_valid_same_frame_event_and_hold_boundaries():
    c=chart([event('light',.2),event('hold',.1,.3, ['air'])],
            [{'channel':'14','start':.2,'end':.2}])
    t=frame_targets(c)
    assert t['onsets'][20,0,0] and t['onset_mask'][20,0,0]
    assert not t['onset_mask'][20,0,1]
    assert t['occupancy'][10,1] and t['occupancy'][29,1]
    assert not t['occupancy'][30,1]
    assert t['hold_ends'][30,1] and t['hold_end_offset_mask'][30,1]


def test_subframe_hold_and_legitimate_parallel_events_survive():
    c=chart([event('hold',.201,.202),event('heavy',.2),event('heavy',.2,lanes=['air'])])
    t=frame_targets(c)
    assert t['onsets'][20,0,4] and t['hold_ends'][20,0]
    assert np.isclose(t['offsets'][20,0,4],.1)
    assert np.isclose(t['hold_end_offsets'][20,0],.2)
    assert t['onsets'][20,:,1].all()


def test_quantization_collision_is_masked_not_silently_overwritten():
    t=frame_targets(chart([event('light',.201),event('light',.207)]))
    assert t['onsets'][20,0,0] and not t['onset_mask'][20,0,0]
    assert not t['offset_mask'][20,0,0]
    assert t['collisions']==[{'frame':20,'lane':0,'kind':'light','count':2}]


def test_window_cores_cover_song_once_with_context_and_no_padding_loss():
    array=np.arange(3800)
    supervised=[]
    for start in window_starts(len(array)):
        block,mask=slice_window(array,start)
        supervised.extend(block[mask].tolist())
    assert supervised==array.tolist()
    shorter,mask=slice_window(np.arange(10),-800)
    assert len(shorter)==3200 and mask.sum()==10


def test_tempo_and_difficulty_conditions_use_beat_and_audio_origin():
    c=chart([])
    c['timing']['tempos']=[{'seconds':-.5,'quarter':'0','bpm':'120'},
                            {'seconds':.5,'quarter':'2','bpm':'240'}]
    values=timing_features(c,'8.5',100)
    assert values.shape==(100,4)
    assert np.isclose(values[0,1],1) and np.isclose(values[50,2],1)
    assert np.all(values[:,3]==-1)


def test_quarantined_chart_never_exposes_supervised_mask():
    c=chart([event('light',.2)]);c['training_eligible']=False
    target=frame_targets(c)
    assert all(not value.any() for key,value in target.items() if 'mask' in key)
