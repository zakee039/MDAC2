"""Frame targets for one joint onset/arrangement model; numpy-only contracts."""
import math
import numpy as np
from .training_chart import CLASSES

HOP_SECONDS=.01
WINDOW_FRAMES=3200
CONTEXT_FRAMES=800
CORE_FRAMES=1600


def frame_targets(chart):
    count=math.ceil(chart['duration']/HOP_SECONDS-1e-8)
    onsets=np.zeros((count,2,len(CLASSES)),dtype=np.float32)
    offsets=np.zeros_like(onsets)
    onset_mask=np.ones_like(onsets,dtype=np.bool_)
    offset_mask=np.zeros_like(onsets,dtype=np.bool_)
    hold_ends=np.zeros((count,2),dtype=np.float32)
    end_mask=np.ones((count,2),dtype=np.bool_)
    end_offsets=np.zeros((count,2),dtype=np.float32)
    end_offset_mask=np.zeros((count,2),dtype=np.bool_)
    occupancy=np.zeros((count,2),dtype=np.float32)
    counts=np.zeros_like(onsets,dtype=np.int16)
    end_counts=np.zeros_like(hold_ends,dtype=np.int16)
    collisions=[]
    for excluded in chart['excluded']:
        channel=excluded['channel']
        sides=[1] if channel in {'11','13','31','33','51','53','D1','D3'} else [0]
        first=max(0,int(excluded['start']/HOP_SECONDS)-1)
        last=min(count,int(excluded['end']/HOP_SECONDS)+2)
        for side in sides:onset_mask[first:last,side,:]=False
    for e in chart['events']:
        frame=min(count-1,int(e['start']/HOP_SECONDS+1e-7))
        kind=CLASSES.index(e['kind'])
        for lane in e['lanes']:
            side=0 if lane=='ground' else 1
            counts[frame,side,kind]+=1
            onsets[frame,side,kind]=1
            offsets[frame,side,kind]=e['start']/HOP_SECONDS-frame
            offset_mask[frame,side,kind]=True
            # A discarded ghost/item must not erase a valid same-frame event.
            onset_mask[frame,side,kind]=True
            if e['kind']=='hold':
                times=np.arange(count,dtype=np.float64)*HOP_SECONDS
                occupancy[(times>=e['start'])&(times<e['end']),side]=1
                tail=int(e['end']/HOP_SECONDS+1e-7)
                if tail<count:
                    hold_ends[tail,side]=1;end_counts[tail,side]+=1
                    end_offsets[tail,side]=e['end']/HOP_SECONDS-tail
                    end_offset_mask[tail,side]=True
    crowded=counts>1
    onset_mask[crowded]=False;offset_mask[crowded]=False
    for f,side,kind in np.argwhere(crowded):
        collisions.append({'frame':int(f),'lane':int(side),'kind':CLASSES[kind],'count':int(counts[f,side,kind])})
    end_mask[end_counts>1]=False;end_offset_mask[end_counts>1]=False
    # These targets may also be generated for review; unsafe charts never
    # accidentally become training data just because tensors can be constructed.
    if not chart['training_eligible']:
        onset_mask[:]=False;offset_mask[:]=False;end_mask[:]=False;end_offset_mask[:]=False
    return {'onsets':onsets,'offsets':offsets,'onset_mask':onset_mask,'offset_mask':offset_mask,
            'hold_ends':hold_ends,'hold_end_mask':end_mask,'hold_end_offsets':end_offsets,
            'hold_end_offset_mask':end_offset_mask,'occupancy':occupancy,
            'occupancy_mask':np.full((count,2),chart['training_eligible'],dtype=np.bool_),
            'collisions':collisions}


def timing_features(chart,difficulty,count):
    frames=np.arange(count)*HOP_SECONDS
    tempos=chart['timing']['tempos']
    positions=np.array([t['seconds'] for t in tempos])
    indexes=np.clip(np.searchsorted(positions,frames,side='right')-1,0,len(tempos)-1)
    from fractions import Fraction
    bpm=np.array([float(Fraction(t['bpm'])) for t in tempos])[indexes]
    q0=np.array([float(Fraction(t['quarter'])) for t in tempos])[indexes]
    quarter=q0+(frames-positions[indexes])*bpm/60
    return np.stack([np.sin(quarter*2*np.pi),np.cos(quarter*2*np.pi),np.log2(bpm/120),
                     np.full(count,(float(difficulty)-9.5))],axis=-1).astype(np.float32)


def window_starts(frame_count):
    return [core-CONTEXT_FRAMES for core in range(0,frame_count,CORE_FRAMES)]


def slice_window(array,start,*,time_axis=0):
    shape=list(array.shape);shape[time_axis]=WINDOW_FRAMES
    result=np.zeros(shape,dtype=array.dtype)
    count=array.shape[time_axis];left=max(0,start);right=min(count,start+WINDOW_FRAMES)
    if right>left:
        source=[slice(None)]*array.ndim;target=source.copy()
        source[time_axis]=slice(left,right);target[time_axis]=slice(left-start,right-start)
        result[tuple(target)]=array[tuple(source)]
    valid=np.zeros(WINDOW_FRAMES,dtype=np.bool_)
    if right>left:
        valid[max(CONTEXT_FRAMES,left-start):min(CONTEXT_FRAMES+CORE_FRAMES,right-start)]=True
    return result,valid
