"""Stitch acoustic centers, then decode complete songs with persistent history."""
import numpy as np
import torch
from .dual_data import PATTERNS, BOS, NAMES
from .joint_decoding import peaks,place_core
from .training_tensors import slice_window,window_starts
from .training_chart import TrainingEvent,LANES
from .generated_constraints import validate_generated_events


@torch.no_grad()
def acoustic_predictions(onset,arrangement,features,frame_count,device='cuda',amp=True):
    output=dict(probability=np.zeros(frame_count,np.float32),offsets=np.zeros(frame_count,np.float32),
                acoustic=np.zeros((frame_count,128),np.float32))
    coverage=np.zeros(frame_count,np.int32)
    onset.eval();arrangement.eval()
    for start in window_starts(frame_count):
        audio,_=slice_window(features,start,time_axis=2)
        b=torch.from_numpy(audio.astype(np.float32)[None]).to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
            predicted=onset(b[:,1:]);encoded=arrangement.encode(b[:,:1])
        values=dict(probability=predicted['logits'][0].float().sigmoid().cpu().numpy(),
                    offsets=predicted['offsets'][0].float().cpu().numpy(),acoustic=encoded[0].float().cpu().numpy())
        place_core(output,coverage,start,values)
    if not np.all(coverage==1):raise AssertionError('Full-song centers must cover each frame once')
    return output


def onset_times(prediction,threshold,duration):
    last=np.floor(duration*1000)/1000-.001
    return [min(last,round(float((i+prediction['offsets'][i])*.01),3)) for i in peaks(prediction['probability'],threshold,3)]


def pattern_events(token,lengths,start,duration,index):
    g,a,h=PATTERNS[token];events=[];base=f'dual-{index}'
    def add(kind,lanes,end,group=None):
        events.append(TrainingEvent(f'{base}-{len(events)}',kind,tuple(lanes),float(start),float(end),
                      (base,), 'dual-v1','predicted',kind=='extra_heavy',group))
    if g==5:add('gemini',LANES,start,base+'-gemini')
    else:
        for lane,s in zip(LANES,(g,a)):
            if s:add(NAMES[s],[lane],start,base+'-double-heavy' if g==a==2 else None)
    for side,lane in enumerate(LANES):
        if h&(1<<side):
            end=min(np.floor(duration*1000)/1000,round(start+float(lengths[side]),3))
            if end-start<.03:return None
            add('hold',[lane],end,base+'-double-hold' if h==3 else None)
    return events


@torch.no_grad()
def arrange(model,acoustic,times,duration,device='cuda'):
    model.eval();acoustic=torch.as_tensor(acoustic,device=device,dtype=torch.float32)
    times=sorted(set(round(t,3) for t in times if 0<=t<np.floor(duration*1000)/1000));hidden=None;previous=BOS;active=np.zeros(2)
    events=[];rejected=[];decisions=[]
    for i,t in enumerate(times):
        previous_gap=t-times[i-1] if i else t
        next_gap=times[i+1]-t if i+1<len(times) else duration-t
        context=torch.tensor(np.log1p([previous_gap,next_gap,*np.maximum(0,active-t)]),device=device,dtype=torch.float32)[None,None]
        frame=min(len(acoustic)-1,int(t*100+1e-7))
        out,hidden=model.sequence(acoustic[frame][None,None],torch.tensor([[previous]],device=device),context,hidden)
        logits=out['logits'][0,0];lengths=out['log_lengths'][0,0].clamp_max(8).expm1().cpu().numpy()
        recent=[e for e in events if e.end>t or abs(e.start-t)<=.0200001]
        chosen=None;raw=int(logits.argmax());reasons=[]
        for token in logits.argsort(descending=True).cpu().tolist():
            candidate=pattern_events(token,lengths,t,duration,i)
            issues=[{'code':'hold_under_30ms'}] if candidate is None else validate_generated_events(recent+candidate,duration)
            if not issues:chosen=(token,candidate);break
            if token==raw:reasons=issues
        if chosen is None:
            previous=BOS;rejected.append(dict(code='no_legal_pattern',index=i,start=t));continue
        token,candidate=chosen
        if token!=raw:rejected.append(dict(code='top_pattern_illegal',index=i,start=t,raw_pattern=raw,chosen=token,issues=reasons))
        previous=token;events.extend(candidate)
        for e in candidate:
            if e.kind=='hold':
                for lane in e.lanes:active[LANES.index(lane)]=e.end
        decisions.append(dict(start=t,token=token,raw_token=raw,probability=float(logits.softmax(-1)[token]),
                              hold_lengths_seconds=lengths.tolist()))
    issues=validate_generated_events(events,duration)
    if issues:raise AssertionError(f'Illegal decoded song: {issues[:2]}')
    return events,dict(candidate_count=len(times),rule_rejection_count=len(rejected),rejected=rejected,
                       decisions=decisions,rule_issues=[],bpm_used=False,grid_snapping=False,empty_gap_fill=False,precision_seconds=.001,
                       hold_pairing='predicted head plus per-lane regressed duration; full-song history, never window-edge tails')
