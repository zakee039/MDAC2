"""Audio-conditioned arrangement without reference or generated label feedback.

Training and inference share the same 32 s / central 16 s event windows.
Only mix audio and adjacent onset intervals enter the network. Output history
is used by the legal decoder, never supplied as a reference-label shortcut.
"""
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .dual_models import Block
from .arrangement_data_v3 import DualDataset, PATTERNS, collate as old_collate
from .dual_inference import pattern_events as legacy_pattern_events
from .dual_data import PATTERNS as LEGACY_PATTERNS

def pattern_events(token,*args):
    return legacy_pattern_events(LEGACY_PATTERNS.index(PATTERNS[token]),*args)
from .generated_constraints import validate_generated_events
from .training_tensors import slice_window, window_starts


class ContextArrangement(nn.Module):
    config = dict(architecture='mix-context-bigru-v3', streams=['mix'], acoustic_hidden=256,
                  dilations=[1,2,4,8,16,32], gru_hidden=384, gru_layers=2,
                  bidirectional=True, internal_dropout=0., input_dropout=.1,
                  window_seconds=32, center_seconds=16, label_history_input=False,
                  hold_history_input=False, bpm=False, grid=False,
                  conditioning='mix audio plus previous/next onset gaps only', gear='postprocess-only; disabled', balance='reference-window lane and single-click moments')

    def __init__(self):
        super().__init__()
        self.audio = nn.Sequential(nn.Conv1d(96,256,5,padding=2), nn.GELU(),
                                   *(Block(256,d) for d in self.config['dilations']))
        self.dropout = nn.Dropout(.1)
        self.gru = nn.GRU(258,384,num_layers=2,batch_first=True,bidirectional=True,dropout=0.)
        self.skip = nn.Linear(256,768)
        self.norm = nn.LayerNorm(768)
        self.pattern = nn.Linear(768,len(PATTERNS))
        self.local_pattern = nn.Linear(256,len(PATTERNS))
        self.length = nn.Linear(768,2)

    def encode(self,audio):
        if audio.ndim!=4 or audio.shape[1:3]!=(1,96):
            raise ValueError('Original mix only: B,1,96,T')
        return self.audio(audio[:,0]/10).transpose(1,2)

    def sequence(self,acoustic,gaps,lengths):
        if gaps.shape[-1]!=2:raise ValueError('Only adjacent time gaps; no reference occupancy')
        local=self.local_pattern(acoustic)
        packed=pack_padded_sequence(self.dropout(torch.cat((acoustic,gaps),-1)),
                                   lengths.cpu(),batch_first=True,enforce_sorted=False)
        encoded,_=self.gru(packed)
        z,_=pad_packed_sequence(encoded,batch_first=True,total_length=acoustic.shape[1])
        z=self.norm(z+self.skip(acoustic))
        return dict(logits=self.pattern(z),local_logits=local,
                    log_lengths=torch.nn.functional.softplus(self.length(z)))

    def forward(self,audio,frames,gaps,lengths):
        encoded=self.encode(audio)
        acoustic=encoded.gather(1,frames[:,:,None].expand(-1,-1,encoded.shape[-1]))
        return self.sequence(acoustic,gaps,lengths)


class ContextDataset(DualDataset):
    def __getitem__(self,index):
        sample=super().__getitem__(index)
        sample.pop('previous')
        sample['context']=sample['context'][:,:2].copy()
        # The old empty-window placeholder is harmless and receives no loss.
        sample['event_mask']=np.ones(len(sample['frames']),bool)
        ri,start=self.index[index]
        lab=self.labels(ri)
        if not np.any((lab['frames']>=max(0,start))&(lab['frames']<start+3200)):
            sample['event_mask'][:]=False
        sample['meta']=np.array([ri,start],np.int64)
        return sample


def collate(samples):
    result=old_collate([{k:v for k,v in s.items() if k!='meta'} for s in samples])
    result['sequence_lengths']=torch.tensor([len(s['frames']) for s in samples])
    result['meta']=torch.from_numpy(np.stack([s['meta'] for s in samples]))
    return result


def decode(logits,log_lengths,times,duration):
    """Greedy legal choice, without lane alternation, random sampling or fills."""
    logits=np.asarray(logits,np.float64);log_lengths=np.asarray(log_lengths,np.float64)
    if len(times)!=len(logits):raise ValueError('One distribution per onset required')
    if not np.isfinite(logits).all() or not np.isfinite(log_lengths).all():raise ValueError('Nonfinite decoder input')
    events=[];decisions=[];rejected=[]
    for i,(t,scores,ll) in enumerate(zip(times,logits,log_lengths)):
        t=round(float(t),3)
        if t<0 or t>=np.floor(duration*1000)/1000:continue
        lengths=np.expm1(np.clip(ll,0,8))
        recent=[e for e in events if e.end>t or abs(e.start-t)<=.0200001]
        raw=int(scores.argmax());chosen=None;reasons=[]
        for token in np.argsort(-scores,kind='stable'):
            candidate=pattern_events(int(token),lengths,t,duration,i)
            issues=[dict(code='hold_under_30ms')] if candidate is None else validate_generated_events(recent+candidate,duration)
            if not issues:chosen=(int(token),candidate);break
            if token==raw:reasons=issues
        if chosen is None:
            rejected.append(dict(code='no_legal_pattern',index=i,start=t));continue
        token,candidate=chosen
        if token!=raw:rejected.append(dict(code='top_pattern_illegal',start=t,raw_pattern=raw,chosen=token,issues=reasons))
        events.extend(candidate)
        probability=np.exp(scores-scores.max());probability/=probability.sum()
        decisions.append(dict(start=t,token=token,raw_token=raw,probability=float(probability[token]),
                              hold_lengths_seconds=lengths.tolist()))
    if validate_generated_events(events,duration):raise AssertionError('Illegal decoded chart')
    return events,dict(candidate_count=len(times),rule_rejection_count=len(rejected),rejected=rejected,
                       decisions=decisions,rule_issues=[],bpm_used=False,grid_snapping=False,empty_gap_fill=False,
                       reference_history_used=False,precision_seconds=.001,
                       hold_pairing='predicted duration with persistent full-song legal occupancy')


@torch.no_grad()
def predict(model,features,times,duration,device='cuda',amp=True):
    """Exact training windows, then unique central ownership of event outputs."""
    model.eval();times=np.asarray(times,np.float64)
    frames=np.floor(times*100+1e-7).astype(np.int64)
    gaps=np.log1p(np.stack((np.diff(times,prepend=0),np.diff(times,append=duration)),1)).astype(np.float32)
    outputs=dict(logits=np.zeros((len(times),len(PATTERNS)),np.float32),
                 log_lengths=np.zeros((len(times),2),np.float32))
    coverage=np.zeros(len(times),np.int32)
    for start in window_starts(int(np.ceil(duration*100-1e-8))):
        take=np.flatnonzero((frames>=max(0,start))&(frames<start+3200))
        if not len(take):continue
        audio,_=slice_window(features[:1],start,time_axis=2)
        b=torch.as_tensor(audio[None],device=device,dtype=torch.float32)
        f=torch.as_tensor((frames[take]-start)[None],device=device)
        g=torch.as_tensor(gaps[take][None],device=device)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
            out=model(b,f,g,torch.tensor([len(take)]))
        core=(frames[take]>=start+800)&(frames[take]<start+2400)
        for key in outputs:outputs[key][take[core]]=out[key][0].float().cpu().numpy()[core]
        coverage[take[core]]+=1
    if not np.all(coverage==1):raise AssertionError('Every onset requires exactly one center owner')
    return outputs


def collapse_stats(tokens,times):
    tokens=list(map(int,tokens));times=list(map(float,times));longest=0;longest_seconds=0.;run=0;first=0
    for i,t in enumerate(tokens):
        if i==0 or t!=tokens[i-1]:run=1;first=i
        else:run+=1
        longest=max(longest,run);longest_seconds=max(longest_seconds,times[i]-times[first])
    counts=Counter(tokens)
    return dict(events=len(tokens),dominant_pattern_fraction=max(counts.values(),default=0)/max(1,len(tokens)),
                longest_identical_run=longest,longest_identical_seconds=longest_seconds,
                pattern_counts={str(k):v for k,v in counts.items()})
