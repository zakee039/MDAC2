"""Versioned derivative labels, independent of source tensors and musical timing."""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
import torch
from .audio_features import sha
from .window_dataset import WindowDataset
from .training_tensors import slice_window

POLICY = 'dual-absolute-time-v1'
NAMES = ('none','light','heavy','extra_heavy','gear','gemini')
# A token includes both lanes and hold starts, so uncertainty cannot create two
# independent lane decisions. Gemini is one token, two heavy attacks another.
PATTERNS = tuple((g,a,h) for g in range(6) for a in range(6) for h in range(4)
                 if (g or a or h) and ((g==5 and a==5) or (g!=5 and a!=5))
                 and not ((g==3 and (a or h)) or (a==3 and (g or h)))
                 and not ((g==4 and h&1) or (a==4 and h&2)))
BOS = len(PATTERNS)


def derive(chart, original):
    n = len(original['onsets'])
    target = np.any(original['onsets']>.5,axis=(1,2)).astype(np.float32)
    mask = np.all(original['onset_mask'],axis=(1,2)) | (target>.5)
    offsets = np.zeros(n,np.float32)
    groups = {}
    for event in chart['events']:
        frame = min(n-1,int(event['start']*100+1e-7))
        groups.setdefault(frame,[]).append(event)
    frames=[]; times=[]; tokens=[]; lengths=[]; valid=[]; context=[]; previous=[]
    active = np.zeros(2)
    all_frames = sorted(groups)
    previous_token = BOS
    for i, f in enumerate(all_frames):
        events=groups[f]; t=float(np.mean([e['start'] for e in events]))
        offsets[f]=np.clip(t*100-f,0,1)
        state=[0,0]; hold=0; duration=np.zeros(2,np.float32); ok=True
        for e in events:
            for lane in e['lanes']:
                side=('ground','air').index(lane)
                if e['kind']=='hold':
                    if hold & (1<<side): ok=False
                    hold |= 1<<side
                    duration[side]=e['end']-e['start']
                else:
                    if state[side]: ok=False
                    state[side]=NAMES.index(e['kind'])
        pattern=(*state,hold)
        ok = ok and pattern in PATTERNS
        token=PATTERNS.index(pattern) if ok else 0
        prev_gap=t-times[-1] if times else t
        next_gap=(float(np.mean([e['start'] for e in groups[all_frames[i+1]]]))-t
                  if i+1<len(all_frames) else chart['duration']-t)
        context.append(np.log1p([max(0,prev_gap),max(0,next_gap),*np.maximum(0,active-t)]))
        frames.append(f);times.append(t);tokens.append(token);lengths.append(np.log1p(duration));valid.append(ok)
        previous.append(previous_token)
        # Unknown composition resets token history; never invent a clean label.
        previous_token=token if ok else BOS
        for e in events:
            if e['kind']=='hold':
                for lane in e['lanes']: active[('ground','air').index(lane)]=e['end']
    return dict(target=target,mask=mask,offsets=offsets,frames=np.asarray(frames,np.int64),
                times=np.asarray(times,np.float64),tokens=np.asarray(tokens,np.int64),
                lengths=np.asarray(lengths,np.float32).reshape(-1,2),valid=np.asarray(valid,bool),
                context=np.asarray(context,np.float32).reshape(-1,4),previous=np.asarray(previous,np.int64))


class DualDataset:
    def __init__(self,config,role):
        if role not in ('train','validation'):
            raise ValueError('Dual baseline never opens previously used locked test')
        self.stage=config['stage']; self.derived=Path(config['derived'])
        self.base=WindowDataset(config['manifest'],config['cache'],role,tensor_cache=config['tensors'])
        self.rows=self.base.rows;self.index=self.base.index;self.root=self.base.root
        report=json.loads((self.derived/'report.json').read_text(encoding='utf-8'))
        if report['manifest_sha256']!=sha(Path(config['manifest'])) or report['policy']!=POLICY:
            raise ValueError('Derived identity mismatch')
        if report['implementation_sha256']!=sha(Path(__file__)):
            raise ValueError('Derived labels require matching implementation')
        self.records={r['chart_sha256']:r for r in report['charts']};self.loaded=OrderedDict();self.verified=set()

    def __len__(self):return len(self.index)

    def labels(self,ri):
        key=self.rows[ri]['chart_sha256']
        if key not in self.loaded:
            rec=self.records[key];path=self.derived/rec['file']
            if key not in self.verified:
                if sha(path)!=rec['sha256']:raise ValueError('Corrupt derived target')
                self.verified.add(key)
            with np.load(path,allow_pickle=False) as a:self.loaded[key]={k:a[k] for k in a.files}
            while len(self.loaded)>2:self.loaded.popitem(last=False)
        self.loaded.move_to_end(key)
        return self.loaded[key]

    def __getitem__(self,i):
        sample=self.base[i]  # Immutable cache validation is shared with preparation.
        ri,start=self.index[i]; lab=self.labels(ri)
        if self.stage=='onset':
            answer={'audio':sample['features'][1:].copy()}
            for key in ('target','mask','offsets'):
                value,core=slice_window(lab[key],start)
                answer[key]=value & core if key=='mask' else value
            return answer
        take=np.flatnonzero((lab['frames']>=max(0,start))&(lab['frames']<start+3200))
        if not len(take):
            return dict(audio=sample['features'][:1].copy(),frames=np.zeros(1,np.int64),
                previous=np.full(1,BOS,np.int64),context=np.zeros((1,4),np.float32),tokens=np.zeros(1,np.int64),
                lengths=np.zeros((1,2),np.float32),valid=np.zeros(1,bool))
        frames=lab['frames'][take]-start
        return dict(audio=sample['features'][:1].copy(),frames=frames,
                    previous=lab['previous'][take].copy(),context=lab['context'][take].copy(),
                    tokens=lab['tokens'][take].copy(),lengths=lab['lengths'][take].copy(),
                    valid=lab['valid'][take]&(frames>=800)&(frames<2400))


def collate(samples):
    if 'target' in samples[0]:
        return {k:torch.from_numpy(np.stack([s[k] for s in samples])) for k in samples[0]}
    maximum=max(len(s['frames']) for s in samples);answer={}
    for k in samples[0]:
        values=[]
        for s in samples:
            a=s[k]
            if k!='audio':
                pad=np.full((maximum-len(a),)+a.shape[1:],BOS if k=='previous' else 0,dtype=a.dtype)
                a=np.concatenate((a,pad),0)
            values.append(a)
        answer[k]=torch.from_numpy(np.stack(values))
    return answer
