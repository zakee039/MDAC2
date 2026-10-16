"""Original-audio context plus genuinely self-generated causal arrangement history."""
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from .dual_models import Block
from .arrangement_context_v4 import ContextDataset as BaseDataset,collate,decode,collapse_stats
from .arrangement_data_v3 import PATTERNS,BOS
from .arrangement_hierarchy_v4 import GROUPS,hierarchical_scores
from .training_tensors import slice_window,window_starts

class ContextDataset(BaseDataset):
    def __getitem__(self,index):
        s=super().__getitem__(index);ri,start=self.index[index];lab=self.labels(ri)
        take=(lab['frames']>=max(0,start))&(lab['frames']<start+3200)
        s['times']=(lab['times'][take]-start*.01).astype(np.float32) if take.any() else np.zeros(1,np.float32)
        return s

class ContextArrangement(nn.Module):
    config=dict(architecture='mix-bigru512-causal-history256-v6',streams=['mix'],acoustic_hidden=256,
        dilations=[1,2,4,8,16,32],gru_hidden=512,gru_layers=2,bidirectional=True,
        history_hidden=256,history_embedding=64,window_seconds=32,center_seconds=16,
        label_history_input=False,generated_history_input=True,generated_hold_occupancy=True,
        teacher_forcing=0.,bpm=False,grid=False,gear='disabled',balance='none',
        orientation='relative mirror direction conditioned on previous generated single click')
    def __init__(self):
        super().__init__()
        self.audio=nn.Sequential(nn.Conv1d(96,256,5,padding=2),nn.GELU(),*(Block(256,d) for d in self.config['dilations']))
        self.gru=nn.GRU(258,512,num_layers=2,batch_first=True,bidirectional=True)
        self.skip=nn.Linear(256,1024);self.norm=nn.LayerNorm(1024);self.dropout=nn.Dropout(.1)
        self.embedding=nn.Embedding(len(PATTERNS)+1,64)
        self.history=nn.GRUCell(1024+64+4,256)
        self.output_norm=nn.LayerNorm(1280)
        self.group=nn.Linear(1280,len(GROUPS));self.orientation=nn.Linear(1280,len(GROUPS)*2)
        self.local_group=nn.Linear(256,len(GROUPS));self.length=nn.Linear(1280,2)
        self.register_buffer('patterns',torch.tensor(PATTERNS,dtype=torch.long),persistent=False)
    def encode(self,audio):
        if audio.ndim!=4 or audio.shape[1:3]!=(1,96):raise ValueError('Original mix only')
        return self.audio(audio[:,0]/10).transpose(1,2)
    def context(self,audio,frames,gaps,lengths):
        encoded=self.encode(audio);a=encoded.gather(1,frames[:,:,None].expand(-1,-1,256))
        packed=pack_padded_sequence(self.dropout(torch.cat((a,gaps),-1)),lengths.cpu(),batch_first=True,enforce_sorted=False)
        z,_=self.gru(packed);z,_=pad_packed_sequence(z,batch_first=True,total_length=frames.shape[1])
        return self.norm(z+self.skip(a)),self.local_group(a)
    def state(self,b,device):
        return dict(hidden=torch.zeros(b,256,device=device),previous=torch.full((b,),BOS,device=device,dtype=torch.long),
            active=torch.zeros(b,2,device=device),instant=torch.full((b,2),-1e6,device=device),special=torch.full((b,),-1e6,device=device))
    def step(self,z,gaps,t,state,history_enabled=True):
        hidden=state['hidden'];previous=state['previous'];active=state['active']
        if not history_enabled:hidden=torch.zeros_like(hidden);previous=torch.full_like(previous,BOS);active=torch.zeros_like(active)
        context=torch.cat((gaps,torch.log1p((active-t[:,None]).clamp_min(0))),-1)
        h=self.history(torch.cat((z,self.embedding(previous),context),-1),hidden)
        out=self.output_norm(torch.cat((z,h),-1));group=self.group(out)
        relative=self.orientation(out).reshape(-1,len(GROUPS),2)
        safe=previous.clamp_max(len(PATTERNS)-1);p=self.patterns[safe]
        single=(previous!=BOS)&(p[:,2]==0)&(((p[:,0]>0)&(p[:,1]==0)&(p[:,0]<=2))|((p[:,1]>0)&(p[:,0]==0)&(p[:,1]<=2)))
        # Canonical single-air orientation is 0, ground is 1; relation 0 keeps the previous side.
        flip=single&(p[:,0]>0)
        orientation=torch.where(flip[:,None,None],relative.flip(-1),relative)
        ll=torch.nn.functional.softplus(self.length(out)).clamp(min=float(np.log1p(.03)),max=8)
        return dict(group_logits=group,orientation_logits=orientation,logits=hierarchical_scores(group,orientation),log_lengths=ll),h
    def choose(self,out,t,state):
        p=self.patterns;attacks=p[:,:2]>0;holds=torch.stack((p[:,2]&1>0,p[:,2]&2>0),-1)
        occupied=state['active']>t[:,None]+1e-6
        close=(t[:,None]-state['instant'])<=.0200001
        extra=(p[:,:2]==3).any(-1)
        illegal=(holds[None]&occupied[:,None]).any(-1)|(attacks[None]&close[:,None]).any(-1)
        illegal|=extra[None]&(occupied.any(-1)|(close.any(-1)))[:,None]
        illegal|=((t-state['special'])<=.0200001)[:,None]
        scores=out['logits'].detach().masked_fill(illegal,-1e9)
        choice=scores.argmax(-1);choice=torch.where(illegal.all(-1),torch.full_like(choice,BOS),choice)
        return choice
    def advance(self,state,h,choice,ll,t,live):
        p=self.patterns[choice.clamp_max(len(PATTERNS)-1)];real=live&(choice!=BOS)
        holds=torch.stack((p[:,2]&1>0,p[:,2]&2>0),-1)&real[:,None]
        attacks=(p[:,:2]>0)&real[:,None]
        active=torch.where(holds,t[:,None]+ll.detach().expm1(),state['active'])
        return dict(hidden=torch.where(live[:,None],h,state['hidden']),previous=torch.where(live,choice,state['previous']),
            active=active,instant=torch.where(attacks,t[:,None],state['instant']),
            special=torch.where(real&(p[:,:2]==3).any(-1),t,state['special']))
    def rollout(self,z,gaps,times,lengths,history_enabled=True):
        state=self.state(len(z),z.device);outputs=[];choices=[]
        for i in range(z.shape[1]):
            t=times[:,i];out,h=self.step(z[:,i],gaps[:,i],t,state,history_enabled)
            chosen=self.choose(out,t,state);live=i<lengths.to(z.device)
            state=self.advance(state,h,chosen,out['log_lengths'],t,live);outputs.append(out);choices.append(chosen)
        return {**{k:torch.stack([o[k] for o in outputs],1) for k in outputs[0]},'chosen_tokens':torch.stack(choices,1)}
    def forward(self,audio,frames,gaps,lengths,times=None,history_enabled=True):
        z,local=self.context(audio,frames,gaps,lengths)
        times=frames.float()*.01 if times is None else times
        out=self.rollout(z,gaps,times,lengths,history_enabled);out['local_group_logits']=local;return out

@torch.no_grad()
def predict(model,features,times,duration,device='cuda',amp=True,history_enabled=True):
    model.eval();times=np.asarray(times,np.float64);frames=np.floor(times*100+1e-7).astype(np.int64)
    if not len(times):return dict(logits=np.zeros((0,len(PATTERNS)),np.float32),log_lengths=np.zeros((0,2),np.float32))
    gaps=np.log1p(np.stack((np.diff(times,prepend=0),np.diff(times,append=duration)),1)).astype(np.float32)
    contexts=np.zeros((len(times),1024),np.float32);coverage=np.zeros(len(times),np.int32)
    for start in window_starts(int(np.ceil(duration*100-1e-8))):
        take=np.flatnonzero((frames>=max(0,start))&(frames<start+3200))
        if not len(take):continue
        audio,_=slice_window(features[:1],start,time_axis=2)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
            z,_=model.context(torch.as_tensor(audio[None],device=device,dtype=torch.float32),torch.as_tensor((frames[take]-start)[None],device=device),torch.as_tensor(gaps[take][None],device=device),torch.tensor([len(take)]))
        core=(frames[take]>=start+800)&(frames[take]<start+2400)
        contexts[take[core]]=z[0].float().cpu().numpy()[core];coverage[take[core]]+=1
    if not np.all(coverage==1):raise AssertionError('Central ownership mismatch')
    # No history resets at audio-window boundaries: one chronological full-song rollout.
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
        out=model.rollout(torch.as_tensor(contexts[None],device=device),torch.as_tensor(gaps[None],device=device),torch.as_tensor(times[None],device=device,dtype=torch.float32),torch.tensor([len(times)]),history_enabled)
    logits=out['logits'][0].float().cpu().numpy();ll=out['log_lengths'][0].float().cpu().numpy()
    chosen=out['chosen_tokens'][0].cpu().numpy()
    # Bind final legal decoder to the exact choices fed back during generation.
    for i,token in enumerate(chosen):
        if token!=BOS:logits[i,token]=max(logits[i].max()+1e-4,logits[i,token])
    events,audit=decode(logits,ll,times,duration)
    actual={round(d['start'],3):d['token'] for d in audit['decisions']}
    mismatch=[i for i,t in enumerate(times) if actual.get(round(float(t),3),BOS)!=chosen[i]]
    if mismatch:raise AssertionError(f'History/actual decoder mismatch at {mismatch[:8]}')
    return dict(logits=logits,log_lengths=ll)
