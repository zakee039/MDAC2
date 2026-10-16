"""One joint TCN with an onset head and mutually exclusive lane/type patterns.

Adds conditional supervision to the established backbone. Prepared targets are
read-only: ambiguous simultaneous source classes are masked for the new pattern
loss rather than silently removed from source charts.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .joint_model import JointChartModel
from .joint_losses import joint_losses,masked_mean
from .audio_features import sha
from .structured_patterns import STATE_NAMES,PATTERNS

ROOT=Path(__file__).resolve().parents[3]
INITIALIZER=ROOT/'tools/private/community/supervised-joint-v1-20260927/formal-lr3e4/best.pt'
INITIALIZER_SHA='78407992ac57b4c7953b77c49cb8e85025fbed3e06015380b07de13e943bb3a6'


class StructuredJointModel(JointChartModel):
    config={**JointChartModel.config,'architecture':'joint-tcn-structured-v2',
            'patterns':[list(p) for p in PATTERNS],'pattern_states':list(STATE_NAMES),'initializer_sha256':INITIALIZER_SHA,
            'one_trainable_model':True,'additional_heads':['event','pattern']}

    def __init__(self,initialize=True):
        super().__init__()
        self.event_head=nn.Linear(128,1)
        self.pattern_head=nn.Linear(128,len(PATTERNS))
        if initialize:
            if sha(INITIALIZER)!=INITIALIZER_SHA:
                raise ValueError('Structured model initializer identity changed')
            saved=torch.load(INITIALIZER,map_location='cpu',weights_only=False)
            missing,unexpected=self.load_state_dict(saved['model'],strict=False)
            if set(missing)!={'event_head.weight','event_head.bias','pattern_head.weight','pattern_head.bias'} or unexpected:
                raise ValueError('Unexpected initializer architecture')

    def forward(self,features,timing):
        if features.ndim!=4 or features.shape[1:3]!=(3,96) or timing.shape!=(features.shape[0],features.shape[-1],4):
            raise ValueError('Structured joint input contract mismatch')
        energy=torch.expm1(features.float().clamp(0,30)).mean(dim=2)
        ratio=energy[:,1]/(energy[:,0]+1e-5)
        activity=(ratio/(ratio+.1))*(energy[:,1]/(energy[:,1]+.01))
        encoded=self.stream_encoder(features.permute(0,1,3,2))
        weights=torch.stack((torch.ones_like(activity),.5+.5*activity,1-.5*activity),dim=1)
        encoded=encoded*weights[:,:,:,None]
        fused=torch.cat((encoded.permute(0,2,1,3).flatten(2),timing,activity[:,:,None]),dim=-1)
        hidden=self.temporal(self.fusion(fused).transpose(1,2)).transpose(1,2)
        b,t,_=hidden.shape
        return {'onset_logits':self.onsets(hidden).reshape(b,t,2,6),
                'offsets':self.offsets(hidden).sigmoid().reshape(b,t,2,6),
                'hold_end_logits':self.hold_ends(hidden),
                'hold_end_offsets':self.hold_end_offsets(hidden).sigmoid(),
                'occupancy_logits':self.occupancy(hidden),'vocal_activity_prior':activity,
                'event_logits':self.event_head(hidden).squeeze(-1),
                'pattern_logits':self.pattern_head(hidden)}


def structured_targets(target):
    instant_indices=[0,1,2,3,5]
    y=target['onsets'][...,instant_indices]>.5
    valid=target['onset_mask'][...,instant_indices].all(dim=(-1,-2))
    presence=y.any(dim=(-1,-2)).float()
    # Lane state: light/heavy/extra-heavy/gear; Gemini is handled separately.
    ordinary=target['onsets'][...,[0,1,2,5]]>.5
    states=(ordinary*torch.arange(1,5,device=y.device)).sum(-1)
    states=torch.where(y[...,3],torch.full_like(states,5),states)
    clean=y.sum(-1).le(1).all(-1)
    label=torch.zeros_like(states[...,0],dtype=torch.long)
    matched=torch.zeros_like(valid)
    for i,(ground,air) in enumerate(PATTERNS):
        select=(states[...,0]==ground)&(states[...,1]==air)&clean
        label=torch.where(select,i,label)
        matched|=select
    return dict(event_target=presence,event_mask=valid,pattern_label=label,
                pattern_mask=valid&matched&(presence>0),ambiguous_pattern=valid&(presence>0)&~matched)


def structured_losses(output,target,*,event_positive_weight,**balance):
    losses=joint_losses(output,target,**balance)
    derived=structured_targets(target)
    losses['event']=masked_mean(F.binary_cross_entropy_with_logits(output['event_logits'],derived['event_target'],
        pos_weight=event_positive_weight,reduction='none'),derived['event_mask'])
    losses['pattern']=masked_mean(F.cross_entropy(output['pattern_logits'].transpose(1,2),
        derived['pattern_label'],reduction='none'),derived['pattern_mask'])
    losses['total']=losses['total']+losses['event']+.25*losses['pattern']
    return losses
