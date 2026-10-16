"""Untrained joint chart model: one shared backbone, multiple task heads.

Torch is optional and imported only by the isolated preparation/training runtime.
This module contains no optimizer, training loop, checkpoint download or save.
"""
import torch
from torch import nn


class TemporalBlock(nn.Module):
    def __init__(self,width,dilation):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv1d(width,width,5,padding=2*dilation,dilation=dilation),
                                  nn.GELU(),nn.Conv1d(width,width,1),nn.Dropout(.1))
        self.norm=nn.LayerNorm(width)

    def forward(self,x):
        return self.norm((x+self.layers(x)).transpose(1,2)).transpose(1,2)


class JointChartModel(nn.Module):
    config={'architecture':'joint-tcn-v1','streams':3,'mel_bins':96,'stream_width':64,
            'hidden_width':128,'dilations':[1,2,4,8,16,32,64,128],'kernel_size':5,
            'context_seconds_each_side':8,'window_seconds':32,'supervised_core_seconds':16,
            'classes':['light','heavy','extra_heavy','gemini','hold','gear'],
            'shared_stream_encoder':True,'one_trainable_model':True,
            'vocal_gate':'soft energy ratio prior; activity is not an onset label'}

    def __init__(self):
        super().__init__()
        self.stream_encoder=nn.Sequential(nn.Linear(96,64),nn.GELU(),nn.LayerNorm(64))
        self.fusion=nn.Linear(64*3+4+1,128)
        self.temporal=nn.Sequential(*(TemporalBlock(128,d) for d in self.config['dilations']))
        self.onsets=nn.Linear(128,12)
        self.offsets=nn.Linear(128,12)
        self.hold_ends=nn.Linear(128,2)
        self.hold_end_offsets=nn.Linear(128,2)
        self.occupancy=nn.Linear(128,2)

    def forward(self,features,timing):
        if features.ndim!=4 or features.shape[1:3]!=(3,96):
            raise ValueError('Expected B x 3 x 96 x T audio features')
        if timing.shape!=(features.shape[0],features.shape[-1],4):
            raise ValueError('Expected B x T x 4 beat/difficulty features')
        # Conservative soft preference: retain accompaniment even under vocals.
        # The network still decides whether ANY event exists at this time.
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
                'occupancy_logits':self.occupancy(hidden), 'vocal_activity_prior':activity}
