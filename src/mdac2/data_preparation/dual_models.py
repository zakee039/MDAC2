"""Independent absolute-time onset and autoregressive arrangement baselines."""
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(width, width, 5, padding=2*dilation, dilation=dilation),
                                  nn.GELU(), nn.Conv1d(width, width, 1), nn.Dropout(.1))
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        normalized=self.norm(x.transpose(1,2)).transpose(1,2)
        return x+.1*self.conv(normalized)


class OnsetModel(nn.Module):
    config = dict(architecture='dual-onset-tcn-v1', streams=['vocals', 'accompaniment'],
                  hidden=256, dilations=[1,2,4,8,16,32,64,128], mel_bins=96,
                  vocal_gain=1.25, accompaniment_gain=.75, hop_seconds=.01, residual='pre_norm_scale_0.1',
                  label='new start in center 10ms cell, lane/type union', bpm=False)

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(96,128,5,padding=2),nn.GELU(),
                                     nn.Conv1d(128,128,3,padding=1),nn.GELU())
        self.fusion = nn.Conv1d(256,256,1)
        self.temporal = nn.Sequential(*(Block(256,d) for d in self.config['dilations']))
        self.head = nn.Conv1d(256,2,1)

    def forward(self, audio):
        if audio.ndim != 4 or audio.shape[1:3] != (2,96):
            raise ValueError('Onset accepts vocals/accompaniment ONLY: B,2,96,T')
        b,_,_,t = audio.shape
        z = self.encoder(audio.reshape(b*2,96,t)/10).reshape(b,2,128,t)
        z = torch.cat((z[:,0]*1.25,z[:,1]*.75),1)
        out = self.head(self.temporal(self.fusion(z))).transpose(1,2)
        return dict(logits=out[:,:,0], offsets=out[:,:,1].sigmoid())


class ArrangementModel(nn.Module):
    config = dict(architecture='dual-arrangement-gru-v1', streams=['mix'], mel_bins=96,
                  acoustic_hidden=128, dilations=[1,2,4,8,16,32], gru_hidden=512,
                  gru_layers=4, gru_dropout=0., input_dropout=.1, history_embedding=64, history_dropout=.2,
                  bpm=False, residual='pre_norm_scale_0.1',
                  conditioning='previous pattern, adjacent time gaps, previous active holds')

    def __init__(self, vocabulary_size):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.audio = nn.Sequential(nn.Conv1d(96,128,5,padding=2),nn.GELU(),
                                   *(Block(128,d) for d in self.config['dilations']))
        self.embedding = nn.Embedding(vocabulary_size+1,64)
        # cuDNN's internal RNN dropout generator is not in torch RNG snapshots.
        # Explicit input dropout is reproducibly restored by the trainer.
        self.input_dropout=nn.Dropout(.1)
        self.gru = nn.GRU(128+64+4,512,num_layers=4,batch_first=True,dropout=0.)
        self.pattern = nn.Linear(512,vocabulary_size)
        self.length = nn.Linear(512,2)

    def encode(self, audio):
        if audio.ndim != 4 or audio.shape[1:3] != (1,96):
            raise ValueError('Arrangement accepts original mix ONLY: B,1,96,T')
        return self.audio(audio[:,0]/10).transpose(1,2)

    def sequence(self, acoustic, previous, context, hidden=None):
        if self.training:
            previous=torch.where(torch.rand_like(previous,dtype=torch.float32)<.2,self.vocabulary_size,previous)
        z = torch.cat((acoustic,self.embedding(previous),context),-1)
        z, hidden = self.gru(self.input_dropout(z),hidden)
        return dict(logits=self.pattern(z), log_lengths=torch.nn.functional.softplus(self.length(z))), hidden

    def forward(self, audio, frames, previous, context):
        acoustic = self.encode(audio)
        gathered = acoustic.gather(1,frames[:,:,None].expand(-1,-1,acoustic.shape[-1]))
        return self.sequence(gathered,previous,context)[0]
