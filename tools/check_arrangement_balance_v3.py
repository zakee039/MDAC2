"""Gear removal preserves co-located notes; moment loss cannot reward Gemini filling."""
import sys,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.arrangement_data_v3 import derive,PATTERNS
from mdac2.data_preparation.arrangement_balance_v3 import balance_loss
from mdac2.data_preparation.supervised_training import atomic_json
original=dict(onsets=np.zeros((100,2,6),np.float32),onset_mask=np.ones((100,2,6),bool))
chart=dict(duration=1.,events=[dict(kind='gear',start=.1,end=.1,lanes=['ground']),dict(kind='gear',start=.2,end=.2,lanes=['air']),dict(kind='light',start=.2,end=.2,lanes=['ground'])])
lab=derive(chart,original)
assert lab['times'].tolist()==[.2] and PATTERNS[lab['tokens'][0]]==(1,0,0)
assert 'target' not in lab and 'mask' not in lab and all(4 not in p[:2] for p in PATTERNS)
tokens=torch.tensor([[PATTERNS.index((1,0,0)),PATTERNS.index((0,1,0))]])
valid=torch.ones_like(tokens,dtype=torch.bool)
matching=torch.full((1,2,len(PATTERNS)),-20.);matching.scatter_(-1,tokens[:,:,None],20.)
dual=torch.full_like(matching,-20.);dual[:,:,PATTERNS.index((5,5,0))]=20.
assert balance_loss(matching,tokens,valid)<1e-8
assert balance_loss(dual,tokens,valid)>.1
biased=torch.full_like(matching,-20.);biased[:,:,PATTERNS.index((1,0,0))]=20.
assert balance_loss(biased,tokens,valid)>.1
x=torch.randn_like(matching,requires_grad=True);loss=balance_loss(x,tokens,valid);loss.backward()
assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
assert balance_loss(x,tokens,torch.zeros_like(valid))==0
atomic_json(Path(sys.argv[1]),dict(passed=True,gear_only_removed=True,coincident_click_preserved=True,no_onset_negatives=True,
    gemini_cannot_satisfy_single_click_balance=True,finite_balance_gradients=True,patterns=len(PATTERNS),locked_test_opened=False))
print('V3 policy and balance checks passed')
