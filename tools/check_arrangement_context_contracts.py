"""Check context padding, center stitching, target exclusion and legal decoding."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.arrangement_context_v2 import ContextArrangement,decode,predict
from mdac2.data_preparation.dual_data import PATTERNS
from mdac2.data_preparation.supervised_training import seed_all,atomic_json
from mdac2.data_preparation.training_tensors import slice_window
from mdac2.data_preparation.generated_constraints import validate_generated_events

seed_all(123);model=ContextArrangement().eval();checks={}
assert list(inspect.signature(model.forward).parameters)==['audio','frames','gaps','lengths']
checks['no_reference_tokens_or_hold_occupancy_in_model_interface']=True
with torch.no_grad():
    a=torch.rand(2,5,256);g=torch.rand(2,5,2)
    together=model.sequence(a,g,torch.tensor([3,5]))
    alone=model.sequence(a[:1,:3],g[:1,:3],torch.tensor([3]))
    error=float((together['logits'][0,:3]-alone['logits'][0]).abs().max())
    assert error<1e-5
    checks['padding_independence_max_logit_error']=error
    features=np.random.default_rng(123).random((3,96,2000),dtype=np.float32)
    times=[.5,2.,6.,12.,15.999,16.001,18.,19.];duration=20.
    stitched=predict(model,features,times,duration,device='cpu',amp=False)
    for start in (-800,800):
        frames=np.floor(np.asarray(times)*100+1e-7).astype(np.int64)
        take=np.flatnonzero((frames>=max(0,start))&(frames<start+3200))
        audio,_=slice_window(features[:1],start,time_axis=2)
        gaps=np.log1p(np.stack((np.diff(times,prepend=0),np.diff(times,append=duration)),1)).astype(np.float32)
        out=model(torch.from_numpy(audio[None]),torch.from_numpy((frames[take]-start)[None]),
                  torch.from_numpy(gaps[take][None]),torch.tensor([len(take)]))
        core=(frames[take]>=start+800)&(frames[take]<start+2400)
        assert np.array_equal(stitched['logits'][take[core]],out['logits'][0].numpy()[core])
    checks['inference_matches_training_window_centers_exactly']=True
    try:model.encode(torch.zeros(1,3,96,20))
    except ValueError:checks['rejects_vocals_accompaniment_streams']=True
    else:raise AssertionError('Wrong streams accepted')
tokens=[PATTERNS.index(p) for p in [(1,0,1),(5,5,0),(2,2,0)]]
logits=np.full((3,len(PATTERNS)),-20.);logits[np.arange(3),tokens]=20.
lengths=np.log1p(np.array([[20.,0.],[0.,0.],[0.,0.]]))
events,audit=decode(logits,lengths,[1.,23.,26.],30.)
assert not validate_generated_events(events,30.)
assert any(e.kind=='hold' and e.start==1. and e.end==21. for e in events)
assert sum(e.kind=='gemini' for e in events)==1
assert sum(e.kind=='heavy' for e in events)==2
assert audit['rule_rejection_count']==0
checks['cross_window_hold_plus_click_gemini_and_double_heavy_distinct']=True
atomic_json(Path(sys.argv[1]),dict(passed=True,checks=checks,parameters=sum(p.numel() for p in model.parameters()),
                                 train_only_or_synthetic=True,locked_test_opened=False))
print(json.dumps(checks))
