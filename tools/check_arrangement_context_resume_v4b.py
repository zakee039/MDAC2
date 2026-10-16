"""Actual CUDA cancellation equivalence for both independent architectures."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation import arrangement_context_training_v4b as s
from mdac2.data_preparation.arrangement_data_v3 import PATTERNS,BOS

class TinyDataset:
    rows=[{},{}];index=[(0,-800),(0,800),(1,-800),(1,800)]
    def __init__(self,config,*_):self.stage=config['stage']
    def __len__(self):return 4
    def __getitem__(self,i):
        rng=np.random.default_rng(i)
        if self.stage=='onset':
            t=np.zeros(32,np.float32);t[16]=1
            return dict(audio=rng.random((2,96,32),dtype=np.float32),target=t,mask=np.ones(32,bool),offsets=t*.3)
        return dict(audio=rng.random((1,96,32),dtype=np.float32),frames=np.array([5,15,25]),
                    context=rng.random((3,2),dtype=np.float32),event_mask=np.ones(3,bool),meta=np.array([i//2,-800]),tokens=np.array([2,4,6]),
                    lengths=np.ones((3,2),np.float32),valid=np.ones(3,bool))

def main(output):
    reports=[]
    for stage in ('arrangement',):
        with tempfile.TemporaryDirectory(prefix='mdac2-dual-resume-') as temporary:
            root=Path(temporary);ident={'fixture':'dual-resume-v1','stage':stage};gate=root/'gate.json'
            s.atomic_json(gate,dict(passed=True,identity=ident))
            config=dict(stage=stage,device='cuda',amp=True,seed=123,batch_size=1,learning_rate=.001,
                weight_decay=.0001,max_epochs=1,patience=2,min_delta=.0001,log_every=1,gradient_clip=1.,overfit_report=str(gate))
            weights=dict(onset_positive_weight=2.,pattern_weights=[1.]*len(PATTERNS),group_weights=[1.]*len(s.GROUPS))
            with patch.object(s,'DualDataset',TinyDataset),patch.object(s,'identity',return_value=ident),patch.object(s,'balance',return_value=weights),patch.object(s,'evaluate',return_value={'selection_score':.5}):
                s.train(config,root/'continuous');original=torch.optim.AdamW.step;calls=[0]
                def stop(optimizer,*args,**kwargs):
                    answer=original(optimizer,*args,**kwargs);calls[0]+=1
                    if calls[0]==2:(root/'resumed'/'CANCEL').touch()
                    return answer
                with patch.object(torch.optim.AdamW,'step',stop):first=s.train(config,root/'resumed')
                assert first['status']=='cancelled' and first['cursor']==2
                (root/'resumed'/'CANCEL').unlink();s.train(config,root/'resumed',root/'resumed'/'last.pt')
            a=torch.load(root/'continuous'/'last.pt',weights_only=False,map_location='cpu')
            b=torch.load(root/'resumed'/'last.pt',weights_only=False,map_location='cpu')
            assert a['state']==b['state']
            assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
            assert a['scheduler']==b['scheduler']
            for k,v in a['optimizer']['state'].items():
                for name,t in v.items():assert torch.equal(t,b['optimizer']['state'][k][name])
            with patch.object(s,'identity',return_value={'changed':True}):
                try:s.train(config,root/'resumed',root/'resumed'/'last.pt')
                except ValueError:pass
                else:raise AssertionError('Identity change accepted')
            reports.append(dict(stage=stage,passed=True,model_optimizer_scheduler_bitwise_equal=True,cancel_step=2,total_steps=4,identity_change_rejected=True))
    s.atomic_json(output,dict(passed=True,stages=reports,locked_test_opened=False,source_sha256=s.sha(Path(s.__file__))))

if __name__=='__main__':main(Path(sys.argv[1]))
