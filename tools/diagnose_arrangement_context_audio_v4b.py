"""Fixed validation-window audio/interval ablations, with no parameter updates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.arrangement_context_training_v4b import identity,make_model,forward_loss,balance,batch_metrics,summarize
from mdac2.data_preparation.arrangement_context_v4 import ContextDataset,collate
from mdac2.data_preparation.supervised_training import seed_all,atomic_json
from mdac2.data_preparation.audio_features import sha

def main(root,output):
    if output.exists():raise ValueError('Never overwrite diagnostics')
    seed_all(20260928);c=json.loads((root/'arrangement.json').read_text(encoding='utf-8'))
    path=root/'formal-arrangement/best.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['identity']!=identity(c):raise ValueError('Identity mismatch')
    m=make_model('arrangement').cuda();m.load_state_dict(saved['model']);m.eval()
    data=ContextDataset(c,'validation');indices=np.linspace(0,len(data)-1,8,dtype=int).tolist()
    report={}
    for mode in ('original','silence','shift_2_37_seconds','zero_intervals'):
        metrics=[]
        for index in indices:
            b={k:v.cuda() for k,v in collate([data[index]]).items()}
            if mode=='silence':b['audio'].zero_()
            if mode=='shift_2_37_seconds':b['audio']=torch.roll(b['audio'],237,dims=-1)
            if mode=='zero_intervals':b['context'].zero_()
            with torch.no_grad():out,_=forward_loss(m,b,c,balance(c))
            metrics.append(batch_metrics(out,b,'arrangement'))
        report[mode]=summarize(metrics,'arrangement')
    atomic_json(output,dict(role='validation',fixed_window_indices=indices,checkpoint_sha256=sha(path),conditions=report,
                           parameter_updates=False,locked_test_opened=False,
                           caveat='Small deterministic subset; silence is out of distribution, shifts also change boundary content'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.experiment.resolve(),a.output.resolve())
