"""Reproduce bounded audio and event-history dependency diagnostics on CPU."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.dual_data import DualDataset,collate,PATTERNS,BOS
from mdac2.data_preparation.dual_training import make_model,batch_metrics,summarize
from mdac2.data_preparation.supervised_training import seed_all,atomic_json
from mdac2.data_preparation.audio_features import sha

def model_from(stage,path):
    model=make_model(stage);saved=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model']);model.eval();return model

@torch.no_grad()
def main(root,output):
    if output.exists():raise ValueError('Never overwrite diagnostics')
    seed_all(20260928);torch.set_num_threads(2)
    c=json.loads((root/'onset.json').read_text());d=DualDataset(c,'validation')
    onset_path=root/'formal-onset/best.pt';model=model_from('onset',onset_path)
    indices=np.linspace(0,len(d)-1,8,dtype=int).tolist()
    audio={k:[] for k in ('original','silence','vocals_only','accompaniment_only','audio_shift_2.37s')}
    for i in indices:
        batch=collate([d[i]])
        for name in audio:
            x=batch['audio'].clone()
            if name=='silence':x.zero_()
            elif name=='vocals_only':x[:,1].zero_()
            elif name=='accompaniment_only':x[:,0].zero_()
            elif name=='audio_shift_2.37s':x=x.roll(237,-1)
            audio[name].append(batch_metrics(model(x),batch,'onset'))
    audio={k:summarize(v,'onset') for k,v in audio.items()}
    c=json.loads((root/'arrangement.json').read_text());d=DualDataset(c,'train');batch=collate([d[0]]);lab=d.labels(0)
    layout_path=Path(c['overfit_report']).parent/'diagnostic.pt';model=model_from('arrangement',layout_path)
    n=int(batch['valid'].sum());target=batch['tokens'][0,:n].tolist();times=lab['times'][:n];acoustic=model.encode(batch['audio'])
    history={}
    for name,own_token,own_hold,mute in [('reference_history',False,False,False),('self_tokens_reference_holds',True,False,False),
             ('reference_tokens_self_holds',False,True,False),('self_history',True,True,False),('zero_audio_reference_history',False,False,True)]:
        hidden=None;previous=BOS;active=np.zeros(2);predicted=[]
        for j in range(n):
            frame=int(batch['frames'][0,j]);token=previous if own_token else int(batch['previous'][0,j])
            context=batch['context'][:,j:j+1].clone()
            if own_hold:context[0,0,2:]=torch.tensor(np.log1p(np.maximum(0,active-times[j])))
            x=torch.zeros_like(acoustic[:,frame:frame+1]) if mute else acoustic[:,frame:frame+1]
            out,hidden=model.sequence(x,torch.tensor([[token]]),context,hidden)
            previous=int(out['logits'][0,0].argmax());predicted.append(previous)
            length=out['log_lengths'][0,0].clamp_max(8).expm1().numpy()
            for side in range(2):
                if PATTERNS[previous][2]&(1<<side):active[side]=times[j]+length[side]
        correct=sum(p==t for p,t in zip(predicted,target))
        history[name]=dict(accuracy=correct/n,correct=correct,events=n)
    atomic_json(output,dict(audio=dict(role='validation',windows=indices,threshold=.5,metrics=audio,checkpoint_sha256=sha(onset_path)),
               history=dict(role='train',window=0,metrics=history,checkpoint_sha256=sha(layout_path)),
               source_sha256=sha(Path(__file__)),device='cpu float32',locked_test_opened=False,
               caveat='bounded input ablation, not source truth, listening acceptance or new test evaluation'))
    print(json.dumps(dict(audio={k:v['onset']['f1'] for k,v in audio.items()},history=history)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();main(args.experiment,args.output)
