"""Train-only bounded probe: is same/switch predictable beyond event intervals?
Diagnostic only; never enters chart generation or replaces the arrangement model.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.arrangement_context_v4 import ContextDataset
from mdac2.data_preparation.arrangement_data_v3 import PATTERNS
from mdac2.data_preparation.supervised_training import atomic_json,seed_all,checkpoint
from mdac2.data_preparation.audio_features import sha

def build(config,role):
    data=ContextDataset(config,role);first={};xs=[];ys=[];records=[]
    for i,(ri,_) in enumerate(data.index):first.setdefault(ri,i)
    for ri,row in enumerate(data.rows):
        data.base[first[ri]];features=data.base.loaded[ri][0][0];lab=data.labels(ri)
        side=np.full(len(lab['tokens']),-1,np.int64)
        for i,t in enumerate(lab['tokens']):
            g,a,h=PATTERNS[t]
            if lab['valid'][i] and h==0 and ((g in (1,2) and a==0) or (a in (1,2) and g==0)):side[i]=int(a>0)
        ids=np.flatnonzero((side[1:]>=0)&(side[:-1]>=0))+1
        if not len(ids):continue
        needed=np.unique(np.concatenate((ids,ids-1)));means={};stds={}
        for i in needed:
            f=int(lab['frames'][i]);local=features[:,max(0,f-8):min(features.shape[1],f+9)].astype(np.float32)/10
            means[i]=local.mean(1);stds[i]=local.std(1)
        values=[]
        for i in ids:
            cur=means[i];prev=means[i-1]
            gap=lab['times'][i]-lab['times'][i-1]
            prevgap=lab['times'][i-1]-lab['times'][i-2] if i>1 else lab['times'][i-1]
            nextgap=lab['times'][i+1]-lab['times'][i] if i+1<len(side) else 0.
            values.append(np.concatenate((prev,cur,cur-prev,stds[i],np.log1p([gap,prevgap,nextgap]))))
        xs.append(np.asarray(values,np.float32));ys.append((side[ids]!=side[ids-1]).astype(np.float32));records.append(dict(chart_sha256=row['chart_sha256'],pairs=len(ids)))
    return np.concatenate(xs),np.concatenate(ys),records

def network(n):return nn.Sequential(nn.Linear(n,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
def metrics(logits,y):
    p=torch.sigmoid(logits);guess=p>=.5;truth=y>=.5;tp=int((guess&truth).sum());fp=int((guess&~truth).sum());fn=int((~guess&truth).sum())
    return dict(log_loss=float(nn.functional.binary_cross_entropy_with_logits(logits,y)),brier=float(((p-y)**2).mean()),accuracy=float((guess==truth).float().mean()),switch_f1=2*tp/max(1,2*tp+fp+fn),switch_rate=float(p.mean()),pairs=len(y))

def main(config,output):
    output.mkdir(parents=True,exist_ok=False);seed_all(20260930);torch.cuda.reset_peak_memory_stats()
    x,y,records=build(config,'train');center=x.mean(0);scale=x.std(0).clip(.03);X=torch.from_numpy((x-center)/scale).cuda();Y=torch.from_numpy(y).cuda()
    # Small training-only learning check, no validation opened until it passes.
    gate=network(X.shape[1]).cuda();opt=torch.optim.AdamW(gate.parameters(),lr=.003,weight_decay=0);initial=None;passed=False
    for step in range(601):
        pred=gate(X[:128]).squeeze(-1);loss=nn.functional.binary_cross_entropy_with_logits(pred,Y[:128]);initial=float(loss) if initial is None else initial
        if step>=100 and float(loss)<initial*.2 and metrics(pred.detach(),Y[:128])['accuracy']>=.95:passed=True;break
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(gate.parameters(),1.,error_if_nonfinite=True);opt.step()
    atomic_json(output/'overfit.json',dict(passed=passed,steps=step,initial_loss=initial,final_loss=float(loss),metrics=metrics(pred.detach(),Y[:128]),role='train',validation_opened=False))
    if not passed:raise RuntimeError('Transition probe training gate failed')
    vx,vy,vrecords=build(config,'validation');VX=torch.from_numpy((vx-center)/scale).cuda();VY=torch.from_numpy(vy).cuda();reports={}
    for mode in ('intervals_only','audio_and_intervals'):
        seed_all(20260930);take=slice(-3,None) if mode=='intervals_only' else slice(None);tr=X[:,take];va=VX[:,take]
        model=network(tr.shape[1]).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001);best=float('inf');stale=0;history=[]
        for epoch in range(30):
            model.train();order=torch.randperm(len(tr),device='cuda')
            for ids in order.split(512):
                opt.zero_grad();logits=model(tr[ids]).squeeze(-1);loss=nn.functional.binary_cross_entropy_with_logits(logits,Y[ids]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            model.eval()
            with torch.no_grad():m=metrics(model(va).squeeze(-1),VY)
            history.append(dict(epoch=epoch+1,**m))
            if m['log_loss']<best-1e-4:
                best=m['log_loss'];stale=0;checkpoint(output/(mode+'.pt'),dict(model=model.state_dict(),center=center,scale=scale,epoch=epoch+1,input_columns=mode,config=config))
            else:stale+=1
            if stale>=5:break
        saved=torch.load(output/(mode+'.pt'),weights_only=False);model.load_state_dict(saved['model']);model.eval()
        with torch.no_grad():
            result=metrics(model(va).squeeze(-1),VY)
            if mode=='audio_and_intervals':
                muted=VX.clone();muted[:,:-3]=torch.as_tensor(-center[:-3]/scale[:-3],device='cuda');result['silenced_audio']=metrics(model(muted).squeeze(-1),VY)
        reports[mode]=dict(best_epoch=saved['epoch'],parameters=sum(p.numel() for p in model.parameters()),validation=result,history=history)
    prior=float(y.mean());logit=torch.full_like(VY,float(np.log(prior/(1-prior))));baseline=metrics(logit,VY)
    atomic_json(output/'report.json',dict(status='completed',diagnostic_only=True,not_a_production_arranger=True,conditions=reports,constant_train_prior=baseline,
        train_charts=records,validation_charts=vrecords,manifest_sha256=sha(Path(config['manifest'])),train_pair_count=len(y),validation_pair_count=len(vy),source_sha256=sha(Path(__file__)),
        feature_definition='original mix +/-80ms mean and std, previous/current spectra, difference, previous/current/next event gaps; no lane history input',
        label='same/switch for directly adjacent valid single light/heavy events only',locked_test_opened=False,gpu_peak_bytes=torch.cuda.max_memory_allocated(),
        limits='reference onset/type pairs; excludes intervening Gemini/holds; does not establish full-generation performance or safe deployment'))
    print(json.dumps({k:v['validation'] for k,v in reports.items()}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(json.loads(a.config.read_text(encoding='utf-8')),a.output.resolve())
