"""Two separately seeded, bounded trainers; no shared trainable state."""
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .audio_features import sha
from .dual_data import DualDataset, PATTERNS, collate
from .dual_models import OnsetModel, ArrangementModel
from .supervised_training import (atomic_json, checkpoint, seed_all, rng_state, restore_rng,
                                  epoch_order, runtime, data_identity as preparation_identity)
from .joint_decoding import peaks
from .joint_evaluation import counts, result


def identity(config):
    original=preparation_identity(config)
    source=Path(__file__).parent
    return dict(preparation=original, stage=config['stage'],
                derived_sha256=sha(Path(config['derived'])/'report.json'),
                code={name:sha(source/name) for name in ('dual_models.py','dual_data.py','dual_training.py')},
                model_config=(OnsetModel if config['stage']=='onset' else ArrangementModel).config,
                patterns=[list(p) for p in PATTERNS])


def make_model(stage):
    if stage=='onset':return OnsetModel()
    if stage=='arrangement':return ArrangementModel(len(PATTERNS))
    raise ValueError('Explicit stage required')


def balance(config):
    report=json.loads((Path(config['derived'])/'report.json').read_text(encoding='utf-8'))
    return report['train_balance']


def forward_loss(model,b,config,weights):
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['amp']):
        out=model(b['audio']) if config['stage']=='onset' else model(b['audio'],b['frames'],b['previous'],b['context'])
    out={k:v.float() for k,v in out.items()}
    if any(not torch.isfinite(v).all() for v in out.values()):raise FloatingPointError('Nonfinite output')
    def mean(v,m):return (v*m).sum()/m.sum().clamp_min(1)
    if config['stage']=='onset':
        m=b['mask']; positive=m&(b['target']>.5)
        ce=torch.nn.functional.binary_cross_entropy_with_logits(out['logits'],b['target'],reduction='none',
                pos_weight=torch.tensor(weights['onset_positive_weight'],device=out['logits'].device))
        offset=torch.nn.functional.smooth_l1_loss(out['offsets'],b['offsets'],reduction='none')
        loss=dict(classification=mean(ce,m),offset=mean(offset,positive))
        loss['total']=loss['classification']+.25*loss['offset']
    else:
        ce=torch.nn.functional.cross_entropy(out['logits'].transpose(1,2),b['tokens'],reduction='none',
                weight=torch.tensor(weights['pattern_weights'],device=out['logits'].device))
        m=b['valid'];hm=(b['lengths']>0)&m[:,:,None]
        length=torch.nn.functional.smooth_l1_loss(out['log_lengths'],b['lengths'],reduction='none')
        loss=dict(classification=mean(ce,m),length=mean(length,hm))
        loss['total']=loss['classification']+.5*loss['length']
    if any(not torch.isfinite(v) for v in loss.values()):raise FloatingPointError('Nonfinite loss')
    return out,loss


def batch_metrics(out,b,stage):
    if stage=='onset':
        total=np.zeros(3,np.int64); errors=[]
        probabilities=out['logits'].sigmoid().detach().cpu().numpy()
        offsets=out['offsets'].detach().cpu().numpy()
        target=b['target'].cpu().numpy();mask=b['mask'].cpu().numpy()
        reference_offsets=b['offsets'].cpu().numpy()
        for p,o,t,m,ro in zip(probabilities,offsets,target,mask,reference_offsets):
            ix=peaks(p,.5,3);pred=[(f+o[f])*.01 for f in ix if m[f]]
            truth=[(f+ro[f])*.01 for f in np.flatnonzero((t>.5)&m)]
            c,e=counts(pred,truth,.05);total+=c;errors.extend(e)
        return dict(counts=total.tolist(),error_sum_seconds=sum(errors),matched=len(errors))
    selected=out['logits'].argmax(-1);m=b['valid']
    confusion=torch.bincount((b['tokens'][m]*len(PATTERNS)+selected[m]).flatten(),minlength=len(PATTERNS)**2)
    hm=(b['lengths']>0)&m[:,:,None]
    lengths=(out['log_lengths'].clamp_max(8).expm1()-b['lengths'].expm1()).abs()
    return dict(confusion=confusion.reshape(len(PATTERNS),len(PATTERNS)).cpu().tolist(),
                hold_error_seconds=float((lengths*hm).sum()),hold_count=int(hm.sum()))


def summarize(metrics,stage):
    if stage=='onset':
        c=np.sum([m['counts'] for m in metrics],axis=0);matched=sum(m['matched'] for m in metrics)
        return dict(onset=result(c),mean_absolute_ms=1000*sum(m['error_sum_seconds'] for m in metrics)/max(1,matched),
                    threshold=.5,tolerance_ms=50,local_peak_radius_ms=30)
    matrix=np.sum([m['confusion'] for m in metrics],axis=0);n=int(matrix.sum())
    recalls={str(i):float(matrix[i,i]/matrix[i].sum()) for i in range(len(PATTERNS)) if matrix[i].sum()}
    # Lane/class decisions are evaluated per actual lane entity, including holds.
    atoms=np.zeros((len(PATTERNS),2,6),bool)
    for i,(g,a,h) in enumerate(PATTERNS):
        for lane,s in enumerate((g,a)):
            if s:atoms[i,lane,{1:0,2:1,3:2,4:5,5:3}[s]]=True
            if h&(1<<lane):atoms[i,lane,4]=True
    cells=np.zeros((2,6,3),np.int64)
    for truth,pred in zip(*np.nonzero(matrix)):
        r=atoms[truth];p=atoms[pred];v=matrix[truth,pred]
        cells[:,:,0]+=(r&p)*v;cells[:,:,1]+=(~r&p)*v;cells[:,:,2]+=(r&~p)*v
    return dict(pattern_accuracy=float(np.trace(matrix)/max(1,n)),events=n,pattern_recalls=recalls,
                six_classes={k:result(c) for k,c in zip(('light','heavy','extra_heavy','gemini','hold','gear'),cells.sum(0))},
                lane_entities={k:result(c) for k,c in zip(('ground','air'),cells.sum(1))},
                mean_hold_duration_error_seconds=sum(m['hold_error_seconds'] for m in metrics)/max(1,sum(m['hold_count'] for m in metrics)),
                mode='reference onsets, teacher-forced previous events; free-running evaluated separately')


@torch.no_grad()
def evaluate(model,data,config,weights):
    model.eval(); numerator={}; denominator={};metrics=[]
    for sample in DataLoader(data,batch_size=config['batch_size'],collate_fn=collate,num_workers=0,
                             generator=torch.Generator().manual_seed(0)):
        b={k:v.to(config['device']) for k,v in sample.items()}
        out,loss=forward_loss(model,b,config,weights)
        if config['stage']=='onset':
            d=dict(classification=int(b['mask'].sum()),offset=int((b['mask']&(b['target']>.5)).sum()))
        else:d=dict(classification=int(b['valid'].sum()),length=int(((b['lengths']>0)&b['valid'][:,:,None]).sum()))
        for k,v in d.items():numerator[k]=numerator.get(k,0)+float(loss[k])*v;denominator[k]=denominator.get(k,0)+v
        metrics.append(batch_metrics(out,b,config['stage']))
    loss={k:numerator[k]/max(1,denominator[k]) for k in numerator}
    loss['total']=loss['classification']+(.25*loss['offset'] if config['stage']=='onset' else .5*loss['length'])
    return dict(loss=loss,metrics=summarize(metrics,config['stage']))


def train(config,run_dir,resume=None):
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    ident=identity(config);gate=json.loads(Path(config['overfit_report']).read_text(encoding='utf-8'))
    if not gate['passed'] or gate['identity']!=ident:raise ValueError('Matching passed train-only overfit gate required')
    if not resume and (run_dir/'config.json').exists():raise ValueError('Existing experiment requires explicit resume')
    seed_all(config['seed']);data=DualDataset(config,'train');valid=DualDataset(config,'validation')
    model=make_model(config['stage']).to(config['device']);w=balance(config)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=2,min_lr=1e-5)
    state=dict(epoch=0,cursor=0,global_step=0,best=1e30,best_epoch=None,stale=0,anchor=1e30)
    if resume:
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        if saved['identity']!=ident or saved['config']!=config:raise ValueError('Resume identity/config mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        state=saved['state'];restore_rng(saved['rng'])
    else:atomic_json(run_dir/'config.json',dict(config=config,identity=ident,environment=runtime(),
                 parameters=sum(p.numel() for p in model.parameters()),train_windows=len(data),validation_windows=len(valid),locked_test_opened=False))
    cancel=[False]
    def interrupted(*_):cancel[0]=True
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    def save(name='last.pt'):
        checkpoint(run_dir/name,dict(config=config,identity=ident,model=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),state=state.copy(),rng=rng_state()))
    def log(value):
        value.update(time=time.strftime('%Y-%m-%d %H:%M:%S'),elapsed_seconds=time.monotonic()-started)
        with (run_dir/'metrics.jsonl').open('a',encoding='utf-8') as s:s.write(json.dumps(value,allow_nan=False)+'\n')
        atomic_json(run_dir/'progress.json',value);print(json.dumps(value),flush=True)
    status='running';atomic_json(run_dir/'status.json',dict(status=status,**state))
    if not resume:save()
    try:
        while state['epoch']<config['max_epochs'] and state['stale']<config['patience']:
            order=epoch_order(data,config['seed'],state['epoch'])
            loader=DataLoader(Subset(data,order[state['cursor']:]),batch_size=config['batch_size'],collate_fn=collate,
                              num_workers=0,generator=torch.Generator().manual_seed(0))
            model.train()
            for sample in loader:
                if cancel[0] or (run_dir/'CANCEL').exists():status='cancelled';break
                b={k:v.to(config['device']) for k,v in sample.items()};optimizer.zero_grad(set_to_none=True)
                out,loss=forward_loss(model,b,config,w);loss['total'].backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();state['global_step']+=1;state['cursor']+=len(sample['audio'])
                if state['global_step']%config['log_every']==0:
                    log(dict(event='batch',**state,loss={k:float(v.detach()) for k,v in loss.items()},gradient_norm=float(norm),
                             learning_rate=optimizer.param_groups[0]['lr'],cuda_peak_bytes=torch.cuda.max_memory_allocated()));save()
            if status=='cancelled':break
            evaluation=evaluate(model,valid,config,w);metric=evaluation['loss']['total'];scheduler.step(metric)
            improved=metric<state['best']
            if improved:state.update(best=metric,best_epoch=state['epoch']+1)
            if metric<state['anchor']-config['min_delta']:state.update(anchor=metric,stale=0)
            else:state['stale']+=1
            state['epoch']+=1;state['cursor']=0
            log(dict(event='epoch',**state,validation=evaluation,improved=improved,learning_rate=optimizer.param_groups[0]['lr']))
            save()
            if improved:save('best.pt')
        if status=='running':status='completed'
        save()
        final=dict(status=status,**state,seconds=time.monotonic()-started,cuda_peak_bytes=torch.cuda.max_memory_allocated(),
                   stopping_reason='safe_cancel' if status=='cancelled' else 'early_stopping' if state['stale']>=config['patience'] else 'max_epochs',
                   locked_test_opened=False,human_listening='pending',game_acceptance='pending',production_promoted=False)
        atomic_json(run_dir/'status.json',final);log(dict(event='end',**final));return final
    except BaseException as error:
        # Keep the periodic checkpoint at a known completed update boundary.
        # Saving partially applied optimizer state on arbitrary failure is unsafe.
        atomic_json(run_dir/'status.json',dict(status='failed',error=repr(error),**state,restore='last.pt successful boundary'))
        raise


def overfit(config,run_dir,steps):
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=False)
    ident=identity(config);seed_all(config['seed']);data=DualDataset(config,'train');w=balance(config)
    selected=[];covered=set()
    for i,(ri,start) in enumerate(data.index):
        chart=json.loads((data.root/data.rows[ri]['labels']).read_text(encoding='utf-8'))
        kinds={e['kind'] for e in chart['events'] if (start+800)/100<=e['start']<(start+2400)/100}
        if kinds-covered:selected.append(i);covered|=kinds
        if len(covered)==6:break
    if len(covered)!=6:raise ValueError('Diagnostic must cover all six types')
    batch={k:v.to(config['device']) for k,v in collate([data[i] for i in selected]).items()}
    model=make_model(config['stage']).to(config['device']);optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=0)
    initial=None;history=[];passed=False;started=time.monotonic();torch.cuda.reset_peak_memory_stats();norm=None
    for step in range(steps+1):
        if step:
            model.train();optimizer.zero_grad(set_to_none=True);out,loss=forward_loss(model,batch,config,w)
            loss['total'].backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step%50==0 or step==steps:
            model.eval()
            with torch.no_grad():out,loss=forward_loss(model,batch,config,w)
            value=float(loss['total']);initial=value if initial is None else initial
            metrics=summarize([batch_metrics(out,batch,config['stage'])],config['stage'])
            entry=dict(step=step,loss={k:float(v) for k,v in loss.items()},metrics=metrics,gradient_norm=None if norm is None else float(norm))
            history.append(entry);print(json.dumps(entry),flush=True)
            accuracy=metrics['onset']['f1'] if config['stage']=='onset' else metrics['pattern_accuracy']
            passed=step>=100 and value<initial*.2 and accuracy>=.95
            if config['stage']=='arrangement':passed=passed and all(v['recall']>=.8 for v in metrics['six_classes'].values())
            if passed:break
    checkpoint(run_dir/'diagnostic.pt',dict(model=model.state_dict(),identity=ident,config=config))
    report=dict(passed=passed,role='train',identity=ident,history=history,steps=step,initial_loss=initial,final_loss=value,
                loss_reduction=1-value/initial,window_indices=selected,covered_classes=sorted(covered),
                windows=[dict(chart_sha256=data.rows[data.index[i][0]]['chart_sha256'],start_frame=data.index[i][1]) for i in selected],
                parameter_count=sum(p.numel() for p in model.parameters()),output_shapes={k:list(v.shape) for k,v in out.items()},
                seconds=time.monotonic()-started,cuda_peak_bytes=torch.cuda.max_memory_allocated(),finite_outputs_gradients=True,
                validation_opened=False,locked_test_opened=False,formal_training_restarts_from_seed=True,environment=runtime())
    atomic_json(run_dir/'report.json',report)
    if not passed:raise RuntimeError('Train-only overfit gate FAILED; formal training forbidden')
    return report
