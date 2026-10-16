"""Audit immutable inputs and create new train/validation-only derivatives."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.dual_data import derive,PATTERNS,POLICY
from mdac2.data_preparation import dual_data
from mdac2.data_preparation.window_dataset import WindowDataset
from mdac2.data_preparation.audio_features import sha
from mdac2.data_preparation.supervised_training import atomic_json,data_identity
from mdac2.data_preparation.dual_training import make_model

def main(output):
    base=json.loads((ROOT/'tools/private/community/supervised-joint-v1-20260927/baseline-lr3e4.json').read_text(encoding='utf-8'))
    original=data_identity(base)
    output.mkdir(parents=True,exist_ok=False);derived=output/'derived';derived.mkdir()
    records=[];freq=np.zeros(len(PATTERNS),np.int64);positive=negative=0;roles={};audit=[]
    for role in ('train','validation'):
        dataset=WindowDataset(base['manifest'],base['cache'],role,tensor_cache=base['tensors']);roles[role]=len(dataset.rows)
        first={}
        for i,(ri,_) in enumerate(dataset.index):first.setdefault(ri,i)
        for ri,row in enumerate(dataset.rows):
            dataset[first[ri]]
            _,_,targets=dataset.loaded[ri];chart=json.loads((dataset.root/row['labels']).read_text(encoding='utf-8'))
            target=derive(chart,targets);file=row['chart_sha256']+'.npz';np.savez_compressed(derived/file,**target)
            records.append(dict(chart_sha256=row['chart_sha256'],file=file,sha256=sha(derived/file),role=role,
                                labels_sha256=row['labels_sha256'],ambiguous_events=int((~target['valid']).sum()),events=len(target['frames'])))
            if role=='train':
                freq+=np.bincount(target['tokens'][target['valid']],minlength=len(PATTERNS))
                positive+=int(((target['target']>.5)&target['mask']).sum());negative+=int(((target['target']<.5)&target['mask']).sum())
            audit.append(dict(chart_sha256=row['chart_sha256'],pcm_sha256=row['pcm']['sha256'],cache_verified=True))
            if (ri+1)%25==0:print(role,ri+1,flush=True)
    # Square-root weighting avoids the old very large independent-sigmoid weights.
    weights=np.sqrt(freq.max()/np.maximum(1,freq)).clip(1,4);weights/=np.average(weights,weights=freq)
    report=dict(policy=POLICY,manifest_sha256=sha(Path(base['manifest'])),implementation_sha256=sha(Path(dual_data.__file__)),
                charts=records,roles=roles,locked_test_opened=False,original_identity=original,
                train_balance=dict(onset_positive_weight=min(8,float(np.sqrt(negative/max(1,positive)))),positive_frames=positive,
                                   negative_frames=negative,pattern_counts=freq.tolist(),pattern_weights=weights.tolist()),
                patterns=[list(p) for p in PATTERNS],audit=audit,source_preparation_modified=False)
    atomic_json(derived/'report.json',report)
    for stage in ('onset','arrangement'):
        c={**base,'stage':stage,'derived':derived.as_posix(),'seed':20260928,'batch_size':4,
           'learning_rate':.0003 if stage=='onset' else .0002,'max_epochs':40,'patience':8,'log_every':25,
           'overfit_report':(output/f'overfit-{stage}'/'report.json').as_posix()}
        atomic_json(output/f'{stage}.json',c)
    atomic_json(output/'preparation.json',dict(status='completed',parameters={s:sum(p.numel() for p in make_model(s).parameters()) for s in ('onset','arrangement')},
        roles=roles,manifest_sha256=report['manifest_sha256'],derived_report_sha256=sha(derived/'report.json'),
        ambiguous={r:sum(x['ambiguous_events'] for x in records if x['role']==r) for r in roles},locked_test_opened=False))
    print((output/'preparation.json').read_text(encoding='utf-8'))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output.resolve())
