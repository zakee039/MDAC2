"""Precompute sparse frame supervision and train-only balance statistics."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.community import digest,write_json
from mdac2.data_preparation import training_tensors
from mdac2.data_preparation.training_tensors import frame_targets,timing_features,window_starts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    m=json.loads(a.manifest.read_text(encoding='utf-8'));root=a.manifest.parent
    report={'schema_version':1,'manifest_sha256':digest(a.manifest),
            'implementation_sha256':digest(Path(training_tensors.__file__)),
            'charts':[],'training_started':False}
    positives=np.zeros(6,dtype=np.int64);negatives=positives.copy()
    end_positive=np.zeros(2,dtype=np.int64);end_negative=end_positive.copy()
    windows=Counter();collisions=0
    for r in m['charts']:
        if not r['training_eligible']:continue
        path=root/r['labels']
        if digest(path)!=r['labels_sha256']:raise ValueError('Labels changed')
        label=json.loads(path.read_text(encoding='utf-8'));target=frame_targets(label)
        errors=target.pop('collisions');collisions+=len(errors)
        target['timing']=timing_features(label,r['rating'],len(target['onsets']))
        dest=a.output/(r['chart_sha256']+'.npz')
        np.savez_compressed(dest,**target)
        count=len(window_starts(len(target['onsets'])));windows[r['split']]+=count
        report['charts'].append({'chart_sha256':r['chart_sha256'],'labels_sha256':r['labels_sha256'],
            'file':dest.name,'sha256':digest(dest),'frames':len(target['onsets']),'windows':count,
            'split':r['split'],'quantization_collisions':errors})
        if r['split']=='train':
            positives+=(target['onsets'].astype(bool)&target['onset_mask']).sum(axis=(0,1))
            negatives+=((~target['onsets'].astype(bool))&target['onset_mask']).sum(axis=(0,1))
            end_positive+=(target['hold_ends'].astype(bool)&target['hold_end_mask']).sum(axis=0)
            end_negative+=((~target['hold_ends'].astype(bool))&target['hold_end_mask']).sum(axis=0)
    report.update({'windows_by_role':dict(windows),'quantization_collision_cells':collisions,
        'train_only_balance':{'classes':list(training_tensors.CLASSES),'positives':positives.tolist(),
            'negatives':negatives.tolist(),'suggested_positive_weights':np.clip(np.sqrt(negatives/np.maximum(positives,1)),1,100).tolist(),
            'hold_end_positive_weights':np.clip(np.sqrt(end_negative/np.maximum(end_positive,1)),1,100).tolist(),
            'policy':'sqrt(negative/positive), clip[1,100]; training-role masks only; untested initial suggestion'}})
    write_json(a.output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='charts'},ensure_ascii=True),flush=True)


if __name__=='__main__':main()
