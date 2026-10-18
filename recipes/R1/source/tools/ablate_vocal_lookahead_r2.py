"""Fixed sequential onset-policy ablations, no fitting or song-specific tuning."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from mdac2.data_preparation.audio_features import sha
from mdac2.data_preparation.vocal_priority import select_onsets as r1_select
from mdac2.data_preparation.vocal_priority_r2 import LookaheadPolicy,select_onsets
from mdac2.data_preparation.dual_inference import onset_times
from mdac2.data_preparation.joint_evaluation import counts,result


def main(output):
    if output.exists():raise FileExistsError(output)
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    m=ROOT/'tools/private/community/training-preparation-v3/manifest.json'
    if sha(m)!='34add12c1c47fc99afc04ae1fa9f06e968d5c2c50db41cb32a047f033668eb71':raise ValueError('Manifest changed')
    rows=[r for r in read(m)['charts'] if r['training_eligible'] and r['split']=='validation']
    assert len(rows)==22
    standard=LookaheadPolicy()
    variants={'route_only':replace(standard,vocal_model_threshold=.12,novelty_threshold=.22,vocal_spacing_frames=8),
              'route_lower_model':replace(standard,novelty_threshold=.22,vocal_spacing_frames=8),
              'route_lower_evidence':replace(standard,vocal_spacing_frames=8),'r2_full':standard}
    totals={name:np.zeros(3,np.int64) for name in ['v2_neural','r1',*variants]}
    errors={name:[] for name in totals};records=[]
    for row in rows:
        key=row['chart_sha256'];labels=m.parent/row['labels']
        if sha(labels)!=row['labels_sha256']:raise ValueError('Labels changed')
        truth=sorted({e['start'] for e in read(labels)['events'] if e['kind']!='gear'})
        cache=ROOT/'tools/private/community/pretraining-feature-cache-v1'/row['pcm']['sha256']
        f=np.load(cache/'features.npy',mmap_mode='r')
        d=read(ROOT/'tools/private/community/supervised-dual-v6-20260930/evaluation/charts-predicted_onsets'/f'{key}.json')['duration']
        with np.load(ROOT/'tools/private/community/supervised-dual-v1-20260928/evaluation/predictions'/f'{key}.npz') as a:
            prediction={k:a[k] for k in ('probability','offsets')}
        results={'v2_neural':onset_times(prediction,.3,d),'r1':r1_select(f,prediction,d)[0]}
        for name,policy in variants.items():results[name]=select_onsets(f,prediction,d,policy)[0]
        item={}
        for name,times in results.items():
            c,e=counts(times,truth,.05);totals[name]+=c;errors[name].extend(e);item[name]=result(c)
        records.append(dict(chart_sha256=key,results=item))
    metrics={name:dict(result(c),mean_absolute_ms=float(np.mean(errors[name]))*1000) for name,c in totals.items()}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(status='completed',role='validation',charts=records,metrics=metrics,
        policies={k:asdict(v) for k,v in variants.items()},
        metric_scope='raw lane-free onsets before arrangement and postprocessing; unmasked non-gear reference starts; not the masked full-chain metric',
        selected='r2_full as predefined exploratory policy; no threshold search',
        command=[sys.executable,*sys.argv],source_sha256=sha(Path(__file__)),locked_test_opened=False),indent=2),encoding='utf-8')
    print(json.dumps(metrics))


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
