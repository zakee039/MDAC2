"""Read-only regression of V5 against existing V4 neural/cache pairs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mdac2.application.studio_strategy_catalog import select_onsets
from mdac2.application.strategies.v5 import run
from mdac2.application.strategies.v5_export import write_bms,read_bms,semantics
from mdac2.application.studio_timing import settings
from mdac2.data_preparation.training_chart import TrainingEvent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--studio',type=Path,default=ROOT/'chart/studio')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Use a new report path')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    prior=read(ROOT/'models/v3/weights/run-prior.json');cases={};results=[]
    for p in sorted(args.studio.glob('*/*/neural-chart.json')):
        request=p.with_name('request.json')
        if not request.exists():continue
        r=read(request);cache=Path(r['cache'])
        if r.get('strategy_id')!='v4' or not (cache/'features/features.npy').exists():continue
        key=(r['source_sha256'],json.dumps(r.get('trim'),sort_keys=True))
        cases[key]=(p,r,cache)
    for p,r,cache in cases.values():
        chart=read(p);f=np.load(cache/'features/features.npy',allow_pickle=False)
        prediction_path=cache/'onsets-bf4b4bd0b33fea90.npz'
        with np.load(prediction_path,allow_pickle=False) as z:pred={k:z[k] for k in z.files}
        _,pred,_=select_onsets('v5',f,pred,chart['duration'])
        for enabled in (False,True):
            config=dict(settings(r.get('timing')),enabled=enabled,subdivisions=12)
            es,report,grid,audit=run([TrainingEvent.from_dict(e) for e in chart['events']],f,pred,
                chart['duration'],prior,config)
            assert audit['engineering_playtest_gate_passed'],(str(p),enabled,audit)
            assert semantics(es)==semantics(read_bms(write_bms(es,chart['duration']),chart['duration']))
            results.append(dict(title=r['metadata']['title'],source_chart=str(p),bpm=enabled,
                events=len(es),counts=dict(Counter(e.kind for e in es)),hold_additions=len(report['holds']['added']),
                hold_extensions=len(report['holds']['extended']),dense_groups=len(report['dense_groups']),
                dense_source_notes=sum(e.source_note_count for e in es if e.kind=='masher'),
                soft_edge_notes=len(report['soft_edges']),passed=True))
        print('passed',r['metadata']['title'],flush=True)
    if not results:raise ValueError('No V4 cache cases found')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(cases=results,passed=len(results),
        scope='Postprocessing of stored V4 neural output; not fresh V5 model inference or game validation'),
        ensure_ascii=False,indent=2),encoding='utf-8')
    print('total',len(results),flush=True)


if __name__=='__main__':main()
