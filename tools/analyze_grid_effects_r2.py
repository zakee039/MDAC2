"""Reconstruct deterministic postprocessing stages; isolate timing edits."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from mdac2.data_preparation.audio_features import sha
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.playtest_postprocess import analyze_audio,add_climax_candidates,arrange_phrases,apply_repeats,repeat_candidates
from mdac2.data_preparation.fixed_grid_r2 import snap_events,align_holds
from mdac2.data_preparation.joint_evaluation import EventMetrics,result

def main(experiment):
    output=experiment/'grid-diagnostics.json'
    if output.exists():raise FileExistsError(output)
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    manifest=ROOT/'tools/private/community/training-preparation-v3/manifest.json'
    rows=[r for r in read(manifest)['charts'] if r['training_eligible'] and r['split']=='validation']
    assert len(rows)==22
    baseline=read(experiment/'validation/validation.json')
    assert sha(manifest)==baseline['identity']['manifest_sha256']
    prior=read(ROOT/'models/v2/weights/run-prior.json')
    tensors={x['chart_sha256']:x for x in read(manifest.parent/'tensors/report.json')['charts']}
    metrics={k:EventMetrics() for k in ('before_grid','snap_only','aligned','final')};records=[]
    for row in rows:
        key=row['chart_sha256'];r=experiment/'validation'/key
        neural=read(r/'neural-chart.json');duration=neural['duration']
        events=[TrainingEvent.from_dict(e) for e in neural['events']]
        f=np.load(ROOT/'tools/private/community/pretraining-feature-cache-v1'/row['pcm']['sha256']/'features.npy',mmap_mode='r')
        with np.load(r/'routing.npz') as a:prob=a['density_probability']
        with np.load(ROOT/'tools/private/community/supervised-dual-v1-20260928/evaluation/predictions'/f'{key}.npz') as a:offsets=a['offsets']
        events,_=add_climax_candidates(events,dict(probability=prob,offsets=offsets),analyze_audio(f,duration),duration)
        events,_=arrange_phrases(events,duration,prior)
        snapped,grid=snap_events(events,duration);aligned,alignment=align_holds(snapped,duration)
        final,_=apply_repeats(aligned,repeat_candidates(f,duration),duration)
        assert [e.to_dict() for e in final]==read(r/'chart.json')['events']
        labels=manifest.parent/row['labels'];assert sha(labels)==row['labels_sha256']
        truth=[TrainingEvent.from_dict(e) for e in read(labels)['events'] if e['kind']!='gear']
        tensor=tensors[key];tp=manifest.parent/'tensors'/tensor['file'];assert sha(tp)==tensor['sha256']
        with np.load(tp) as a:mask=a['onset_mask']
        for name,es in [('before_grid',events),('snap_only',snapped),('aligned',aligned),('final',final)]:
            metrics[name].add(es,truth,duration,neural['audit'],mask)
        records.append(dict(chart_sha256=key,grid_repairs=len(grid['repairs']),snapped_positions=len(grid['edits']),
            max_grid_movement_ms=max([abs(x['shift_ms']) for x in grid['edits']],default=0),
            changed_holds=alignment['changed_holds'],extended=alignment['extended'],shortened=alignment['shortened'],
            rejected_alignments=len(alignment['rejected']),
            alignment_shifts_ms=[round((b-a)*1000,6) for c in alignment['changes'] for a,b in zip(c['before'],c['after']) if a!=b]))
    reports={k:v.report() for k,v in metrics.items()}
    for v in reports.values():
        classes=[x for k,x in v['six_classes'].items() if k!='gear'];v['macro_five_class_f1']=sum(x['f1'] for x in classes)/5
    output.write_text(json.dumps(dict(status='completed',metrics=reports,charts=records,final_events_exact_match=True,
        command=[sys.executable,*sys.argv],source_sha256=sha(Path(__file__)),
        purpose='same generated events, isolate 5ms snapping and hold alignment; no new tuning or model inference'),indent=2),encoding='utf-8')
    print(json.dumps({k:dict(onset=v['sampling']['f1'],hold=v['holds']['hold_joint']['f1'],macro=v['macro_five_class_f1']) for k,v in reports.items()}))

if __name__=='__main__':main(Path(sys.argv[1]).resolve())
