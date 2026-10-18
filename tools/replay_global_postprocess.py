"""Replay saved neural outputs through current global rules; no inference/training.

Reads only the existing 22 validation charts and two explicitly requested songs.
Refuses to overwrite outputs; preserves the original neural and package inputs.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from mdac2.data_preparation.audio_features import sha,validate_cache
from mdac2.data_preparation.global_postprocess import process,STORAGE,POLICY_ID,issues,avoidance_violations
from mdac2.data_preparation.training_chart import TrainingEvent
from mdac2.data_preparation.six_export import write_bms,read_bms,semantics
from mdac2.data_preparation.joint_evaluation import EventMetrics
from mdac2.data_preparation.playability_audit import audit_playability


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def events(c):return [TrainingEvent.from_dict(e) for e in c['events']]


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    save(a.output/'status.json',dict(status='running'))
    try:
        run(a)
    except Exception:
        save(a.output/'status.json',dict(status='failed',traceback=traceback.format_exc()))
        raise


def run(a):
    prep=ROOT/'tools/private/community/training-preparation-v3'
    parent_identity=read(a.parent/'validation/identity.json')
    assert sha(prep/'manifest.json')==parent_identity['data_manifest_sha256']
    assert sha(ROOT/'models/v2/manifest.json')==parent_identity['model_manifest_sha256']
    manifest=read(ROOT/'models/v2/manifest.json')
    for c in manifest['checkpoints'].values():assert sha(ROOT/c['path'])==c['sha256']
    prior_path=ROOT/manifest['postprocessing']['prior_path']
    assert sha(prior_path)==parent_identity['prior_sha256']
    prior=read(prior_path)
    defaults=read(ROOT/'recipes/generation-defaults.json')
    assert defaults['postprocess_profile']==POLICY_ID
    files=[Path(__file__),ROOT/'tools/generate_current_playtest.py',ROOT/'recipes/generation-defaults.json',
           ROOT/'tests/unit/test_global_postprocess.py']
    files += [ROOT/'src/mdac2/data_preparation'/n for n in
              ['global_postprocess.py','hold_refinement_rc1.py','fixed_grid_r2.py','playtest_postprocess.py',
               'generated_constraints.py','training_chart.py','six_export.py','joint_evaluation.py']]
    (a.output/'source').mkdir()
    for p in files:shutil.copyfile(p,a.output/'source'/p.name)
    identity=dict(command=[sys.executable,*sys.argv],parent=str(a.parent.resolve()),defaults=defaults,
        parent_identity_sha256=sha(a.parent/'validation/identity.json'),
        data_manifest_sha256=sha(prep/'manifest.json'),checkpoints=manifest['checkpoints'],
        model_manifest_sha256=sha(ROOT/'models/v2/manifest.json'),prior_sha256=sha(prior_path),
        sources={str(p.relative_to(ROOT)):sha(p) for p in files},
        neural_outputs_reused=True,weights_retrained=False,separation_repeated=False,locked_test_opened=False)
    save(a.output/'identity.json',identity)
    rows=[r for r in read(prep/'manifest.json')['charts'] if r['training_eligible'] and r['split']=='validation']
    assert len(rows)==22
    tensors={r['chart_sha256']:r for r in read(prep/'tensors/report.json')['charts']}
    jobs=[('validation/'+r['chart_sha256'],ROOT/'tools/private/community/pretraining-feature-cache-v1'/r['pcm']['sha256'],r) for r in rows]
    for name,cache in [('pipa','20260927-pipaxing-v1'),('snow','20261016-snow-song-show-v1')]:
        cache=ROOT/'tools/private/playtests'/cache
        audio=read(cache/'audio.json')
        assert sha(Path(audio['source']))==audio['source_sha256']
        assert sha(cache/'mix.wav')==audio['wav_sha256']
        jobs.append((name,cache/'features',dict(pcm=dict(sha256=audio['pcm_sha256']))))
    metrics={k:EventMetrics() for k in ('previous','strict')};records=[]
    for i,(name,cache,row) in enumerate(jobs):
        if (a.output/'CANCEL').exists():
            save(a.output/'status.json',dict(status='cancelled',completed=i));return
        source=a.parent/name;out=a.output/name;out.mkdir(parents=True)
        old=read(source/'chart.json');neural=read(source/'neural-chart.json');duration=old['duration']
        validate_cache(cache,row)
        features=np.load(cache/'features.npy',mmap_mode='r',allow_pickle=False)
        with np.load(source/'onset-selection.npz',allow_pickle=False) as z:
            prediction={k:z[k] for k in ('probability','offsets')}
        after,report=process(events(neural),features,prediction,duration,prior)
        assert not issues(after,duration) and not avoidance_violations(after)
        assert not report['final_dense_pairs']
        bms=write_bms(after,duration)
        assert semantics(after)==semantics(read_bms(bms,duration))
        audit=audit_playability(after,duration)
        assert audit['engineering_playtest_gate_passed']
        save(out/'report.json',report);save(out/'playability-audit.json',audit)
        chart=dict(old,events=[e.to_dict() for e in after],recipe=POLICY_ID,storage=STORAGE,
            class_counts=dict(Counter(e.kind for e in after)),postprocess_policy=report['policy'],
            postprocess_report_sha256=sha(out/'report.json'),human_listening='pending',game_acceptance='pending')
        save(out/'chart.json',chart);(out/'experimental.bms').write_text(bms,encoding='utf-8')
        inputs={f:sha(source/f) for f in ('chart.json','neural-chart.json','onset-selection.npz')}
        inputs['features.npy']=sha(cache/'features.npy');save(out/'inputs.json',inputs)
        old_ids={e.event_id for e in events(old)};new_ids={e.event_id for e in after}
        rec=dict(name=name,previous_events=len(old_ids),events=len(after),
            removed_ids=sorted(old_ids-new_ids),added_ids=sorted(new_ids-old_ids),
            moved=len(report['avoidance']['edits']),dropped=report['avoidance']['dropped'],
            exclusive_initial=report['exclusive_initial'],exclusive_final=report['avoidance']['exclusive'],
            violations=report['final_avoidance_violations'],dense_pairs=report['final_dense_pairs'],
            gemini_to_single=sum(e['gemini_to_opposite_light'] for e in report['avoidance']['edits']),
            double_to_single=sum(e['double_heavy_to_opposite_heavy'] for e in report['avoidance']['edits']))
        records.append(rec)
        if name.startswith('validation/'):
            labels=prep/row['labels'];assert sha(labels)==row['labels_sha256']
            truth=[e for e in events(read(labels)) if e.kind!='gear']
            t=tensors[row['chart_sha256']];tp=prep/'tensors'/t['file'];assert sha(tp)==t['sha256']
            with np.load(tp,allow_pickle=False) as z:mask=z['onset_mask']
            for key,es in [('previous',events(old)),('strict',after)]:
                metrics[key].add(es,truth,duration,neural['audit'],mask)
        save(a.output/'status.json',dict(status='running',completed=i+1,total=24,last=name))
        print(name,len(old_ids),'->',len(after),'drop_groups',len(rec['dropped']),flush=True)
    save(a.output/'comparison.json',dict(charts=records,metrics={k:v.report() for k,v in metrics.items()},
        interpretation='Same neural outputs; postprocessing only. Reused validation, not independent final test.'))
    assert all(sha(ROOT/p)==digest for p,digest in identity['sources'].items())
    save(a.output/'status.json',dict(status='completed',charts=24,all_engineering_gates_passed=True,
        human_listening='pending',game_acceptance='pending',production_promoted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
