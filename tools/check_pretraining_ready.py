"""Audit prepared artifacts and every window without training or test inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.community import digest,write_json
from mdac2.data_preparation.training_chart import CLASSES,POLICY_ID
from mdac2.data_preparation.window_dataset import WindowDataset


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition,message):
    if not condition:raise ValueError(message)


def audit(manifest,cache,tensors,acoustic,acoustic_manifest,export_report,forwards):
    started=time.perf_counter();m=read(manifest);mh=digest(manifest)
    require(m['policy']==POLICY_ID and tuple(m['classes'])==CLASSES,'Label contract')
    require(m['split_frozen'] and not m['training_started'],'Preparation status')
    require(digest(Path(m['selection_path']))==m['selection_sha256'],'Selection changed')
    for name,checksum in m['code'].items():
        path=ROOT/'tools'/name if name=='build_training_dataset.py' else ROOT/'src/mdac2/data_preparation'/name
        require(digest(path)==checksum,'Label implementation changed: '+name)
    charts=m['charts'];eligible=[r for r in charts if r['training_eligible']]
    require(len({r['chart_sha256'] for r in charts})==len(charts),'Duplicate charts')
    for key in ('provisional_family','audio_sha256','pcm'):
        roles={}
        for row in charts:
            value=row[key]['sha256'] if key=='pcm' else row[key]
            roles.setdefault(value,set()).add(row['split'])
        require(all(len(x)==1 for x in roles.values()),'Cross-role leakage: '+key)
    for row in charts:
        require(digest(Path(row['chart_path_absolute']))==row['chart_sha256'],'Source BMS changed')
        path=manifest.parent/row['labels']
        require(digest(path)==row['labels_sha256'],'Label checksum changed')
        label=read(path)
        require(label['training_eligible']==row['training_eligible'],'Eligibility mismatch')
    unique={r['pcm']['sha256']:r for r in eligible}
    for row in unique.values():
        require(digest(Path(row['audio_path_absolute']))==row['audio_sha256'],'Source audio changed')
    feature=read(cache/'report.json')
    require(feature['status']=='completed' and not feature['errors'],'Features incomplete')
    require(feature['manifest_sha256']==mh and feature['scope']=='all_eligible_audio','Feature report scope')
    require(len(feature['complete'])==len(unique)==feature['planned'],'Feature count')
    require({x['pcm_sha256'] for x in feature['complete']}==set(unique),'Feature coverage')
    for pcm,row in unique.items():
        meta=read(cache/pcm/'metadata.json')
        require(meta['source_audio_sha256']==row['audio_sha256'],'Feature source audio identity')
        require(abs(meta['duration']-row['pcm']['seconds'])<=2/44100,'Feature duration')
        require(meta['diagnostics']['reconstruction_max_error']<=1e-6,'Stem reconstruction')
    targets=read(tensors/'report.json')
    require(targets['manifest_sha256']==mh,'Tensor manifest identity')
    require({r['chart_sha256'] for r in targets['charts']}=={r['chart_sha256'] for r in eligible},'Tensor coverage')
    windows={};valid_frames={}
    for role in ('train','validation','locked_test'):
        # Locked-test access here is structural only: no predictions or tuning.
        dataset=WindowDataset(manifest,cache,role,tensor_cache=tensors,allow_locked_test=True)
        total=0
        for i in range(len(dataset)):
            batch=dataset[i];valid=batch['loss_region'];total+=int(valid.sum())
            require(batch['features'].shape==(3,96,3200),'Window shape')
            require(batch['timing'].shape==(3200,4),'Timing shape')
            for name,array in batch.items():
                require(np.isfinite(array).all(),'Nonfinite window '+name)
                if 'mask' in name:
                    require(array.dtype==np.bool_ and not array[~valid].any(),'Mask outside core')
            for name in ('onsets','hold_ends','occupancy'):
                require(np.isin(batch[name],[0,1]).all(),'Nonbinary targets')
            for name,mask in (('offsets','offset_mask'),('hold_end_offsets','hold_end_offset_mask')):
                values=batch[name][batch[mask]]
                require(np.all((values>=-1e-6)&(values<1+1e-6)),'Offset range')
        expected=sum(r['frames'] for r in targets['charts'] if r['split']==role)
        require(abs(total-expected)<=len(dataset.rows),'Window core coverage')
        require(len(dataset)==targets['windows_by_role'][role],'Window count')
        windows[role]=len(dataset);valid_frames[role]=total
        print(json.dumps({'audited_role':role,'windows':len(dataset)}),flush=True)
    a=read(acoustic);old=read(acoustic_manifest)
    require(a['manifest_sha256']==digest(acoustic_manifest),'Fingerprint report source')
    fields=lambda rows:sorted((r['chart_sha256'],r['pcm']['sha256'],r['audio_sha256'],r['split'],r['provisional_family']) for r in rows)
    require(fields(old['charts'])==fields(charts),'Fingerprint roles changed')
    require(set(a['audio_order'])=={r['pcm']['sha256'] for r in charts},'Fingerprint coverage')
    require(digest(acoustic.parent/'fingerprints.npy')==a['fingerprints_sha256'],'Fingerprint checksum')
    require(digest(ROOT/'tools/check_audio_families.py')==a['source_code_sha256'],'Fingerprint implementation changed')
    require(a['cross_role_pairs']==0,'Unresolved acoustic leakage candidates')
    exp=read(export_report)
    require(exp['manifest_sha256']==mh and exp['passed']==len(eligible) and not exp['failures'],'Export audit failed')
    require(exp['exporter_sha256']==digest(ROOT/'src/mdac2/data_preparation/six_export.py'),'Exporter changed')
    require({read(path)['device'] for path in forwards}=={'cpu','cuda'},'CPU and CUDA forward reports required')
    for path in forwards:
        f=read(path)
        require(f['manifest_sha256']==mh and f['sample_role']=='train','Forward source')
        require(f['parameters_before_sha256']==f['parameters_after_sha256'],'Parameters changed')
        require(not any(f[k] for k in ('backward_called','optimizer_created','checkpoint_saved','training_started')),'Training was started')
        for name,key in (('joint_model.py','model_code_sha256'),('joint_losses.py','loss_code_sha256')):
            require(digest(ROOT/'src/mdac2/data_preparation'/name)==f[key],'Forward implementation changed')
    return {'schema_version':1,'engineering_ready':True,'training_started':False,
        'manifest_sha256':mh,'charts':len(charts),'eligible_charts':len(eligible),
        'quarantined_charts':len(charts)-len(eligible),'eligible_by_role':dict(Counter(r['split'] for r in eligible)),
        'feature_audio':len(unique),'windows_by_role':windows,'supervised_core_frames':valid_frames,
        'export_roundtrips':exp['passed'],'fingerprint_cross_role_candidates':a['cross_role_pairs'],
        'audit_seconds':time.perf_counter()-started,
        'artifacts':{str(p.resolve()):digest(p) for p in [manifest,cache/'report.json',tensors/'report.json',acoustic,export_report,*forwards]},
        'code':{p.name:digest(p) for p in (ROOT/'src/mdac2/data_preparation').glob('*.py')},
        'audit_code_sha256':digest(Path(__file__)),
        'pending':['supervised trainer/decoding/evaluation implementation by Sol',
                   'human listening and alignment review; community references are not gold',
                   'real game/Boss scene loading and production workstation integration'],
        'locked_test_access':'structure/finite/mask checks only; no model predictions or quality metrics'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','cache','tensors','acoustic','acoustic-manifest','export-report','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--forward',type=Path,action='append',required=True)
    a=p.parse_args()
    result=audit(a.manifest,a.cache,a.tensors,a.acoustic,a.acoustic_manifest,a.export_report,a.forward)
    write_json(a.output,result);print(json.dumps(result,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
