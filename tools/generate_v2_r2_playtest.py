"""Generate a new-song V2 vs R2-pending comparison with shared verified audio."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from generate_joint_playtest import audio,features as prepare_features,read,save,package
from mdac2.data_preparation.audio_features import sha,validate_cache


def main(a):
    out=a.output.resolve();source=a.source.resolve()
    out.mkdir(parents=True,exist_ok=True)
    request=dict(source=str(source),source_sha256=sha(source),title=a.title,creator='mikee',scene='scene_01',
        versions=['V2 frozen','R2 pending = R1 + hold-feedback-rc2'],training=False,parameter_tuning=False)
    if (out/'request.json').exists():
        if read(out/'request.json')!=request:raise ValueError('Another request owns output')
    else:save(out/'request.json',request)
    if (out/f'{a.stage}-status.json').exists() and read(out/f'{a.stage}-status.json')['status']=='completed':
        raise FileExistsError('Stage completed; retain results')
    started=time.monotonic()
    def status(step):save(out/f'{a.stage}-status.json',dict(status='running',step=step,seconds=time.monotonic()-started))
    try:
        if a.stage=='prepare':
            status('audio')
            record=audio(SimpleNamespace(source=source,output=out))
            if not 0<record['duration']<=600:raise ValueError('Song outside supported duration')
            print('Audio seconds: '+str(record['duration']),flush=True)
            status('separation_and_features')
            prepare_features(SimpleNamespace(output=out))
        else:
            import numpy as np
            import torch
            from mdac2.data_preparation.dual_models import OnsetModel
            from mdac2.data_preparation.arrangement_context_v6 import ContextArrangement,predict,decode
            from mdac2.data_preparation.training_tensors import window_starts,slice_window
            from mdac2.data_preparation.joint_decoding import place_core
            from mdac2.data_preparation.dual_inference import onset_times
            from mdac2.data_preparation.supervised_training import seed_all
            from mdac2.data_preparation.playtest_postprocess import process as v2_process,Policy
            from mdac2.data_preparation.vocal_priority_r2 import select_onsets,LookaheadPolicy
            from mdac2.data_preparation.hold_refinement_rc1 import process as r2_process,HoldPolicy
            from mdac2.data_preparation.fixed_grid_r2 import GRID
            from mdac2.data_preparation import six_export,six_grid_export_r2
            from mdac2.data_preparation.playability_audit import audit_playability
            if read(out/'prepare-status.json')['status']!='completed':raise ValueError('Preparation incomplete')
            for name in ('v2','r2-pending'):
                if (out/name).exists():raise FileExistsError('Preserve partial/completed prediction: '+name)
            status('identity_checks')
            record=read(out/'audio.json');assert sha(source)==record['source_sha256']
            assert sha(out/'mix.wav')==record['wav_sha256']
            validate_cache(out/'features',dict(pcm=dict(sha256=record['pcm_sha256'])))
            manifest=read(ROOT/'models/v2/manifest.json');recipe=read(ROOT/'recipes/R1/manifest.json')
            candidate=read(ROOT/'tools/private/community/hold-feedback-rc2-20261017/identity.json')
            assert sha(ROOT/'recipes/R1/manifest.json')==candidate['recipe_sha256']
            for name,item in {**manifest['postprocessing']['sources'],**recipe['sources']}.items():
                assert sha(ROOT/name)==item['sha256'],name
            hold_module='src/mdac2/data_preparation/hold_refinement_rc1.py'
            assert sha(ROOT/hold_module)==candidate['sources'][hold_module]
            prior_path=ROOT/manifest['postprocessing']['prior_path']
            assert sha(prior_path)==manifest['postprocessing']['prior_sha256']
            prior=read(prior_path)
            seed_all(20260928);torch.set_num_threads(4)
            models={};provenance={}
            for stage,klass in [('onset',OnsetModel),('arrangement',ContextArrangement)]:
                item=manifest['checkpoints'][stage];path=ROOT/item['path'];assert sha(path)==item['sha256']
                saved=torch.load(path,map_location='cpu',weights_only=False)
                for name,digest in saved['identity']['code'].items():
                    assert sha(ROOT/'src/mdac2/data_preparation'/name)==digest,name
                model=klass().cuda().eval();model.load_state_dict(saved['model']);models[stage]=model
                provenance[stage]=dict(item,path=str(path))
            identity=dict(request=request,source_tool_sha256=sha(Path(__file__)),model_manifest_sha256=sha(ROOT/'models/v2/manifest.json'),
                recipe_manifest_sha256=sha(ROOT/'recipes/R1/manifest.json'),r2_candidate_identity_sha256=sha(ROOT/'tools/private/community/hold-feedback-rc2-20261017/identity.json'),
                registry_sha256=sha(ROOT/'models/registry.json'),checkpoints=provenance,prior_sha256=sha(prior_path),
                v2_policy=asdict(Policy()),r2_vocal_policy=asdict(LookaheadPolicy()),r2_hold_policy=asdict(HoldPolicy()),
                feature_metadata_sha256=sha(out/'features/metadata.json'),command=[sys.executable,*sys.argv])
            save(out/'identity.json',identity)
            shutil.copyfile(Path(__file__),out/'generation-source.py')
            f=np.load(out/'features/features.npy',mmap_mode='r',allow_pickle=False);n=f.shape[-1];duration=record['duration']
            predictions=dict(probability=np.zeros(n,np.float32),offsets=np.zeros(n,np.float32));coverage=np.zeros(n,np.int32)
            status('shared_onset_model')
            with torch.inference_mode():
                for start in window_starts(n):
                    window,_=slice_window(f[1:],start,time_axis=2)
                    tensor=torch.as_tensor(window[None],device='cuda',dtype=torch.float32)
                    with torch.autocast('cuda',dtype=torch.bfloat16):o=models['onset'](tensor)
                    place_core(predictions,coverage,start,dict(probability=o['logits'][0].float().sigmoid().cpu().numpy(),offsets=o['offsets'][0].float().cpu().numpy()))
            assert np.all(coverage==1)
            np.savez_compressed(out/'prediction.npz',**predictions)
            for name in ('v2','r2-pending'):
                dest=out/name;dest.mkdir()
                status(name+'_arrangement')
                if name=='v2':times=onset_times(predictions,.3,duration)
                else:
                    times,routing,routing_report=select_onsets(f,predictions,duration)
                    np.savez_compressed(dest/'routing.npz',**routing);save(dest/'routing.json',routing_report)
                scores=predict(models['arrangement'],f,times,duration)
                np.savez_compressed(dest/'arrangement-scores.npz',times=np.asarray(times),**scores)
                es,decoding=decode(scores['logits'],scores['log_lengths'],times,duration)
                save(dest/'neural-chart.json',dict(duration=duration,events=[e.to_dict() for e in es],audit=decoding))
                status(name+'_postprocessing')
                if name=='v2':after,report=v2_process(es,f,predictions,duration,prior);exporter=six_export
                else:
                    after,report=r2_process(es,f,dict(probability=routing['density_probability'],offsets=predictions['offsets']),duration,prior)
                    exporter=six_grid_export_r2
                audit=audit_playability(after,duration);assert audit['engineering_playtest_gate_passed']
                bms=exporter.write_bms(after,duration)
                assert exporter.semantics(after)==exporter.semantics(exporter.read_bms(bms,duration))
                (dest/'experimental.bms').write_text(bms,encoding='utf-8')
                save(dest/'report.json',report);save(dest/'playability-audit.json',audit)
                chart=dict(title=a.title,creator='mikee',scene='scene_01',difficulty_input=9.5,duration=duration,
                    source_audio=record,events=[e.to_dict() for e in after],class_counts=dict(Counter(e.kind for e in after)),
                    checkpoint_identity=provenance,model_id='v2',recipe='V2 default' if name=='v2' else 'R2 pending (hold-feedback-rc2)',
                    postprocess_policy=report['policy'],postprocess_report_sha256=sha(dest/'report.json'),
                    model_manifest_sha256=identity['model_manifest_sha256'],base_decoding=decoding,
                    bpm_used=False,gear_postprocessing=False,human_listening='pending',game_acceptance='pending',production_promoted=False)
                if name!='v2':chart.update(export_grid=GRID,bpm_used_for_storage=240)
                save(dest/'chart.json',chart)
                if name=='v2':save(out/'chart.json',chart) # verified media parent uses V2 events
                print(name+': '+str(chart['class_counts']),flush=True)
            status('verified_shared_media_package')
            package(SimpleNamespace(output=out,title=a.title+' - V2 原方案',creator='mikee',difficulty=9.5))
            assert sha(ROOT/'models/registry.json')==identity['registry_sha256']
        save(out/f'{a.stage}-status.json',dict(status='completed',seconds=time.monotonic()-started,
            command=[sys.executable,*sys.argv],source_tool_sha256=sha(Path(__file__))))
    except BaseException as exc:
        save(out/f'{a.stage}-status.json',dict(status='failed',error=repr(exc),seconds=time.monotonic()-started));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','infer'],required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--title',required=True)
    main(p.parse_args())
