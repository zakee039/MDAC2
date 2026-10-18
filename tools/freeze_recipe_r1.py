"""Freeze the explicitly user-selected R1 recipe without modifying V2."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT/'tools/private/community/vocal-grid-r2-20261017'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def create_json(path, value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)
        stream.write('\n')


def main():
    target = ROOT/'recipes/R1'
    registry_path = ROOT/'models/registry.json'
    registry = read(registry_path)
    before_registry = sha(registry_path)
    if target.exists() or 'R1' in registry.get('recipes',{}):
        raise FileExistsError('R1 is already registered or partially created; never overwrite')
    if registry['current_model'] != 'v2':
        raise ValueError('Expected the selected V2 model base')
    completion = read(EXPERIMENT/'completion.json')
    baseline_path = ROOT/'models/v2/manifest.json'
    baseline = read(baseline_path)
    if completion['engineering']!='passed' or completion['status']!='completed':
        raise ValueError('Completed engineering audit required')
    if sha(baseline_path)!=completion['baseline_manifest_sha256']:
        raise ValueError('Frozen V2 identity changed')
    for stage,identity in completion['checkpoint_identity'].items():
        if sha(ROOT/baseline['checkpoints'][stage]['path'])!=identity['sha256']:
            raise ValueError('Checkpoint changed: '+stage)
        for name,digest in identity['bound_sources'].items():
            if sha(ROOT/'src/mdac2/data_preparation'/name)!=digest:
                raise ValueError('Training-bound source changed: '+name)
    if sha(ROOT/baseline['postprocessing']['prior_path'])!=baseline['postprocessing']['prior_sha256']:
        raise ValueError('Training-only arrangement prior changed')
    sources = dict(completion['source_sha256'])
    for path,meta in baseline['postprocessing']['sources'].items():sources[path]=meta['sha256']
    for path,digest in sources.items():
        if sha(ROOT/path)!=digest:raise ValueError('Approved source changed: '+path)
    evidence_names = ['completion.json','validation/validation.json','onset-ablation.json','grid-diagnostics.json','http-check.json','full-tests.xml']
    evidence = {name:dict(path=(EXPERIMENT/name).relative_to(ROOT).as_posix(),sha256=sha(EXPERIMENT/name)) for name in evidence_names}
    if evidence['validation/validation.json']['sha256']!=completion['validation_sha256']:
        raise ValueError('Validation changed')
    deliveries=[]
    for item in completion['packages']:
        title='琵琶行' if item['song']=='pipa' else 'Snow Song Show'
        original=Path(item['delivery']);delivery=original.parent/f'{title}_mikee_MDAC2试玩_R1.mdm'
        if delivery.exists():raise FileExistsError(delivery)
        if sha(original)!=item['sha256']:raise ValueError('Approved playtest changed')
        deliveries.append((item,title,original,delivery))
    # All inputs are verified before publication. New outputs never replace old ones.
    (target/'source').mkdir(parents=True)
    frozen_sources={}
    for path,digest in sources.items():
        destination=target/'source'/path
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/path,destination)
        if sha(destination)!=digest:raise ValueError('Source copy mismatch')
        frozen_sources[path]=dict(snapshot=destination.relative_to(ROOT).as_posix(),sha256=digest)
    packages=[]
    for item,title,original,delivery in deliveries:
        with delivery.open('xb') as stream,original.open('rb') as source:shutil.copyfileobj(source,stream)
        if sha(delivery)!=item['sha256']:raise ValueError('Package copy mismatch')
        packages.append(dict(song=title,delivery=str(delivery),sha256=item['sha256'],
            origin=str(original),byte_identical_to_user_selected_r2=True,internal_title_preserved=True,
            preview=item['preview']))
    inherited_policy=dict(baseline['postprocessing']['policy'])
    inherited_policy.pop('tail_extension')
    inherited_policy.pop('version')
    manifest=dict(schema_version=1,recipe_id='R1',series='R',product='MDAC2',
        application_version='0.1.0',release_date='2026-10-17',
        created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        status='user_selected_playtest_recipe',
        base_model=dict(model_id='v2',display_name='V2',manifest='models/v2/manifest.json',
                        manifest_sha256=sha(baseline_path)),
        checkpoints=baseline['checkpoints'],new_neural_weights=False,
        origin_experiment=EXPERIMENT.relative_to(ROOT).as_posix(),
        origin_policy='vocal-lookahead-r2 + playtest-grid-r2',
        compatibility=dict(supported_base_models=['v2'],other_models_verified=False,
            composition='Build from neural candidates and regenerate arrangement. Do not append to already postprocessed V2 chart.',
            shared_with_v2=['single_lane_phrases','training_only_run_prior','evidence_based_climax_candidates','similar_passage_mirroring','legality_checks'],
            replaces_v2=['hold_tail_extension_250ms'],
            extends_v2=['vocal_lookahead_and_lower_thresholds','vocal_gate_on_climax_candidates','5ms_fixed_time_grid','hold_head_tail_alignment_100ms'],
            old_two_second_vocal_latch=False),
        pipeline=['one_second_vocal_routing','frozen_arrangement_full_song_generated_history',
            'vocal_gated_climax_candidates','single_lane_phrases','5ms_snap','hold_endpoint_alignment',
            'similar_passage_mirroring','read_only_validation','fixed240_BMS_export'],
        vocal_policy=completion['policy'],export_grid=completion['export_grid'],
        shared_postprocessing_parameters=inherited_policy,
        prior=dict(path=baseline['postprocessing']['prior_path'],sha256=baseline['postprocessing']['prior_sha256']),
        hold_alignment=dict(candidate_tolerance_seconds=.1,note_anchor='same_or_opposite_lane',
            hold_pair_target='mean_then_5ms_grid',relations=['head_head','tail_tail','tail_head'],
            minimum_duration_seconds=.03,old_250ms_tail_extension=False),
        sources=frozen_sources,evidence=evidence,
        validation=dict(dataset_manifest_sha256=completion['data_manifest_sha256'],train=172,validation=22,
            previously_used_test=25,quarantine=127,independent_test=False,
            metrics=completion['validation_metrics']['vocal_trial'],limitations=completion['limitations']),
        human_feedback=dict(date='2026-10-17',
            user_quote='这版体感很不错定为R1版本，刚才我们提到的那些后处理中有些是适用于V2版本的，同时写好文档，V系列和R系列的区别。',
            status='positive_experience_user_selected',
            scope='Current delivered r2 version; user did not separately specify per-song acceptance',
            per_song_acceptance_not_inferred=True,automated_game_verification=False,broad_game_acceptance=False),
        playtests=packages,
        naming_note='Official uppercase R1 is the selected recipe, not old vocal experiment r1 or playtest-postprocess-r1.1.',
        desktop_default_pipeline_changed=False,
        entrypoints=dict(generate='tools/explore_vocal_grid_r2.py',package='tools/package_grid_playtest_r2.py',
                         review='tools/build_vocal_grid_review_r2.py'),
        registration_command=[sys.executable,*sys.argv],registration_tool_sha256=sha(Path(__file__)))
    create_json(target/'manifest.json',manifest)
    create_json(target/'registration-audit.json',dict(status='passed',before_registry_sha256=before_registry,
        base_manifest_unchanged=True,model_weights_unchanged=True,frozen_sources=len(frozen_sources),
        packages_byte_identical=True,manifest_sha256=sha(target/'manifest.json'),
        historical_evidence_unchanged=True,new_training=False,new_inference=False))
    if sha(registry_path)!=before_registry or sha(baseline_path)!=completion['baseline_manifest_sha256']:
        raise ValueError('Registry or base changed during registration')
    registry.update(schema_version=2,current_recipe='R1')
    registry.setdefault('recipes',{})['R1']=dict(manifest='recipes/R1/manifest.json',base_model='v2',
        manifest_sha256=sha(target/'manifest.json'),status='user_selected_playtest_recipe')
    registry['selection_note']='current_model selects the V base; current_recipe selects its downstream recipe. Registration does not change the desktop generation pipeline.'
    temporary=ROOT/'models/registry.r1.next.json'
    create_json(temporary,registry)
    temporary.replace(registry_path)
    print(json.dumps(dict(model='v2',recipe='R1',manifest_sha256=sha(target/'manifest.json'),
                         weights='unchanged',source_files=len(frozen_sources),packages=packages),ensure_ascii=False))


if __name__=='__main__':main()
