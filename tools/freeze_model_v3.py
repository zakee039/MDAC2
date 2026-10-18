"""Freeze the user-selected strict global pipeline as V3, preserving V2."""
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mdac2.data_preparation.audio_features import sha
from mdac2.data_preparation.six_export import read_bms,semantics


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    with p.open('x',encoding='utf-8') as f:
        json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    target=ROOT/'models/v3'
    origin=ROOT/'tools/private/community/global-1ms-conflicts-v2-20261017'
    delivery=Path('E:/OneDrive - stu.sau.edu.cn/MuseDash/Custom_Albums')
    if target.exists():raise FileExistsError('V3 already exists; never overwrite a frozen release')
    titles={'pipa':'琵琶行','snow':'Snow Song Show'}
    for title in titles.values():
        if (delivery/f'{title}_mikee_MDAC2试玩_V3.mdm').exists():raise FileExistsError('V3 package already exists')
    # Verify the complete V2 directory before and after freezing, not just weights.
    v2_files={p:sha(p) for p in (ROOT/'models/v2').rglob('*') if p.is_file()}
    v2=read(ROOT/'models/v2/manifest.json')
    assert sha(ROOT/'models/v2/manifest.json')=='9ae31c1894420492e4eae64cf88989d3f199523d00b78dbc7801c30642896562'
    complete=read(origin/'completion.json')
    assert complete['status']=='completed' and complete['charts_checked']==24
    replay=read(origin/'replay/identity.json')
    # The selection adapter and default model name change only for this release.
    for name,digest in replay['sources'].items():
        if Path(name).name not in ('generate_current_playtest.py','generation-defaults.json'):
            assert sha(ROOT/name)==digest,(name,'evaluated source changed')
    checkpoints=deepcopy(v2['checkpoints'])
    for c in checkpoints.values():assert sha(ROOT/c['path'])==c['sha256']
    prior=ROOT/v2['postprocessing']['prior_path'];assert sha(prior)==replay['prior_sha256']
    old_packages={}
    for slug in titles:
        p=read(origin/f'{slug}-package/package-report.json')
        path=Path(p['delivery']);assert sha(path)==p['package_sha256'];old_packages[path]=p['package_sha256']
    (target/'weights').mkdir(parents=True);(target/'source').mkdir();(target/'evidence').mkdir()
    for stage,c in checkpoints.items():
        dest=target/'weights'/f'{stage}.pt';shutil.copyfile(ROOT/c['path'],dest)
        assert sha(dest)==c['sha256'];c['path']=dest.relative_to(ROOT).as_posix()
    shutil.copyfile(prior,target/'weights/run-prior.json')
    names=set(replay['sources'])
    names.update(['src/mdac2/data_preparation/v2_vocal_bias.py','src/mdac2/data_preparation/vocal_priority.py',
                  'src/mdac2/data_preparation/vocal_priority_r2.py','tools/package_predicted_playtest.py',
                  'tools/freeze_model_v3.py'])
    sources={}
    for name in sorted(names):
        p=ROOT/name;dest=target/'source'/Path(name)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        sources[p.relative_to(ROOT).as_posix()]=dict(snapshot=dest.relative_to(ROOT).as_posix(),sha256=sha(dest))
    evidence={}
    for name in ['completion.json','replay/identity.json','replay/comparison.json']:
        dest=target/'evidence'/Path(name).name;shutil.copyfile(origin/name,dest)
        evidence[name]=dict(snapshot=dest.relative_to(ROOT).as_posix(),sha256=sha(dest))
    packages=[]
    for slug,title in titles.items():
        p=read(origin/f'{slug}-package/package-report.json');source=Path(p['delivery'])
        with zipfile.ZipFile(source) as z:
            assert z.testzip() is None
            old={name:z.read(name) for name in z.namelist()}
        data=dict(old);info=json.loads(data['info.json']);info['name']=title+' - V3'
        data['info.json']=(json.dumps(info,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
        text=data['map2.bms'].decode('utf-8-sig')
        lines=text.splitlines();assert sum(s.startswith('#TITLE ') for s in lines)==1
        data['map2.bms']=('\n'.join('#TITLE '+title+' - V3' if s.startswith('#TITLE ') else s for s in lines)+'\n').encode('utf-8')
        assert semantics(read_bms(text,p['duration']))==semantics(read_bms(data['map2.bms'].decode('utf-8'),p['duration']))
        assert all(data[k]==old[k] for k in old if k not in ('info.json','map2.bms'))
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for name,b in sorted(data.items()):
                entry=zipfile.ZipInfo(name,(1980,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED;z.writestr(entry,b)
        path=delivery/f'{title}_mikee_MDAC2试玩_V3.mdm'
        with path.open('xb') as f:f.write(buffer.getvalue())
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None and all(z.read(k)==b for k,b in data.items())
        packages.append(dict(song=title,delivery=str(path),sha256=sha(path),parent_sha256=p['package_sha256'],
            events=p['logical_events'],chart_and_audio_unchanged=True,only_display_titles_changed=True,
            game_verified=False,human_listening='pending_for_this_strict_revision'))
    metrics=read(origin/'replay/comparison.json')['metrics']['strict']
    manifest=dict(schema_version=1,model_id='v3',product='MDAC2',application_version='0.1.0',release_date='2026-10-18',
        status='user_selected_playtest_baseline',parent_model='v2',parent_manifest_sha256=sha(ROOT/'models/v2/manifest.json'),
        version_note='User-selected V3 = unchanged V2 network weights + soft vocal bias + strict global 1ms postprocessing. No retraining.',
        checkpoints=checkpoints,checkpoint_format=v2['checkpoint_format'],inference=dict(v2['inference'],
            selection_profile='v2-vocal-soft-bias-v1',maximum_vocal_odds_multiplier=2,storage_step_seconds=.001),
        postprocessing=dict(policy=read(origin/'replay/pipa/report.json')['policy'],
            prior_path='models/v3/weights/run-prior.json',prior_sha256=sha(prior),sources=sources,evidence=evidence),
        validation=dict(dataset_manifest_sha256=replay['data_manifest_sha256'],train_charts=172,validation_charts=22,
            previously_used_test_charts=25,quarantined_charts=127,metrics=metrics,
            objective_quality='Reused validation; stricter deletion slightly lowers reference matching. Not an independent test.'),
        human_feedback=dict(selection_quote='更改为v3，保留v2',
            prior_revision_quote='嗯，整体上还是不错的',
            scope='V3 designation authorized; newest strict transfer/drop revision has no explicit real-play feedback yet.',
            automated_game_verification=False,broad_game_acceptance=False),
        playtests=packages,neural_weights_retrained=False,desktop_default_pipeline_changed=False,
        supervised_generation_entrypoint='tools/generate_current_playtest.py',defaults=read(ROOT/'recipes/generation-defaults.json'),
        application_tag_changed=False,weight_distribution='Local weights excluded from Git; no music or training data distributed.')
    save(target/'manifest.json',manifest)
    (target/'README.md').write_text('# 模型发行 V3\n\n2026-10-18用户指定当前组合为V3，V2完整保留。\n\n'
        'V3 = V2原神经权重 + 柔性人声采音偏好 + 全局1ms严格避让后处理。不是重新训练的网络。\n\n'
        'weights/onset.pt、arrangement.pt与V2对应SHA完全相同，run-prior.json也是冻结副本。'
        'source保存本次完整执行源码，manifest记录配置、评价与包身份。\n\n'
        '新监督制谱入口tools/generate_current_playtest.py读取V3；桌面手工ChartIR流程不变。'
        '冻结R1仍依赖V2，不自动组合到V3。应用仍为v0.1.0，不新增Git tag。\n\n'
        '最新两份V3包仅更改游戏内标题，谱面/音频与全局避让修订1相同；该修订待用户实机反馈。\n',encoding='utf-8')
    assert all(sha(p)==digest for p,digest in v2_files.items())
    assert set(v2_files)=={p for p in (ROOT/'models/v2').rglob('*') if p.is_file()}
    assert all(sha(p)==digest for p,digest in old_packages.items())
    registry=read(ROOT/'models/registry.json');registry['current_model']='v3'
    registry['models']['v3']=dict(manifest='models/v3/manifest.json',manifest_sha256=sha(target/'manifest.json'),
        application_version='0.1.0',status='user_selected_playtest_baseline')
    registry['last_selected_recipe']=registry.get('current_recipe');registry['current_recipe']=None
    registry['selection_note']='V3 includes its frozen default pipeline. No R override active. Historical R1 still explicitly depends on V2; V2 and its artifacts remain unchanged. Manual ChartIR is separate.'
    registry['active_generation_defaults']['status']='user_selected_v3_baseline'
    temp=ROOT/'models/registry.next.json';save(temp,registry);temp.replace(ROOT/'models/registry.json')
    save(target/'registration-audit.json',dict(v2_directory_unchanged=True,v2_file_hashes={str(p.relative_to(ROOT)):d for p,d in v2_files.items()},
        parent_packages_unchanged=True,manifest_sha256=sha(target/'manifest.json'),packages=packages,
        command=[sys.executable,*sys.argv]))
    print(json.dumps(dict(model='v3',v2_preserved=True,packages=packages),ensure_ascii=False))


if __name__=='__main__':main()
