"""Register the user-approved r1.1 playtest combination without altering v1."""
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mdac2.data_preparation.audio_features import sha
from mdac2.data_preparation.playtest_postprocess import Policy


def main():
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    def save(p, data):
        with p.open('x', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write('\n')
    target = ROOT/'models/v2'
    if target.exists():
        raise FileExistsError('v2 exists; never overwrite a selected release')
    origin = ROOT/'tools/private/community/playability-postprocess-r1-20261016'
    completed = read(origin/'completion-r1.1.json')
    if completed['engineering'] != 'passed':
        raise ValueError('Approved artifacts have not passed engineering audit')
    prior = origin/'run-prior-r1.1.json'
    if sha(prior) != '6a27352cb2c0f9d6fc97f8f06d5742274d8c959a9fd5863fffad96568b875558':
        raise ValueError('Prior changed')
    manifest = read(ROOT/'models/v1/manifest.json')
    for item in manifest['checkpoints'].values():
        if sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('v1 checkpoint changed')
    module = ROOT/'src/mdac2/data_preparation/playtest_postprocess.py'
    if sha(module) != sha(origin/'source-r1.1'/module.name):
        raise ValueError('Approved postprocessing source changed')
    (target/'weights').mkdir(parents=True)
    (target/'source').mkdir()
    for stage, item in manifest['checkpoints'].items():
        dest = target/'weights'/f'{stage}.pt'
        shutil.copyfile(ROOT/item['path'], dest)
        if sha(dest) != item['sha256']:
            raise ValueError('Checkpoint copy failed')
        item['path'] = dest.relative_to(ROOT).as_posix()
    shutil.copyfile(prior, target/'weights/run-prior.json')
    sources = {}
    for path in (module, ROOT/'tools/refine_model_v1_playtest.py', ROOT/'tools/build_playtest_comparison.py'):
        dest = target/'source'/path.name
        shutil.copyfile(path, dest)
        sources[path.relative_to(ROOT).as_posix()] = dict(snapshot=dest.relative_to(ROOT).as_posix(), sha256=sha(dest))
    manifest.update(model_id='v2', created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        version_note='Release v2 = unchanged v1 neural weights + frozen playtest-postprocess-r1.1. Not historical training experiment v2.',
        parent_model='v1', postprocessing=dict(policy=asdict(Policy()),
            prior_path='models/v2/weights/run-prior.json', prior_sha256=sha(prior), sources=sources,
            evidence='tools/private/community/playability-postprocess-r1-20261016/completion-r1.1.json',
            evidence_sha256=sha(origin/'completion-r1.1.json')),
        human_feedback=dict(scope='Pipa Xing r1.1 positive real-play feedback; Snow Song Show has unspecified odd points, deferred',
            user_quote='这版琵琶行玩起来真人感真不错，snow song show有点奇怪的点，但可能是特例，后续再说',
            automated_game_verification=False, broad_game_acceptance=False),
        vocal_priority_experiment_included=False, desktop_default_pipeline_changed=False)
    validation = read(origin/'validation-r1.1/validation.json')['metrics']['after']
    manifest['validation'].update(actual_onset_five_class_macro_f1=validation['macro_five_class_f1'],
        actual_onset_five_class_micro_f1=validation['micro_five_class']['f1'],
        actual_onset_hold_joint_f1=validation['holds']['hold_joint']['f1'],
        objective_quality='Postprocessing improves structural statistics, not overall reference accuracy; reused validation, not independent test')
    manifest['playtests'] = []
    for slug, title in [('pipa', '琵琶行'), ('snow', 'Snow Song Show')]:
        source = Path(f'E:/下载/{title}_mikee_MDAC2试玩_模型v1_编排r1.1.mdm')
        dest = Path(f'E:/下载/{title}_mikee_MDAC2试玩_模型v2.mdm')
        expected = ('8673aeacc2f9e14624fff1d643203dd79743d329032c2e0c1bbf2cf6c6297df1' if slug == 'pipa'
                    else 'eaf5dd9f453710b10d777d5fb2518328c5d24a6b69a6593bd161817bd7d3e27d')
        if sha(source) != expected:
            raise ValueError('Approved package changed')
        with dest.open('xb') as f, source.open('rb') as g:
            shutil.copyfileobj(g, f)
        if sha(dest) != expected:
            raise ValueError('Package copy failed')
        manifest['playtests'].append(dict(song=title, sha256=expected, delivery=str(dest),
            byte_identical_to_approved_r1_1=True, internal_title_preserved=True))
    manifest.pop('playtest', None)
    save(target/'manifest.json', manifest)
    (target/'README.md').write_text('# 模型发行 v2\n\n'
        '用户认可的当前组合：v1 两份神经网络权重 + 编排后处理 r1.1。权重未重新训练；发行编号与历史训练实验编号分开。\n\n'
        'weights/onset.pt、weights/arrangement.pt、weights/run-prior.json 是本地冻结副本，不进入 Git。manifest.json 记录完整 SHA、策略、源码快照与试听范围。\n\n'
        '《琵琶行》获得正向实机反馈；《Snow Song Show》仍有用户报告的个别奇怪之处。客观分类质量未显著提高，不能扩大为所有歌曲验收。\n\n'
        '人声优先采音属于后续独立实验，不包含在 v2。保留 v1、旧包和 v0.1.0 tag，桌面默认生成入口尚未替换。\n', encoding='utf-8')
    registry = read(ROOT/'models/registry.json')
    registry['current_model'] = 'v2'
    registry['models']['v2'] = dict(manifest='models/v2/manifest.json', application_version='0.1.0',
                                  status='user_selected_playtest_baseline')
    temporary = ROOT/'models/registry.next.json'
    save(temporary, registry)
    temporary.replace(ROOT/'models/registry.json')
    print(json.dumps(dict(model='v2', manifest_sha256=sha(target/'manifest.json'), weights='unchanged', packages=2)))


if __name__ == '__main__':
    main()
