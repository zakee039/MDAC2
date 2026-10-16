"""Assemble recorded experiment evidence without running or tuning a model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric_rows(validation,test):
    rows = [('采音','sampling'),('地面轨道','ground'),('空中轨道','air')]
    result = []
    for label,key in rows:
        a = validation['sampling'] if key=='sampling' else validation['lanes'][key]
        b = test['sampling'] if key=='sampling' else test['lanes'][key]
        result.append(f'| {label} | {a["precision"]:.4f} | {a["recall"]:.4f} | {a["f1"]:.4f} | {b["precision"]:.4f} | {b["recall"]:.4f} | {b["f1"]:.4f} |')
    for kind,label in [('light','轻击'),('heavy','重击'),('extra_heavy','特重击'),('gemini','Gemini'),('hold','长条头（成功配尾后的物件）'),('gear','齿轮')]:
        a,b = validation['six_classes'][kind],test['six_classes'][kind]
        result.append(f'| {label} | {a["precision"]:.4f} | {a["recall"]:.4f} | {a["f1"]:.4f} | {b["precision"]:.4f} | {b["recall"]:.4f} | {b["f1"]:.4f} |')
    for kind,label in [('hold_tail','长条尾'),('hold_joint','长条头尾联合')]:
        a,b = validation['holds'][kind],test['holds'][kind]
        result.append(f'| {label} | {a["precision"]:.4f} | {a["recall"]:.4f} | {a["f1"]:.4f} | {b["precision"]:.4f} | {b["recall"]:.4f} | {b["f1"]:.4f} |')
    return '\n'.join(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    args = parser.parse_args()
    root = args.experiment.resolve()
    selected = read(root/'selection.json')
    config = read(root/selected['selected_config'])
    formal = read(root/'formal-lr3e4/status.json')
    overfit = read(root/'overfit-lr3e4/report.json')
    resume = read(root/'resume-check-v2.json')
    validation = read(root/'evaluation-lr3e4/validation.json')
    test = read(root/'evaluation-lr3e4/locked-test.json')
    frozen = read(root/'evaluation-lr3e4/frozen.json')
    review = read(root/'review-v2/review.json')
    audit = read(root/'final-input-audit.json')
    backend_log = (root/'backend-tests.log').read_text(encoding='utf-8')
    if '529 passed' not in backend_log or not audit['engineering_ready'] or not overfit['passed'] or not resume['passed']:
        raise ValueError('Final engineering evidence incomplete')
    if selected['frozen_sha256'] != sha(root/'evaluation-lr3e4/frozen.json') or test['frozen_sha256'] != selected['frozen_sha256']:
        raise ValueError('Frozen identity differs across selection and final test')
    snapshot = root/'source-final'
    snapshot.mkdir(exist_ok=True)
    source_paths = [ROOT/'src/mdac2/data_preparation'/name for name in
                    ('supervised_training.py','joint_decoding.py','joint_evaluation.py','joint_model.py','joint_losses.py','window_dataset.py')]
    source_paths += [ROOT/'tools'/name for name in
                     ('train_joint_supervised.py','evaluate_joint_supervised.py','build_supervised_review.py','check_supervised_resume.py','report_joint_supervised.py')]
    source_hashes = {}
    for path in source_paths:
        shutil.copy2(path,snapshot/path.name)
        source_hashes[str(path.relative_to(ROOT))] = sha(path)
    summary = dict(engineering_training_completed=True,model_quality_passed=False,human_listening='pending',
        game_acceptance='pending',production_promoted=False,selection=selected,training=formal,
        data=dict(manifest_sha256=audit['manifest_sha256'],eligible=audit['eligible_by_role'],quarantined=127,
                  windows=audit['windows_by_role'],feature_audio=217,preparation_inputs_unchanged=True,
                  final_audit_seconds=audit['audit_seconds']),
        overfit=dict(passed=overfit['passed'],steps=overfit['steps'],loss_reduction=overfit['loss_reduction'],
                     onset_f1=overfit['history'][-1]['onset']['f1'],tail_f1=overfit['history'][-1]['tail']['f1']),
        resume=resume,checkpoint_sha256=frozen['checkpoint_sha256'],decoder=frozen['decoder'],
        validation=validation['metrics'],locked_test=test['metrics'],backend_tests='529 passed; two existing warnings',
        source_sha256=source_hashes,review_examples=len(review['examples']),review_players=32,
        review_browser='representative mix/vocals/accompaniment/offline prediction mix playable; not a quality judgment')
    (root/'result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    table = metric_rows(validation['metrics'],test['metrics'])
    raw = read(root/'evaluation-lr3e4/threshold-search.json')
    raw_hold = next(s for s in raw['six_classes']['hold']['grid'] if s['threshold']==raw['six_classes']['hold']['chosen'])
    raw_tail = next(s for s in raw['tails']['grid'] if s['threshold']==raw['tails']['chosen'])
    examples = '\n'.join(f'| {e["focus"]} | {e["title"]} | {e["start"]:.3f}–{e["end"]:.3f} | {sum(e["reference_counts"].values())} | {sum(e["predicted_counts"].values())} | 待听评 |' for e in review['examples'])
    text = f'''# MDAC2 联合监督训练实验 v1

2026-09-27，工程根 `E:\\project\\MDAC2`。本轮训练与自动评价实际完成；模型质量未通过，真人听感和真实游戏可用性仍待验收，未晋升生产模型。

## 数据与环境

沿用固定 v3 清单，SHA-256 `{audit['manifest_sha256']}`。训练／验证／锁定测试为172／22／25张，1759／217／251窗；127张问题谱保持隔离，217份三流全曲缓存复用。开始和结束时完整身份、有限数值、中心mask及角色检查均通过，结束复核耗时{audit['audit_seconds']:.2f}秒。准备批次中的 `training_started=false` 是历史准备阶段声明，没有被本轮覆盖；当前训练状态在新实验目录。

仅训练一个 joint-tcn-v1 联合模型，826,078参数。固定人声分离权重不训练；原曲／人声／伴奏、时基及难度为网络输入，事件ID／来源代码／作者／谱名均不进入网络。Python3.13.12，torch2.14.0+cu130，RTX5060 Laptop，bf16前向＋float32损失，确定性算子，`CUBLAS_WORKSPACE_CONFIG=:4096:8`，种子20260927。环境完整记录在 `formal-lr3e4/config.json`。

## 训练结果及选择

- 小批：训练集窗口0与2覆盖六类／尾点／占用。550次更新通过，loss {overfit['initial_loss']:.6f}→{overfit['final_loss']:.6f}，下降{overfit['loss_reduction']:.2%}；起点F1 {overfit['history'][-1]['onset']['f1']:.4f}，尾点F1 {overfit['history'][-1]['tail']['f1']:.4f}，输出及梯度有限，峰值显存{overfit['cuda_peak_bytes']/2**20:.2f}MiB。正式训练重新初始化。代码修复后重新运行诊断绑定新hash，没有沿用诊断权重。
- 第一版 LR=0.001、最多30轮／patience5，在第7轮早停，记录最佳第2轮。0.5阈值退化为不出音，低阈值造成大量误报，验证六类macro F1仅0.02918，无有效配对长条。旧实现将min_delta同时用于最佳保存，错过第5轮的小幅最低loss；该缺陷已在后续训练器中修复，旧结果和源码留在 `formal`／`source-baseline`。
- 对照保持相同架构与训练角色权重，仅降低LR至0.0003、patience10、min_delta0.0001；AdamW weight_decay0.0001，batch8，gradient clip1.0，最多30轮。训练完成{formal['epoch']}轮／{formal['global_step']}次更新，{formal['seconds_this_invocation']:.2f}秒，峰值显存{formal['cuda_peak_bytes']/2**20:.2f}MiB。最低验证loss为{formal['best']:.8f}，在第{formal['best_epoch']}轮；最佳保存使用实际最低loss，早停单独使用显著改善anchor，ReduceLROnPlateau降低学习率。
- 最终选择依据：验证六类macro F1由0.02918升至{validation['metrics']['macro_class_f1']:.5f}，同时保留采音、轨道、长条及稀有类失败信息。没有以总loss或采音召回替代排键质量。
- checkpoint：`formal-lr3e4/best.pt`，SHA-256 `{frozen['checkpoint_sha256']}`；最新可恢复状态为 `formal-lr3e4/last.pt`，包含模型／优化器／scheduler／RNG／epoch内游标／早停状态。取消文件每批检查，有限日志每50步及每轮落盘，原子checkpoint。
- CUDA实际取消／恢复：在tiny fixture第2步取消，恢复完成4步，与连续训练的模型及优化器逐位一致；变更数据身份被拒绝。实际报告 `resume-check-v2.json`，不是只用mock optimizer验证。

## 冻结与评价

六类阈值按light／heavy／extra_heavy／gemini／hold／gear顺序为 `{frozen['decoder']['thresholds']}`，尾点0.5，占用0.2；NMS半径3帧，跨轨配对容差20ms，导出明确量化至1ms。阈值先用验证集原始逐轨head／tail事件F1选择，再用成功头尾联合F1选择占用阈值；不是全组合全局最优搜索。验证未配对的原始长条头F1为{raw_hold['f1']:.4f}，原始尾点F1为{raw_tail['f1']:.4f}，下表为配对及规则后物件指标。

`frozen.json`时间{frozen['frozen_at']}，随后选定方案写入selection.json，再打开锁定测试。锁定测试25首仅评价此方案一次，产物 `locked-test.json`；没有使用测试重新选择权重、参数、阈值或片段。测试后的低分只作为失败证据。

| 项目 | 验证P | 验证R | 验证F1 | 锁定测试P | 锁定测试R | 锁定测试F1 |
|---|---:|---:|---:|---:|---:|---:|
{table}

采音只看不同起点时间；同刻Gemini／双重击仅算一个采音点。轨道忽略类别；六类／长条按实际轨道实体计数，所以双轨Gemini和双轨Hold分别有两个监督实体。起点匹配50ms，尾点100ms；另保存25ms严格指标。已排除幽灵／道具负监督屏蔽及量化冲突格。社区参考并非人工gold。

特重击验证 TP/FP/FN={validation['metrics']['extra_heavy']['tp']}/{validation['metrics']['extra_heavy']['fp']}/{validation['metrics']['extra_heavy']['fn']}，误报{validation['metrics']['extra_heavy']['false_positives_per_minute']:.5f}/分钟；锁定测试={test['metrics']['extra_heavy']['tp']}/{test['metrics']['extra_heavy']['fp']}/{test['metrics']['extra_heavy']['fn']}，误报{test['metrics']['extra_heavy']['false_positives_per_minute']:.5f}/分钟。测试17个全部漏检，不能说稀有类已经学会。

规则拒绝率验证{validation['metrics']['rule_rejection_rate']:.2%}、锁定测试{test['metrics']['rule_rejection_rate']:.2%}。跨16秒中心窗的预测长条分别保留{sum(s['cross_window_holds'] for s in validation['songs'])}／{sum(s['cross_window_holds'] for s in test['songs'])}个，这是解码行为证据，不是这些长条已匹配正确的证明。

## 拼接、规则及审阅

整曲先按中心窗口唯一拥有每帧输出，再全曲做局部峰值、偏移、尾点与占用配对。没有窗口边缘造尾点，也不设跨窗持续上限。缺尾／单侧Gemini有拒绝记录。Gemini为一个双轨逻辑物件；双重击为两个heavy实体及共有group_id；头尾均一致时合并双轨Hold。特重击仍按原契约同刻双轨排他，Hold占用为半开区间，未增加额外留白。

初始全量重复规则检查在大量低阈值候选下过慢；中断该评价进程后复用身份有效的网络预测，改为时间索引相关候选检查。随机重叠对照测试确认接受物件和拒绝ID集合与原选择器一致。全部最终预测重新通过既有validator。

最新试听材料在 `review-v2/预测听评.html`：8组验证集片段、32个原曲／人声／伴奏／原曲＋预测击打播放器、双轨参考／预测时间轴、6首实验BMS及完整逻辑旁表。v1的按钮在浏览器技术检查中未触发，v2改为离线24kHz波形叠加，直接使用媒体控件；代表片段三流及叠加都已在浏览器加载／播放，三流互斥播放及游标联动正常。仍没有独立真人听评，BMS回读通过也不是游戏资源包或游戏加载验收。

| 场景 | 曲目 | 全曲秒区间 | 参考逻辑物件 | 预测逻辑物件 | 听感 |
|---|---|---|---:|---:|---|
{examples}

自动观察：人声起音候选片段有28个预测对24个参考，长音候选只保留1个跨入片段的长条、漏掉大量参考物件；间奏候选84对43、疑似漏音候选48对31，存在过量击打。以上是片段计数与形态，不证明特定物件来自人声或真的发生分离漏音；筛选依赖能量／flux／流相关性proxy，需要听辨确认。

## 实际命令与恢复

均在 `E:\\project\\MDAC2` 执行，GPU解释器简称 `$py = 'tools/.deps/pretraining-py313/Scripts/python.exe'`，实验根简称 `$r = 'tools/private/community/supervised-joint-v1-20260927'`，实际使用UTF-8控制台。首次完整preflight复核35.52秒，结束复核{audit['audit_seconds']:.2f}秒；完整命令保存在设计16。以下为实际监督运行命令（输出另重定向至同目录console日志）：

```powershell
& $py tools/train_joint_supervised.py --config "$r/baseline.json" --run-dir "$r/overfit" --overfit --steps 1000
& $py tools/train_joint_supervised.py --config "$r/baseline.json" --run-dir "$r/overfit-final" --overfit --steps 1000
& $py tools/check_supervised_resume.py "$r/resume-check.json"
& $py tools/train_joint_supervised.py --config "$r/baseline.json" --run-dir "$r/formal"
& $py tools/train_joint_supervised.py --config "$r/baseline-lr3e4.json" --run-dir "$r/overfit-lr3e4" --overfit --steps 1000
& $py tools/check_supervised_resume.py "$r/resume-check-v2.json"
& $py tools/train_joint_supervised.py --config "$r/baseline-lr3e4.json" --run-dir "$r/formal-lr3e4"
& $py tools/evaluate_joint_supervised.py --config "$r/baseline.json" --checkpoint "$r/formal/best.pt" --output "$r/evaluation"
& $py tools/evaluate_joint_supervised.py --config "$r/baseline-lr3e4.json" --checkpoint "$r/formal-lr3e4/best.pt" --output "$r/evaluation-lr3e4"
& $py tools/evaluate_joint_supervised.py --config "$r/baseline-lr3e4.json" --checkpoint "$r/formal-lr3e4/best.pt" --output "$r/evaluation-lr3e4" --locked-test
& $py tools/build_supervised_review.py --config "$r/baseline-lr3e4.json" --evaluation "$r/evaluation-lr3e4" --output "$r/review-v2"
& .venv/Scripts/python.exe -m pytest -q
```

528/529项? 以最终实际输出为准：本次最终 **529 passed、2条既有警告**，日志 `backend-tests.log`；新增8项整曲／规则／指标测试全部通过。CPU工作流未安装Torch，GPU环境保持隔离。源码快照与hash在source-final／result.json，首版旧训练器另留source-baseline。

安全取消／恢复用法（恢复是操作说明，本轮正式训练未被取消；实际恢复实验见CUDA fixture）：

```powershell
New-Item -ItemType File "$r/formal-lr3e4/CANCEL"
# 等待status.json变为cancelled并确认last.pt已落盘
Remove-Item -LiteralPath "$r/formal-lr3e4/CANCEL"
& $py tools/train_joint_supervised.py --config "$r/baseline-lr3e4.json" --run-dir "$r/formal-lr3e4" --resume "$r/formal-lr3e4/last.pt"
```

恢复严格核对代码／数据／模型／配置，不能改max_epochs或LR后冒充同一断点。已完成30轮的此配置恢复不会额外增加训练。新实验需新目录，不能覆盖准备批次。

本地试听服务只监听127.0.0.1:8773，保留用于当前页面；服务PID在review-v2-server.pid，关闭服务后HTML及WAV仍可在本机直接打开。未上传素材，没有Git提交／推送（当前目录不是Git仓库），未自动晋升或改生产profile。

## 未解决问题

1. 六类宏平均F1仅验证0.155／测试0.136，严重类别与轨道误报，不能把采音F1或低loss当成可用编排。
2. 长条头尾联合F1很低，常错误配尾／输出1ms长条，或遗漏长音；后续需同时检查尾点学习、占用边界和配对方法。跨窗保留不等于持续时间正确。
3. 特重击测试0/17召回且5次误报；稀有训练参考206个，泛化失败。齿轮、Gemini和重击也仍偏弱。
4. 常同拍输出多类或双轨击打。规则仅执行当前版本的排他／gear／长条冲突，不宣称完整可玩性过滤；不能为了抬分偷偷剪来源标签。
5. 人声起音、长音、间奏、漏音是自动筛选候选，未获得真人声部／音乐质感反馈。分离器权重固定，社区谱没有逐note声部真值。
6. 源谱时基用于本次评价；自动估拍到新音频的误差、生产六类UI、完整资源包、Boss场景和真实游戏加载均未验收。127张隔离谱不进入训练；锁定测试已使用，本轮后续不得据其反馈调参并再次声称独立评价。
'''
    text = text.replace('528/529项? 以最终实际输出为准：本次最终','本次最终')
    (root/'训练实验报告.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(report=str(root/'训练实验报告.md'),training_completed=True,quality_passed=False,source_files=len(source_hashes))))


if __name__ == '__main__':
    main()
