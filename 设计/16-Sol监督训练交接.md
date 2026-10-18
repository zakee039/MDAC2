# Sol监督训练交接

**2026-09-28当前状态：双模型基线实现、过拟合、GPT-6 Sol正式监督训练、22首验证及诊断预览全部完成；模型质量未通过，尤其自由历史排键退化。无训练进程或自动续训，听感待评、游戏未验收、未晋升。当前权威见设计17和第8节完成记录，下文单模型／暂停均为历史。**

2026-09-27，工程根 `E:\project\MDAC2`。监督基线训练与自动评价已完成，但模型质量失败。用户已明确v1/v2试玩很差，并更正此前误把预览中的参考谱当成预测，撤回初步正面听感。当前按用户最新要求转为**诊断优先，暂停继续改模型、调阈值和出新包**。只读故障定位及未验收对照实验见第7节；旧训练、准备批次和试玩包保留，未晋升生产模型。

**首轮工程验收记录（历史）：** 开始／结束完整preflight复核分别35.52／35.51秒，全部2,227窗、217全曲缓存和219份标签／张量身份仍通过；角色172／22／25及127张隔离谱未变，未重算分轨。训练集小批过拟合550步通过；CUDA取消／恢复后模型和优化器与连续运行逐位一致。正式对照完成30轮／6,600步，最佳第27轮；后端最终529 passed、2条既有警告。验证采音F1=0.57846、六类macro F1=0.15545；冻结后一次锁定测试分别0.64797／0.13585，特重击0/17召回、5次误报，长条头尾联合F1=0.02683。预测页为8组／32个播放器，三流及离线预测叠加做过代表片段浏览器播放检查；当时记录的听评pending现已被用户负反馈更新，见第7节。首轮证据见 [训练实验报告](../tools/private/community/supervised-joint-v1-20260927/训练实验报告.md)。

## 1. 历史首轮：模型数量与任务

**训练1个联合制谱模型，同时学习采音和排键。** 模型共享主干，输出地空两轨六类起点、细化落点、长条尾点及占用；排键不是纯规则。规则只用于物件合法性与冲突选择。现成MelBandRoformer用于冻结分轨，BeatThis沿用既有节拍流程，这轮不训练它们。

已实现配置及张量形状见 [15-联合制谱模型与训练数据契约](15-联合制谱模型与训练数据契约.md)。`joint-tcn-v1` 有826,078参数，训练器从明确种子随机初始化；实验checkpoint需显式加载，未更改生产默认模型。人声优先是软融合先验，不是“音量大就生成note”，声部偏好尚无人工质量结论。

## 2. 固定数据入口

下表路径相对工程根；不要把旧的320张候选或v1/v2派生批次当成当前训练集。

| 项目 | 路径／数量 |
|---|---|
| 当前标签与角色清单 | `tools/private/community/training-preparation-v3/manifest.json` |
| 清单SHA-256 | `34add12c1c47fc99afc04ae1fa9f06e968d5c2c50db41cb32a047f033668eb71` |
| 候选范围 | 8.5～10.5，346张；219张自动检查可用，127张隔离 |
| 可用角色 | 训练172、验证22、锁定测试25张 |
| 全曲分轨及三流mel缓存 | `tools/private/community/pretraining-feature-cache-v1`，217份不同音频，约9.35小时 |
| 监督张量及训练角色权重建议 | `tools/private/community/training-preparation-v3/tensors/report.json`，219份NPZ |
| 32秒窗口 | 中心16秒监督；训练1759、验证217、锁定测试251窗 |
| 归并实例 | `tools/private/community/training-preparation-v3/归并审阅.html`，12个训练角色片段 |
| 三流试听 | 同批次 `分轨审阅.html`，原曲／人声／伴奏对照；其中一例属于隔离谱，仅提供原曲并明确标注 |
| 导出回读 | `tools/private/community/training-preparation-v3/six-export-audit-v2.json`，219/219通过 |
| 相近录音排查 | `tools/private/community/audio-family-audit-v1/report.json`，338份全量候选音频、0跨角色候选；其v2角色与v3一致，由preflight复核 |
| 最终前向检查 | 同批次 `forward-cpu.json`、`forward-cuda.json`，仅真实训练窗前向及loss数值检查 |
| 最终全量工程检查 | 同批次 `preflight.json`，校验文件身份、各角色每个窗口和以上证据 |
| 损失数值契约 | 同批次 `loss-contract.json`，6项无梯度分析式检查 |

原始副本、所有503包的展开内容及572张BMS继续保留在 `tools/private/community/20260926-preparation`。被隔离的127张也有标签还原结果和原因，没有删除或通过规则偷偷裁剪成“可用”。新清单固定整首角色后才切窗，同PCM／同压缩音频／同临时家族不跨角色。

## 3. 可直接使用的环境和接口

GPU解释器：`E:\project\MDAC2\tools\.deps\pretraining-py313\Scripts\python.exe`。Python3.13.12，torch2.14.0+cu130；RTX5060 Laptop约8GiB。PyTorch来自用户的D:/TORCH，环境与已有CPU工作流隔离。依赖冻结在 `tools/experiments/separation/requirements-pretraining-py313-gpu.lock`。

核心接口位于 `src/mdac2/data_preparation/`：

- `window_dataset.WindowDataset(manifest, cache, 'train', tensor_cache=tensors)`：默认拒绝未完成、身份不符或覆盖不足的缓存；验证角色单独实例化。Windows第一版可先用DataLoader `num_workers=0`，减少重复模型外缓存与子进程问题。
- `joint_model.JointChartModel()`：输入 `features` 为B×3×96×3200，`timing` 为B×3200×4。其他batch字段均为监督目标／mask，不能喂给模型。
- `joint_losses.joint_losses(output, batch, positive_weights=..., hold_end_positive_weights=...)`：权重由训练角色张量报告读取，转换至当前设备；不要用验证／测试分布重算。损失权重只是初始建议。
- `candidate_rules`、`training_chart`、`six_export`：候选规则选择、事件验证、实验六类导出。完整整曲输出解码器和训练器尚需Sol实现。

所有工具入口自行加入src路径。直接写训练脚本时也需加入工程src，或设置PYTHONPATH；不要误用旧MDAC目录。

复核命令（在工程根执行；仅前向，绝不训练）：

```powershell
tools/.deps/pretraining-py313/Scripts/python.exe tools/check_training_forward.py --manifest tools/private/community/training-preparation-v3/manifest.json --cache tools/private/community/pretraining-feature-cache-v1 --tensor-cache tools/private/community/training-preparation-v3/tensors --device cuda --output tools/private/community/supervised-joint-v1-20260927/forward-cuda-recheck-new.json
```

完整数据复核（还需同批次已有CPU前向报告；输出只写检查报告）：

```powershell
.venv/Scripts/python.exe tools/check_pretraining_ready.py --manifest tools/private/community/training-preparation-v3/manifest.json --cache tools/private/community/pretraining-feature-cache-v1 --tensors tools/private/community/training-preparation-v3/tensors --acoustic tools/private/community/audio-family-audit-v1/report.json --acoustic-manifest tools/private/community/training-preparation-v2/manifest.json --export-report tools/private/community/training-preparation-v3/six-export-audit-v2.json --forward tools/private/community/training-preparation-v3/forward-cpu.json --forward tools/private/community/training-preparation-v3/forward-cuda.json --output tools/private/community/supervised-joint-v1-20260927/preflight-recheck-new.json
```

准备阶段曾使用 `--allow-partial-cache` 验证已完成训练窗；正式训练不可使用该调试选项。锁定测试默认拒绝打开，只有最终评价才显式 `allow_locked_test=True`；preflight的测试角色访问只是结构检查，没有模型预测、调参或质量指标。

## 4. 已完成监督工作与本轮交接

1. 训练器 `supervised_training.py` 和入口 `tools/train_joint_supervised.py`：确定性种子、训练角色权重、有限轮数、早停、最低验证loss checkpoint、原子保存、优化器／scheduler／RNG／epoch游标恢复及CANCEL文件。首次LR0.001退化基线完整保留；修复最佳保存与min_delta混用后，以LR0.0003对照改善，架构及数据不变。
2. 小批诊断只打开train角色，窗口0和2覆盖六类，loss下降96.55%，起点F1=0.92593、尾点F1=0.92857；正式训练不继承诊断权重。真实CUDA fixture验证取消／恢复，报告 `resume-check-v2.json`。
3. `joint_decoding.py` 先拼接各窗中心且每帧覆盖恰好一次，再全曲解码、尾点＋占用配对；保留跨窗长条，无窗口造尾。Gemini是一对逻辑物件，双重击为两个heavy实体＋group_id。规则加时间索引，随机重叠对照原选择器通过，未改准备阶段规则或剪来源标签。
4. 参数和阈值只从train／validation选择，`selection.json`及`evaluation-lr3e4/frozen.json`绑定权重／数据／源码。冻结后25首locked_test已一次评价，报告 `locked-test.json`。六类阈值为[0.6,0.6,0.6,0.6,0.5,0.5]，尾点0.5，占用0.2；此测试现已使用，不得回头据测试调参并再次声称独立评估。
5. 审阅页 `review-v2/预测听评.html` 提供三流、离线24kHz预测击打叠加、双轨时间轴、6份实验BMS及逻辑旁表。自动筛选的起音／长音／间奏／疑似漏音片段有计数与proxy，尚无真人听评，不宣称声部真值或分离漏音已确认。当前模型质量未通过，不接生产晋升。

实验配置：`baseline-lr3e4.json`；最佳权重 `formal-lr3e4/best.pt`，SHA-256 `78407992ac57b4c7953b77c49cb8e85025fbed3e06015380b07de13e943bb3a6`；可恢复状态 `formal-lr3e4/last.pt`；每50步／每轮日志 `metrics.jsonl`，最终 `status.json` 为completed。AdamW、batch8、LR0.0003、weight_decay0.0001、gradient clip1.0、最多30轮／patience10／min_delta0.0001、bf16前向／float32 loss、seed20260927。运行530.16秒，峰值PyTorch分配525.75MiB。最低验证loss0.26085230；保存实际最低值，早停显著改善另计anchor。

实际主命令（均在本工程根；完整首次诊断／失败对照／评价命令和输出在实验报告）：

```powershell
$env:PYTHONUTF8='1'
$py = 'tools/.deps/pretraining-py313/Scripts/python.exe'
$r = 'tools/private/community/supervised-joint-v1-20260927'
& $py tools/train_joint_supervised.py --config "$r/baseline-lr3e4.json" --run-dir "$r/overfit-lr3e4" --overfit --steps 1000
& $py tools/check_supervised_resume.py "$r/resume-check-v2.json"
& $py tools/train_joint_supervised.py --config "$r/baseline-lr3e4.json" --run-dir "$r/formal-lr3e4"
& $py tools/evaluate_joint_supervised.py --config "$r/baseline-lr3e4.json" --checkpoint "$r/formal-lr3e4/best.pt" --output "$r/evaluation-lr3e4"
& $py tools/evaluate_joint_supervised.py --config "$r/baseline-lr3e4.json" --checkpoint "$r/formal-lr3e4/best.pt" --output "$r/evaluation-lr3e4" --locked-test
& $py tools/build_supervised_review.py --config "$r/baseline-lr3e4.json" --evaluation "$r/evaluation-lr3e4" --output "$r/review-v2"
& .venv/Scripts/python.exe -m pytest -q
```

这些是已执行命令；现有run、诊断及冻结评价默认拒绝覆盖，locked_test已完成也拒绝重复运行。新实验必须另设输出目录。取消用 `New-Item -ItemType File "$r/formal-lr3e4/CANCEL"`，等待cancelled后移除此标记，再用同配置及 `--resume "$r/formal-lr3e4/last.pt"`；本轮正式训练未取消，恢复实测是隔离CUDA fixture。完成30轮的当前配置不会因resume增加轮数，不能改配置后冒充相同断点。

本轮开始preflight复核使用第3节同样参数，输出新实验 `preflight-recheck.json`，结束输出 `final-input-audit.json`，没有覆盖准备批次。原报告中training_started=false保留其训练前历史含义。本地试听只监听127.0.0.1:8773，当前服务PID在 `review-v2-server.pid`；关闭后HTML和WAV仍可在本机打开。

## 5. 必须保留的边界

- 用户规则：幽灵不学习，音符归轻击；突袭／倒置突袭归轻击；普通连打起点双重击，真实长条另保留；Boss持续连打双轨长条；Boss单次近战为排他特重击；道具暂排除。特重击同刻双轨排他、长条半开占用是版本化工程默认值，没有擅加前后留白。
- 社区谱没有逐note“来自人声还是伴奏”的真值。软人声偏好未经过训练或听评验证，可在开发／验证集合消融；不要伪造声部监督。
- 训练时基来自源谱；应用推理时应接当前确认时基／BeatThis结果。自动估计误差和训练到推理的差异还需实曲验证，源谱时基成功不代表自动估拍已达到同精度。
- 346张中135张评级来源不一致，按modern→legacy→BMS保留选择依据；180张没有显式音乐启动标记，记录implicit-zero策略。这些是全候选统计，不能说每张都人工同步过。
- 特重击训练参考仅206个，禁止用总体正确率掩盖它的失败。14个同轨同类量化冲突单元已屏蔽相关loss，原毫秒事件仍保留。
- 自动同族／声学检查不是翻唱、重混的完整人审。社区参考不等于gold；原素材公开许可未确定，保持本地使用。
- 双轨长条导出为两条物理长条，原逻辑分组及来源ID保存在TrainingChart旁表。BMS回读通过不等于真实Boss场景／游戏加载通过。
- 尚未通过：模型制谱质量（长条尾配对、特重击泛化、同拍多类／双轨误报明显失败）、分轨与预测真人听评、自动时基到新音频验证、生产工作台六类完整接线及真实游戏验收。监督训练和自动评价已完成，但不能替代上述验收。

## 6. 2026-09-27 新曲《琵琶行》首次试玩包

用户当时对先前验证片段给出“试听试看感觉还可以”的初步反馈，随后明确要求用 `E:\下载\琵琶行.mp4` 生成实际试玩谱，默认场景、创作者mikee。**最新更正：用户看的是参考谱，已撤回该正面听感；不能再用它支持模型有效。**以下为当时交付记录，当前质量失败与诊断见第7节。

已交付 `E:\下载\琵琶行_mikee_MDAC2试玩.mdm`（3,851,101字节，SHA-256 `6996c6b553749e55152a46e13a89bc6774cd1da6eba8c57b0684dc9622beea25`），独立工作目录 `tools/private/playtests/20260927-pipaxing-v1`。包包含map2.bms、原曲OGG、15秒demo、默认封面及info；scene_01，levelDesigner／LEVELDESIGN=mikee，音乐作者未提供记Unknown；难度输入9.5未人工评级。没有放入游戏目录或自动启动游戏。

复用冻结第27轮checkpoint与全部出音阈值／解码器，仅对用户新音频执行冻结分离器及新曲预测；没有重训、重新划分或打开锁定测试。全曲146.239274秒，10窗中心唯一覆盖，保留444个逻辑物件：轻击314、重击54（25组双重击＋4单轨）、特重击0、Gemini44组、Hold6、Gear26。六类回读、规则和包资源检查通过；OGG样本数相同、前20秒波形相关偏移0样本，游戏加载仍pending。

新增独立入口 `tools/generate_joint_playtest.py`（audio／beats／features／predict／package），不是生产六类工作台接线。BeatThis原始376拍中有21个过近候选，按高相干完整网格进行可审计候选去重，保留355个原始时间点；没有把拍点强制量化。既有固定／两段时基拟合仍未通过35／80ms门槛（P95 52.09ms、最大81.24ms）。本次明确使用实验性的观测拍间插值／边界短外推，未把失败拟合标为确认通过。原始时基及去重明细保留；这仍是新曲自动时基尚需人审的证据。

6个Hold中3个只有1ms（72.724、73.142、73.562秒起），保留冻结输出供试玩审阅，未静默删改。试玩需注意整体同步、极短长条、Gemini／双重击和齿轮行为。2项去重定向测试及既有解码／六类导出共14 passed；报告、实际命令、数据／模型身份、阶段耗时和源码快照见 [试玩说明](../tools/private/playtests/20260927-pipaxing-v1/试玩说明.md)。原视频、既有准备与训练产物不变，未晋升生产模型。

### 试玩反馈更新：失败

用户随后实测反馈“效果非常糟糕”，具体“空档非常多经常好几秒无操作，而且gemini通常两三组叠在一起”。本首当前质量状态改为**用户试玩失败**；第6节上文与package-report中的pending是交付当时历史状态，不是当前结论。

复核确认16段≥2秒无击打／有效Hold的空档，合计57.271秒，最长6.513秒；70个同刻同轨瞬时击打叠层位置，另35对0～20ms近邻叠层；44组Gemini中39组伴随其他瞬时击打。0.705秒就同时有Gemini＋双轨轻击＋双轨重击。旧“Gemini成对”8秒预览的15组Gemini也全部伴随其他击打，原图形覆盖／混音提示不能体现实际可玩性。最终包回读仍等于预测，非出包丢音造成空档。

新增独立playability_audit和出包前拦截，20项定向测试通过；原训练标签、冻结解码／权重／阈值和原包保留。空档与模型分数偏低直接相关；自动时基和泛化贡献尚未分离，不凭新曲反馈任意降低阈值或填满休止。检查已补，候选仲裁／模型覆盖尚未修复。证据见 [问题诊断](../tools/private/playtests/20260927-pipaxing-diagnosis-v1/问题诊断.md) 和同目录audit.json。

### 同位置唯一规则的后续落实

用户随后明确同一位置禁止两个note，只有长按＋点击例外。现已新增生成policy `generated-single-position-v2`：`generated_constraints.py`提供统一占用校验和按分数候选选择，Gemini／显式双重击组均保持完整；Hold＋一个点击允许，Hold＋多个点击、Hold互相重叠、齿轮冲突和原特重击排他违规均拒绝。近重合使用原工程20ms容差，不移动落点。

`generate_joint_playtest.py`现先保存v1配对候选及原audit，再执行新选择，chart明确记录generation_policy；出包通过playability_audit复用同一validator。冻结v1解码及旧训练数据保持原hash，不把新的推理结果包装成原锁定评价结果。该选择只处理v1保留的候选，还未解决漏音或动态阈值。

实际32项定向测试通过。《琵琶行》已缓存prediction.npz静态复核444→330个逻辑物件，同刻叠层70→0，20ms近邻叠层→0；5个带点击的长条保留，原16段长空档仍在，BMS语义回读通过。新结果位于 `tools/private/playtests/20260927-pipaxing-occupancy-v2/report.json`，没有覆盖旧包或导出未经试玩的新包。命令：`.venv/Scripts/python.exe tools/check_generated_occupancy.py --playtest tools/private/playtests/20260927-pipaxing-v1 --frozen tools/private/community/supervised-joint-v1-20260927/evaluation-lr3e4/frozen.json --output tools/private/playtests/20260927-pipaxing-occupancy-v2`。契约详见15节4.1。

### 用户授权另存v2

随后用户明确“不要替换，生成v2”。已另存 `E:\下载\琵琶行_mikee_MDAC2试玩_v2.mdm`，游戏曲名“琵琶行 v2”、默认scene_01、谱师mikee；原包hash不变。v2 SHA-256=`d4000e0fec3b57bf04dbd0a90680a70c8c02e45967a19a1325ede59e07ae614b`，3850294字节，330个逻辑物件／354轨道实体。原曲、demo、封面逐字节复用；BMS回读／占用／包资源检查通过。动态阈值未启用，16段长空档与3个1ms长条仍在，真人试玩pending。记录位于 `tools/private/playtests/20260927-pipaxing-v2/版本说明.md` 和package-report.json；入口tools/export_playtest_revision.py，不覆盖父版本或重跑分轨／训练。

## 7. 最新状态：v2失败，撤回预览正评，先定位问题

用户继续反馈大片空闲、基本都是Gemini；随后明确预览看错，把参考当成预测，要求先找清问题、不急于修改。v2交付时的pending现在更新为**用户试玩失败**；旧静态产物保留历史。8个预览的预测均与旧整曲预测相应切片逐项相同，不存在一套已经良好的短片模型。

已完成只读诊断，详见 [诊断结论](../tools/private/community/joint-baseline-diagnosis-20260927/诊断结论.md)：

- 194张训练／验证标签重新生成11种张量字段，与缓存逐值相同；92,283个普通轻击的轨道和秒位置与来源单元一致。22张验证谱重做前向，与旧缓存所有输出最大差0。未访问锁定测试、未改划分或加入隔离谱。
- 训练参考双轨击打仅8.83%。同口径验证参考双轨8.72%，旧预测91.38%；原始地空轻击分数平均相关0.99719。给定正确轻击落点后的选轨准确率50.58%，多数轨基线50.68%；8张确定性抽取的训练谱也只有51.68%，且91.88%双轨。这是模型排键退化，v2同轨去重不能修复。
- 只评击打、20ms归并、50ms匹配、共同内段的采音F1：原设置60.21%，音频平移8.13秒34.17%，三流静音0，相位旋转四分之一拍43.62%，只移除相位64.24%；无需音频的固定半拍网格62.56%。不同于历史整曲sampling口径，不能混用。模型确实用音频，但声音／时基融合及评价门槛都有问题；不能推论BPM应删除。
- 漏匹配4,672个参考击打中4,509个（96.51%）附近原始击打分数未过冻结阈值，主要在出包前失败。独立sigmoid／加权BCE／独立阈值允许选轨不确定性变成双轨并发，是有依据的机制解释；具体损失、表征、声学同步贡献仍待单项消融和人工参考核查。
- 《琵琶行》v2实际174个击打位置154个双轨，其中真正Gemini只有21组；用户感受到的双键滥用不只来自Gemini类别。v1/v2包hash重新核对不变，没有生成v3。

在用户要求诊断优先之前，曾启动新目录`tools/private/community/supervised-structured-v2-20260927`：沿用一个共享联合模型，增加出音和18种合法瞬时组合头；原best初始化，LR0.0001、batch8、max30、patience7。仅训练角色目标适配，36个歧义帧只对新组合loss屏蔽，不改原张量。100步过拟合通过；正式第9轮早停、最佳第2轮、200.95秒、峰值568,480,768字节。新best SHA-256=`86e2be815722f7bbbdb9624e4fa0ce64b151f6858fa9137a2783af9aa5cd205a`；最佳验证组合准确率38.85%，精确帧出音F1=0.52849。**这是未验收实验，不是可用的新模型。**新`structured_decoding.py`只是未验证草稿，不作为生成入口；后续验证／调参／冻结／出包尚未执行。

实际命令及配置／身份／checkpoint见上述诊断结论§5、新实验config、overfit-final/report.json、formal/status.json和metrics.jsonl。诊断入口为`tools/diagnose_joint_baseline.py`、`tools/diagnose_joint_cached_stages.py`；不含训练或参数选择。所有训练已结束，无自动续跑；训练成功、听感合格、游戏可用分别记：工程完成／旧预测失败、新实验待验／未通过。下一步应先确立人工核对的诊断落点、给定正确落点的组合选择与整段排键检查，再讨论单项训练修正，不能以填空小节掩盖退化。

## 8. 2026-09-28 双模型监督交接（自动工程完成，质量失败）

本次目录 `tools/private/community/supervised-dual-v1-20260928`。实际模型参数采音3,331,074、排键6,505,080，独立初始化／优化器／权重；配置onset.json、arrangement.json。派生数据只生成172训练及22验证，217缓存复用，不访问锁定测试内容。

新入口prepare_dual_training.py、train_dual_supervised.py、check_dual_contracts.py、check_dual_resume.py、evaluate_dual_supervised.py。代码在src/mdac2/data_preparation/dual_*.py。预检绑定既有preflight-recheck；派生报告逐文件记录身份。训练与恢复门槛、损失及完整接口见设计17。

已发现并处理：首次post-norm采音恒定输出，小批未通过；改pre-norm及残差尺度0.1后100步采音F1=0.96606、漏点0/185、误点13、平均误差0.871ms。首次GRU内部dropout在CUDA恢复检查中不逐位相同；改GRU内部dropout=0，使用显式输入dropout0.1并保留历史token dropout0.2，重新执行两模型门槛。旧失败产物保留，最终报告路径及实际Sol身份随启动补记。

### 正式启动证据

最终门槛：overfit-onset-v3（100步；F1=0.966057；loss下降96.00%；峰值253,057,536字节）；overfit-arrangement-v2（400步；组合准确率0.978378；峰值294,680,576字节）；resume-check-v2两个模型逐位恢复通过；contracts-final通过。ready-for-sol.json绑定最终身份。

已真实调用 `collaboration.spawn_agent(task_name=sol_training, model=gpt-6-sol, fork_turns=none)`，canonical代理身份 `/root/sol_training`，未替换其他模型。Sol于北京时间2026-09-28 02:07:54启动采音正式训练PID35508，配置onset.json、目录formal-onset；状态和命令见sol-status.json与Sol监督记录.md。主代理继续审阅结果与试听页。训练尚在进行，不提前记完成。

所有入口的最终源码快照与SHA保存在source-final；新增预览入口build_dual_review.py默认实际预测在前、参考折叠，整曲和24秒片段，待训练验证后生成。CANCEL与恢复使用第17设计契约。

### 排键历史依赖的追加诊断

门槛通过后进一步检查训练第一核心82事件：正确历史teacher-forced为79/82（96.34%），自身token历史为20/82（24.39%），只改为预测长条占用仍79/82；原曲表征置零而提供正确历史仍77/82（93.90%）。整段与逐步teacher-forced最大logit差4.8e-6、实际BMS回读通过、规则拒绝0。说明此过拟合实例主要依赖参考历史token，而非已证明音频驱动排键；20%历史dropout不足以消除暴露偏差。报告overfit-free-running-smoke.json及overfit-history-diagnosis.json，未用验证／测试调参。正式固定基线继续采集完整自由生成结果，不在运行中修改绑定源码。

### 本次完成结果与接续边界

Sol顺序完成采音11轮（best3）／排键24轮（best16）及22验证。最佳采音SHA bf4b4bd0b33fea90b9f9f4606127ea1d0cf0a4bbd2bea31d153953f81e5d8d50；排键SHA 2f654fd4da5ac20462570e05d8c90b2085e87c3c8c997f0b28b72a5cb27d601d。阈值0.3，采音50/20ms F1 76.43%/66.31%；排键teacher-forced55.92%／正确落点自由运行24.38%，全链六类宏F1 6.29%、长条联合1.47%，特重0/21、Gemini0。不能把前者用作实际生成质量。

详细配置、全部实际命令、三组分项指标、源码及数据身份、故障与消融见[双模型训练实验报告](../tools/private/community/supervised-dual-v1-20260928/训练实验报告.md)。acceptance.json区分执行成功、质量失败、听感pending、游戏未验收。依旧没有新的独立锁定测试。模型数量、输入和绝对时间契约见17。

review-v1/预测听评.html有整曲＋24秒诊断片段，36播放器，参考默认折叠；pipaxing-review-v1/琵琶行-实际预测.html全曲750个物件全部轻击、地385／空365，无叠层／≥2秒中段空档但排键仍失败。只生成诊断JSON/BMS及音频，未出新MDM。两页在127.0.0.1:8774，可从review-server.json找到服务PID；服务支持Range，播放和跳转技术核验已通过，不能等同真人听评。

工程结束审计final-audit.json确认原manifest及旧v1/v2包hash不变、36音频hash/时长通过。后续不得把已经早停的配置改参数后继续同一断点；新的历史自条件训练、声部偏好或尾点方案另立版本，先增加自由生成门槛再正式训练。当前没有训练/评价进程，也没有重复自动化。


## 9. 当前接续：排键 v2 修复

用户已授权“修”。仅重训排键，复用双模型v1采音best与阈值0.3。新目录supervised-dual-v2-20260928，ready-for-sol.json包含真实入口命令；设计17末节为当前排键契约。先核验overfit、contracts和resume-check均passed，再启动formal-arrangement；完成后用evaluate_arrangement_context_v2.py生成22验证，不开锁定测试。每轮整曲无参考历史评分选best；不得恢复v1或改其配置。最大35轮／patience8，CANCEL/last.pt恢复边界保留。实际结果追加记录，单类锁死、轨道失衡或六类失败仍需如实标失败；不自行出生产包。


### v2实际完成与当前接续边界

best20/28轮，SHA 53e971ed17e6a579e20e0400610d8ad6461caf98ed9f17c38a78fe40e48fee32。Sol运行训练及评价后回复触发额度限制，主代理完成校准、对照、浏览器验收和审计。当前审阅采用calibration-v1/alpha-0.5：采音不变，类别分数减0.5×log训练权重；有限候选仅验证选择，无测试。原首曲地空365/1→192/200，最长355连→9；整链宏F1 13.16%、长条联合1.68%、齿轮0命中，**仍不是可用生产模型**。新review-v2与pipaxing-review-v2位于8775，旧8774保留。源码/权重/数据/命令与所有限制见新实验修复记录.md；无剩余训练进程，无自动重复训练。后续需处理类别及长条泛化，不能因轨道数量均衡宣称验收。


当前用户追加：齿轮移交后处理，不再作为后续排键学习和主要验收目标；旧权重/六类报告不改。均衡不能只看22曲合计或±25个百分点宽门槛。参考21/22谱地面45%～55%，预测仅10/22；琵琶行排除Gemini单轨点击227地/396空。下轮版本需按17末节更新派生标签/词表和逐曲、单轨、局部均衡评价，不在现有冻结实验中热改。


### 2026-09-28 排键 v3 可执行补充

执行入口 `tools/train_arrangement_context_v3.py`；配置 `tools/private/community/supervised-dual-v3-20260928/arrangement.json`。独立派生数据、权重和源码；齿轮移除监督，保留同位置非齿轮物件。41种合法组合、6,873,684参数，原曲TCN+双向GRU结构不变；采音冻结。逐窗口对参考的地/空、单轨地/空、双轨分布施加均方误差，不硬凑1:1。验证选模改用五类分项与逐曲三种比例误差，避免用Gemini抵消单轨偏置。局部10秒计数、连续单轨长度同时输出。

注意：新版派生仅排键标签，旧采音的齿轮来源落点仍存在；不可宣称整个采音训练已去齿轮。后处理齿轮默认关闭。过拟合/合同/取消恢复检查均通过（详见该实验三个检查报告和overfit/report.json）；正式训练及真实预览效果仍待验收。无人工评审则待听评，无游戏测试则待实机。


### 正式启动与监督
2026-09-28 21:57:46 +08:00，由主代理实际启动正式训练，PID41672，命令见supervision.json，输出formal-stdout.log/formal-stderr.log。此前GPT-6 Sol因额度限制未能启动，已向用户说明；超过恢复时间重试后Sol于21:59:09实际接管，只是接管监督，不改写启动身份。见Sol监督记录.md/sol-status.json。原mdac2-sol同线程自动监督已更新为15分钟，避免断连后无人检查，禁止重复训练。

已备好同口径对比 tools/compare_arrangement_context_v3.py：同时过滤v2/v3及参考的齿轮，报告五类、逐曲/单轨地空、双轨和最长单轨串。加权CE后验校正固定alpha候选[0,0.5,1]，只按22验证选择（tools/calibrate_arrangement_context_v3.py）；这是近似校正，不针对琵琶行调参。原始结果保留，alpha=0必须严格复现原始TP/FP/FN。独立新版预览工具 build_arrangement_context_review_v3.py / predict_arrangement_context_cached_song_v3.py 已准备，训练完成再生成，不提前宣称改善。


### 验证候选：受限整曲轨道软先验（尚未选定）
新增独立 `arrangement_lane_prior_v3.py` 和 `tools/calibrate_arrangement_lane_prior_v3.py`，不修改已经绑定训练身份的v3核心文件。先验证选择加权CE校正alpha，再固定其值，从轨道先验强度[0,0.5,1]按相同22验证评分选择。仅无长条的单轨轻击/重击允许使用同类型反轨候选，最大logit增益受强度限制；平均置信度损失 + 4×整曲地空/单轨占比偏离45%–55%区间的平方作为软目标，不强制达标、不约束局部交替。Gemini、双重击、特重击和长条不因该先验改变。若合法性回退引起类型/长条/双轨数量变化，整曲退回未加该先验的解码结果。零强度完全保留原结果；合成检查已验证不补点、类型/双轨保留及高置信预测保持。

候选必须同时满足五类macro/micro F1相比强度0下降不超过0.01、正确落点组合准确率下降不超过0.02，否则不参与选择。这只是排键结构后处理，不表示模型听懂音乐，也不意味着听感或实机验收。不能根据琵琶行挑强度。候选结果、原始模型与未加先验结果均保留；新包不晋升生产。


## LOG-20260930-001 v3完成，客观排键验收失败

训练正常完成28轮/6160步，best22，956.09秒，峰值显存765382144字节。best.pt SHA256 c38878a365e351427d771d488e47b423dcc7389aa7a3be983ddfa81d1ed6e2f0。主代理启动、GPT-6 Sol短暂接管后额度受限至2026-10-04 10:29；主代理9月30日完成剩余评估，不冒称Sol完成全部监督。旧监督状态保留.before-final，现已更正。

实际运行（统一GPU解释器tools/.deps/pretraining-py313/Scripts/python.exe；工作目录E:/project/MDAC2）：
1. tools/evaluate_arrangement_context_v3.py --experiment tools/private/community/supervised-dual-v3-20260928 --output tools/private/community/supervised-dual-v3-20260928/evaluation
2. tools/calibrate_arrangement_context_v3.py --experiment tools/private/community/supervised-dual-v3-20260928 --output tools/private/community/supervised-dual-v3-20260928/calibration-v1
3. tools/calibrate_arrangement_lane_prior_v3.py --experiment tools/private/community/supervised-dual-v3-20260928 --output tools/private/community/supervised-dual-v3-20260928/lane-prior-v1
4. tools/compare_arrangement_context_v3.py --experiment tools/private/community/supervised-dual-v3-20260928 --old-evaluation tools/private/community/supervised-dual-v2-20260928/calibration-v1/alpha-0.5 --output tools/private/community/supervised-dual-v3-20260928/comparison.json
5. tools/build_arrangement_context_review_v3.py --experiment tools/private/community/supervised-dual-v3-20260928 --evaluation tools/private/community/supervised-dual-v3-20260928/lane-prior-v1/strength-0.5 --output tools/private/community/supervised-dual-v3-20260928/review-v3
6. tools/predict_arrangement_context_cached_song_v3.py --experiment tools/private/community/supervised-dual-v3-20260928 --evaluation tools/private/community/supervised-dual-v3-20260928/lane-prior-v1/strength-0.5 --cached tools/private/playtests/20260927-pipaxing-v1 --output tools/private/community/supervised-dual-v3-20260928/pipaxing-review-v3

验证选择alpha=0（不校正）、软先验强度0.5。完整22验证同去齿轮口径：五类macro F1 v2 15.7911%→v3 14.6905%，micro 25.1881%→24.0806%；轻击29.7330%、重击5.3073%、特重17.6471%、Gemini15.4736%、长条头类别5.2914%，长条头尾联合1.0748%（v2 1.6821%）。均衡45%–55%曲数10→13，单轨9→11；逐曲综合比例误差0.06841→0.06926，平均最长同轨点击串27.36→19.82。部分结构改善不能抵消分类、长条和综合比例退步。

琵琶行同750落点：487轻击、252Gemini、6重击、4长条、1特重；地557/空445，单轨302/192（61.13%地）；双轨33.78%（v2 16.93%）。该曲先验未实际换轨（chosen_bias=0），因此Gemini倍增来自当前模型/验证选择，不能归咎软换轨把单击转成Gemini。训练不含齿轮，但旧采音仍含其来源点。

结论：训练执行成功，排键客观质量失败；保留v3失败对照，不替换v2（v2本身也未通过生产验收），不新出MDM、不晋升生产、不开放锁定测试。39音频hash/时长检查与HTTP206通过，BMS回读生成时通过；不等于听感或游戏验收。预览127.0.0.1:8776/review-v3/预测听评.html、pipaxing-review-v3/琵琶行-实际预测.html已加红色失败提示。无人反馈，待听评/未实机。源码和权重身份核验匹配，manifest未变，源素材/旧包未修改。

当前证据支持：仅加分布损失不足以解决泛化；验证选模的平衡收益仍可能伴随分类退步。下一次架构/损失修改必须另建版本，以单轨选择与Gemini/双重击区分、音频依赖消融及长条尾预测为独立门槛，不能直接在冻结v3断点上热改或宣称仅调配比已修好。本轮有界实验、评价、预览、日志已完成，停止重复定时监督，保留失败证据供后续诊断。


## LOG-20260930-002 排键v4：组合与轨道分开判断

用户授权继续改进。v3诊断：训练Gemini频率6.2689%，权重1.4917，使加权贡献9.3514%；单轨轻击地/空频率36.03%/34.12%，权重仅0.622/0.639。固定8验证窗v3原音组合准确39.04%，移位2.37秒39.95%；五类macro原音16.53%、移位17.16%。静音准确0，说明需要非零音频，但移位不降分，不能声称局部音乐内容已正确用于排键。另有结构性假设：平面41组合softmax分裂地/空单轨概率，而对称Gemini不分裂，轨道不确定可能倾向对称类；这是机制假设，未单独因果验证。

v4独立目录supervised-dual-v4-20260930，复用v3去齿轮派生与原缓存/172训练22验证，锁定测试不开放，采音v1冻结0.3。模型保留原曲TCN+2层双向GRU，实际6,893,154参数。41组合按地空镜像合并成24组，先预测组，再用每组独立2向head决定方向；对称组无方向损失。组分数+方向logit减组内最大值用于合法解码，最佳方向不会降低组证据，不能因为方向不确定把单轨类挤成Gemini。归一化ranking score不是校准概率。Gemini与双重击是不同组，hold+click仍可表达。无BPM/格点/补点/标签历史输入/自动齿轮。

损失=无加权组合组CE + 非对称组方向CE + 0.25局部音频组CE + 0.5长条log1p时长SmoothL1；不使用旧类别权重或分布损失。本版组合分组与取消加权同时变化，不能把效果归因于单一因素。完整验证选模公式暂沿用v3，原始无轨道后处理版本直接评价；旧软换轨先验不使用。不靠Gemini数下降单独判断成功，必须看五类与长条召回、单轨质量，避免退化为全轻击。

已用GPU解释器运行check_arrangement_context_contracts_v4.py、check_arrangement_hierarchy_v4.py、check_arrangement_context_resume_v4.py（报告在新实验）。41组合可达、方向不确定不抬高Gemini、跨窗长条/双重区别、padding/窗口、CUDA取消恢复逐位一致均通过。train_arrangement_context_v4.py --config tools/private/community/supervised-dual-v4-20260930/arrangement.json --run-dir tools/private/community/supervised-dual-v4-20260930/overfit --overfit --steps 1000：150步178训练事件97.19%组合/规则后准确，静音0.56%，loss下降91.09%，峰值349483520字节。仅为门槛，不代表泛化。正式随机种子重启，28轮/早停7，LR0.0002，batch8，保留best/last/CANCEL恢复。Sol仍受限到10月4日，主代理监督，不能冒称Sol执行。


### v4b 单因素对照：恢复组合组加权

v4中途验证显示少数类型召回很低，另建v4b（不修改v4）。相同分层模型6,893,154参数、初始化seed/学习率/数据/有界轮数，仅将组合组CE与局部组合组CE恢复训练集平方根频率权重cap4并按频数均值归一；方向CE保持无权重，仍无分布损失。权重基于24个合并组合组而非41个方向组合，详情group-weights.json。目的是隔离取消加权对少数物件的影响，两版保留同一验证选模。均不得仅以Gemini减少判成功。

v4b训练集150步178事件组合100%，loss下降93.11%，静音/规则检查见overfit/report.json；CUDA取消恢复逐位一致passed。继承同一v4模型及分层/窗口/合法性检查，v4b仅训练损失不同。主代理监督，Sol额度限制不冒称其执行；所有操作E:/project/MDAC2，复用v3派生/采音v1/阈值0.3，原始素材、缓存与测试不动。命令tools/train_arrangement_context_v4b.py --config tools/private/community/supervised-dual-v4b-20260930/arrangement.json --run-dir tools/private/community/supervised-dual-v4b-20260930/formal-arrangement；GPU解释器固定，28轮/早停7。完成后evaluate_arrangement_context_v4b.py，比较与分层诊断工具v4可通用；预测工具predict_arrangement_context_cached_song_v4b.py。原始输出不加软换轨，不套v3类别后验校准。


## LOG-20260930-003 v4/v4b有界对照完成，部分指标改善但未修好

两版均28轮/6160步，best25，独立评估/分层诊断/音频消融/预览全完成。主代理执行，Sol因额度限制未参与。v4 SHA 286f9e61f1de7b7498f1b28428f5547edb06d0eb42e56e991cfba1c8cb26bc84；v4b SHA 5f11f9015d38739fbe9f4b83e28c779f37053b86b01851eb9da1d6fb505db6bf。原manifest不变，源码checkpoint身份一致，未开测试、未重分轨。完整命令及分项在各实验结果与接续.md，自动流水线completion-status.json有实际子进程命令；最终源码快照source-final。

同去齿轮口径（v2→v4→v4b）：五类宏F1 15.79%→10.67%→16.69%；micro 25.19%→26.42%→23.38%；长条头尾联合1.68%→0.74%→2.72%。地空45%–55%曲数10→9→9，单轨9→6→6；逐曲综合比例误差0.06841→0.07547→0.09626。v4b宏/长条局部改善但micro、均衡退步，不是整体修好。v4b五类F1轻29.08%、重7.76%、特重24.24%、Gemini13.90%、长条头8.44%；Gemini预测实体3446、参考1246，仍过量。

正确参考落点原始组合组准确率v4 70.88%、v4b65.47%，但给定正确类型的轨道方向准确率44.91%/46.83%，完整41组合准确36.46%/34.46%。前者高不能冒称完整排键成功。方向近随机、类型加权有明显多类权衡，是本轮确认的问题；不能断言根因全部确定。静音时多数类准确率可以升高，必须结合macro：v4原音/移位macro13.50%/10.67%，v4b14.78%/12.38%，只是固定8验证窗，不能证明已听懂音乐。

琵琶行同一采音落点：v4轻742/Gemini6/长条1/特重1/重0，退化多数轻击；v4b轻437/Gemini251/长条33/重29/特重1（长按+点击合法共点使逻辑物件751），地474空528，总比例47.31%地看似均衡，但单轨地198空269=42.40%地，Gemini仍过多。v4b作为分项略改善的诊断预览提供，不推荐替换生产或旧包。

预览v4=127.0.0.1:8777，v4b=127.0.0.1:8778，各自review-v4[b]/预测听评.html及pipaxing-review-v4[b]/琵琶行-实际预测.html。两版均加失败/未验收标注，78份媒体hash/时长检查、BMS回读和HTTP206通过。无人听评则pending，未实机；无新MDM，无自动晋升。训练/pipeline进程均结束，暂停本轮重复监督自动化。

后续应先验证相邻物件同轨/换轨关系及合法生成历史怎样进入模型，而非继续微调静态地空配额；需防参考历史泄漏，新增训练集自由生成门槛与方向/组合独立指标。分层+权重对照证明两者不能单独解决方向问题；不在冻结v4/v4b上热改。任何下一轮另立版本与有限验收，不把重复验证成绩当新独立测试。


## LOG-20260930-004 v5受限解码修复与同轨/换轨探针

用户授权继续修复。本轮保留v4b主体权重，实验目录tools/private/community/arrangement-repair-v5-20260930；v5是解码版本，不冒称重新训练一个更好的6.9M主体模型。先验证组合组权重的完整校正alpha[0,.5,1]，因长条联合F1从2.724%降到1.810%/1.172%而拒绝，两候选保留group-calibration。

采用另一个受限候选：只对基线已合法解码为Gemini且与最强单轨轻/重击差值小于margin的位置选择单击。margin候选[0,.25,.5,1]只按22验证选择，最终0.5。没有固定数量目标、配额、随机换轨、网格、BPM、补点或删除落点。任何改动导致其他位置或长条/特重改变时整曲退回基线。零margin严格复现原始TP/FP/FN；合成检查强Gemini、双重击、长条、特重均保留。对全部22验证和琵琶行逐事件再次核验，仅指定Gemini改为一个普通点击，其他事件及所有落点完全保留。

验证同口径v4b→v5：五类宏F1 16.6851%→16.9955%，micro23.3802%→24.1836%；轻击29.08%→29.49%，重击7.76%→11.46%，特重24.24%保持、长条头8.44%保持、头尾联合2.724%保持；Gemini自身F1 13.90%→11.34%，说明减少误发伴随更多漏掉真实Gemini，不能只报数量下降。双轨12.19%→5.89%，逐曲综合比例误差0.09626→0.09822，没有解决方向/逐曲均衡。

琵琶行未用于选margin：原251Gemini改为167，84个位置换成单击；新轻470/重80/长条33/特重1，总逻辑物件751，采音落点仍750。长条及特重逐事件完全一致。单轨地245空306（44.46%地），总地437空481，最长单轨18；不因总比例好看宣布修好。没有新MDM或生产晋升。试听页review-v5/预测听评.html、pipaxing-review-v5/琵琶行-实际预测.html，39音频hash/时长及BMS回读检查通过，待人工听评/未实机。

同轨/换轨诊断探针另存transition-probe，非生产排键器。只用原曲±80ms局部均值/标准差、前后频谱和相邻落点间隔，标签为相邻有效单轨轻/重事件的same/switch，不输入参考轨道历史。训练128对100步过拟合门槛100%通过，随后训练intervals-only与audio+intervals小MLP，各最多30轮/早停5，从训练集归一化；角色172训练22验证不变，未开测试。9091验证对：仅间隔logloss0.62039、Brier0.21759、准确61.57%、switchF1 53.55%；加音频logloss0.63842、Brier0.22417、准确62.62%、switchF1 50.00%。准确率略升但概率质量与换轨F1下降，不足以支持接入出谱；探针未部署。不将其小参数量当作新正式模型。

实际命令统一GPU解释器tools/.deps/pretraining-py313/Scripts/python.exe，工作目录E:/project/MDAC2：
- tools/calibrate_arrangement_groups_v5.py --experiment tools/private/community/supervised-dual-v4b-20260930 --output tools/private/community/arrangement-repair-v5-20260930/group-calibration
- tools/calibrate_arrangement_gemini_margin_v5.py --experiment tools/private/community/supervised-dual-v4b-20260930 --output tools/private/community/arrangement-repair-v5-20260930/gemini-calibration
- tools/probe_arrangement_transitions_v5.py --config tools/private/community/supervised-dual-v4b-20260930/arrangement.json --output tools/private/community/arrangement-repair-v5-20260930/transition-probe
- tools/check_arrangement_gemini_margin_v5.py tools/private/community/arrangement-repair-v5-20260930/decoder-checks.json
- tools/predict_arrangement_cached_song_v5.py --experiment tools/private/community/arrangement-repair-v5-20260930 --evaluation tools/private/community/arrangement-repair-v5-20260930/gemini-calibration/alpha-0.5 --cached tools/private/playtests/20260927-pipaxing-v1 --output tools/private/community/arrangement-repair-v5-20260930/pipaxing-review-v5
- tools/build_arrangement_review_v5.py --experiment tools/private/community/arrangement-repair-v5-20260930 --evaluation tools/private/community/arrangement-repair-v5-20260930/gemini-calibration/alpha-0.5 --output tools/private/community/arrangement-repair-v5-20260930/review-v5

本轮由主代理执行，Sol仍额度受限，不冒称其参与。主体checkpoint为5f11f9015d38739fbe9f4b83e28c779f37053b86b01851eb9da1d6fb505db6bf，源码身份匹配。数据划分、217三流缓存、固定分离器和准备批次原样保留，manifest同34add12…eb71；采音v1阈值0.3未改，齿轮后处理仍关闭。当前有限修复有验证收益但整体排键仍未通过，真实Gemini召回/可靠方向和长条尾仍是问题；22验证反复选参不能当独立最终评价。下一步若引入序列关系，必须先证明优于时间间隔基线并检验完整生成，不直接上线探针或机械交替。


## LOG-20260930-005 排键v6自主生成上下文改造与正式启动

用户确认需要学习当前音频下的已出键接续并授权增大参数、开始迭代。实际模型10,993,122参数，详见tools/private/community/supervised-dual-v6-20260930/实施记录.md。原曲TCN256+双向GRU512音频上下文，64维自己生成组合embedding、前后时间间隔、预测长条占用和256维单向GRUCell生成历史联合预测组合/相对方向/长条。参考前一个键从不作为输入，teacher forcing=0，离散合法选择反馈stop-gradient，隐藏状态窗口内反传。参考当前键仅用于loss。

训练仍按32秒窗口中央16秒loss，前8秒自主历史暖机；整曲推理先中央拼接音频上下文再持续历史，长条跨窗保留。训练窗口BOS重置与整曲持续历史存在长度分布差异，是明确局限；本轮用整曲自由生成选模及移除历史消融检查，不只看训练loss。参数和历史结构同时改动，不能单一归因。

训练集小批200步178事件：组合99.4382%、loss下降99.0823%，五类召回门槛通过，静音对照满足下降门槛，过拟合权重不用于正式训练。历史层有效梯度、固定音频上下文下因果性、移除历史会改变输出、跨窗长条状态/长按加点击/特重排他/Gemini双重区别检查通过；实际训练整曲反馈和最终合法选择一致检查通过。CUDA安全取消第2步后恢复到第4步，模型/优化器/调度器逐位一致，身份更改拒绝。报告在overfit/report.json、contracts.json、full-song-feedback-check.json、resume-check.json。

主代理root实际启动PID43160，非Sol（其额度仍受限至2026-10-04 10:29）。命令GPU解释器tools/.deps/pretraining-py313/Scripts/python.exe tools/train_arrangement_context_v6.py --config tools/private/community/supervised-dual-v6-20260930/arrangement.json --run-dir tools/private/community/supervised-dual-v6-20260930/formal-arrangement；20轮/早停6，batch4，seed20260930，LR0.0002，best/last/CANCEL身份恢复。实际开始时间及完整命令见supervision.json。不要重复启动，不修改绑定模型/训练源码。

有限completion pipeline主代理启动PID7912：tools/complete_arrangement_history_v6.py；等待训练上限3小时，各阶段30分钟。正式完成后评价reference/predicted落点、整曲历史移除、同口径v4b/v5对照、same/switch关系、分组方向、原音/静音/移位/零间隔/无历史消融，再生成review-v6与pipaxing-review-v6，审计身份/39媒体/BMS。pipeline-process.json、completion-status.json实际命令为准，不与它重复评价。流水线已运行的层级诊断入口是兼容只读v4工具，v6专用工具补充说明ranking包含合法反馈选择的小幅boost；不是校准概率。

源身份快照source/source-sha256.json，数据manifest仍34add12…eb71；172训练22验证25已用测试127隔离不变，217三流缓存固定分离器复用，v3去齿轮派生复用。采音v1/.3冻结，旧采音齿轮来源落点局限仍在；gear后处理关闭。无BPM、网格、补点、均衡配额、软换轨；不动v1-v5原模型/旧包/准备批次，不开放测试、不生产晋升。

本轮训练及评价尚在执行，不能把以上训练集门槛当泛化结果。同一mdac2-sol监督安排已更新为v6，每15分钟仅重要变化通知，不创建新自动化；全部训练、评价、审阅、媒体预览和结果记录完成后再暂停。没有人工听评则待听评，未实机，训练执行/客观排键/听感/游戏分别报告。


## LOG-20260930-006 v6完成：结构改善，实际链路质量退步

主代理root完成监督，Sol额度限制未参与。20轮/8800步，best18，6636.95秒，CUDA峰值570604032字节。best.pt SHA426c1e6a358050a8bcf716638b553a3c1dbe2b42761fbd0c48ccf364a31853d5；checkpoint配置/源码/数据身份再次核验一致，manifest未变，不开锁定测试。训练及completion流水线均实际完成，无重复启动、无训练故障。完整实际子进程命令见completion-status.json；训练入口命令supervision.json，最终源码source-final。review工具继承的v4标题在生成前更正为v6，review-label-fix.json记录，未改绑定模型/损失。

正确参考落点：五类macro22.388%、micro35.375%、组合34.678%、长条头尾联合5.572%。但实际采音落点：macro12.879%、micro24.495%、长条联合1.969%。同去齿轮口径v4b/v5/v6：macro16.685%/16.996%/12.879%，micro23.380%/24.184%/24.495%，长条联合2.724%/2.724%/1.969%。多数轻击micro略升不能抵消少数类别及长条退步。实际采音链路五类F1：轻29.932%、重7.698%、特重6.061%、Gemini13.054%、长条头7.649%、尾4.392%；采音precision71.619%、recall80.805%、F175.935%，11000TP/4359FP/2613FN。旧采音/.3冻结，去齿轮口径改变不代表采音重新训练。

整曲移除生成历史：参考落点macro21.123%（低于22.388%），实际落点macro14.679%（高于12.879%），micro23.749%，长条联合1.745%。历史改变结构，但对实际采音错误条件下的宏指标有害，不声称已学会符合音频的可靠接续。

逐曲地面45%～55%：v4b9/22→v6 20/22，单轨6/22→16/22，平均最长同轨串25.77→7.64，综合参考比例误差0.09626→0.03617；没有历史时0/22、最长88.86。这是结构性改善而非音符类型/音乐质量验收。9091相邻参考单轨对：真实换轨38.873%；v6预测可比单轨对换轨82.884%，switchTP2299/FP3977、precision36.632%、recall65.054%、F146.871%（v4b30.130%）。高召回伴随过多换轨，关系正确也可能两个绝对轨都错。组类型准确67.467%，正确类型条件方向49.383%，不是整链成功率。

固定8验证窗音频消融：组合准确原音39.041%、移位39.498%、静音42.009%、无历史39.954%；五类macro原音15.666%、移位12.216%、静音9.433%、无历史17.205%。音频移位降低类型宏指标，但未降低组合准确，静音的多数类准确率陷阱仍在；不能由这组小样本证明方向听懂音乐。

琵琶行同750采音落点：轻635、重32、Gemini74、长条16、特重1，758逻辑物件（长按加点击合法共点）；地430/空402，单轨346/322=51.796%地，最长同轨11。Gemini数量较v4b251/v5167下降，但长条由33减至16，不能只按Gemini减少判断改善。没有新增落点/BPM/grid/补空/配额/软换轨；本曲不选参。

39媒体SHA/时长、BMS回读、规则及checkpoint/manifest审计通过，media-audit.json；实际预览review-v6/预测听评.html及pipaxing-review-v6/琵琶行-实际预测.html加明确质量未通过提示。acceptance.json已由主代理审阅从pending更正为failed，context-findings.json详细区分事实与未验证机制假设。仍待听评、未实机、不生成新生产包，不自动替换v4b/v5。

未解决：参考当前方向与自己先前方向偏离后的相对关系监督一致性、训练窗口BOS重置与整曲历史长度差异、采音误/漏点引起上下文分布变化。它们是下轮需单独验证的假设，不是已定位因果结论。增加参数并接入真实生成历史已完成有限实验，但整体质量失败；后续需独立版本、验证历史关系标签与采音噪声鲁棒性，不机械强制交替，不在冻结v6上热改。22验证反复选参不是独立测试。


收尾：2026-09-30T15:58:25.482425+08:00，8780独立预览启动，review-v6/预测听评.html及pipaxing-review-v6/琵琶行-实际预测.html的HTTP200、音频Range206通过（http-check.json）。主代理审阅及完整记录完成，原mdac2-sol监督安排已实际PAUSED，不创建新自动化。最终acceptance为训练执行通过、整体排键质量失败、待听评/未实机；保留旧预览/模型/包。


## LOG-20261002-001 按用户要求交付琵琶行v6独立试玩包

用户请求“做个试玩包”，使用当前8780页面v6实际预测，默认scene_01、mikee、难度标签沿用9.5。输出tools/private/playtests/20261002-pipaxing-v6及E:/下载/琵琶行_mikee_MDAC2试玩_v6.mdm，SHA25d0b9b9eee26cbd34677cc98243c1ba2624da94b377d14f667a4ac4f4d387d4，3852070字节。758逻辑/832轨道实体，轻635重32Gemini74长条16特重1，与当前预测逐事件回读一致。

入口.venv/Scripts/python.exe tools/package_predicted_playtest.py --prediction tools/private/community/supervised-dual-v6-20260930/pipaxing-review-v6 --parent tools/private/playtests/20260927-pipaxing-v1 --output tools/private/playtests/20261002-pipaxing-v6 --delivery E:/下载/琵琶行_mikee_MDAC2试玩_v6.mdm --title “琵琶行 MDAC2试玩 v6”；实际参数列表、源码SHA与源音频/checkpoint身份见package-report.json。复用旧包经SHA验证的music.ogg/demo.ogg/cover.png，音频字节不变，继承已核验零采样长度差/零偏移。新包ZIP/资源哈希/BMS语义时间轨道/占用/交付副本校验通过，旧包及源素材未修改。

仅实验试玩导出，尚未实际游戏加载/试玩，不改变v6客观质量失败、待听评/未实机结论，不自动晋升生产模型；原监督仍暂停，不启动新训练。


## LOG-20261016-001 应用v0.1.0与v1模型基线登记

用户实机反馈当前v6“游玩体验竟然还不错”，明确要求将当前应用定位v0.1.0，在github.com/zakee039/MDAC2打tag，当前模型定为v1。该授权覆盖版本登记与tag发布。应用Python/前端原已为0.1.0，本轮统一说明并固定model release v1 = onset历史v1 best + arrangement历史v6 best18，阈值0.3。models/registry.json选择v1，models/v1/manifest.json记录完整SHA/结构/反馈范围与历史客观分项。模型发行号与实验号分开，不改旧实验身份。

本地models/v1/weights/onset.pt与arrangement.pt为实际试玩原始checkpoint的逐字节副本，SHA分别bf4b4bd0…8d50、426c1e6a…53d5；由.gitignore排除，不向Git上传音频、社区谱、准备批次、缓存或权重。原始材料/实验/旧包不改。用户正向试玩反馈单独登记，不改写v6客观验证失败，不夸大成所有歌曲或自动化游戏验收。应用基线用户已选择，仍保留后续改进事项。

当前工程原没有.git，远端为空私有仓库且账户具备写权限；主代理初始化本地main，GitHub标准Git HTTPS连接失败而API可用，后续提交/tag结果另行追加。

版本发布前工程回归：tools/release_check.py --output tools/private/releases/v0.1.0-20261016/engineering，后端549项、前端测试及构建均通过，源码输入未变化。已核验待提交397文件约2.16MB，私有素材/缓存/权重均排除。新增.gitattributes保留原始字节，避免Git换行转换破坏已绑定训练源码SHA。远端空仓库通过API初始化README并恢复相同本地bootstrap历史；后续版本提交将以其为父提交，保持正常快进。


## LOG-20261016-002 v1冻结模型的独立编排r1.1试玩修订

用户确认当前版本可玩、采音满意，要求减少连续交叉、加入同轨三连/四连、长条尾补齐、相似段自动镜像最后执行、高潮加密；另明确三连/四连是同轨连续点击。本轮不重训，不修改v1权重、训练绑定源码、v0.1.0 tag、原数据与旧包。新契约见设计18，实验tools/private/community/playability-postprocess-r1-20261016；最终策略playtest-postprocess-r1.1。CLI已接通，桌面默认生成流程未替换。

只用172训练谱拟合64,033对关系的时距/同轨长度先验；动态规划减少长交替，保留已有三连/四连，不固定模板/1:1配额。复用v1缓存后依次高强度声学候选加密→普通单轨编排→长条尾延长最多250ms→匹配相似段镜像→只读规则检查。无BPM/grid/空档硬填/齿轮，Gemini和双重击及长按加点击保留。相似段按绝对秒多尺度检测、同类型一对一匹配，跨段长条等不安全建议拒绝；是明确授权的试玩自动策略，不伪造人工确认结构组。

22验证按同去齿轮/掩码口径：五类macro12.8787%→12.8813%，micro24.4955%→24.4159%，采音75.9354%→75.8839%，长条头尾联合1.9690%不变。轻击29.7953%、重击7.8475%、特重6.0606%、Gemini13.0545%、长条头7.6486%；尾4.3923%。分类总体接近且有小幅退步，不能把结构改善说成采音/分类进步。受限普通单轨段的换轨率80.5925%→46.8607%，三连271→1693、四连97→306，长交替四步之后的转移计数5559→0；这不是所有物件或独立段落总数，长条补齐使统计覆盖略有变化，且可能过度抑制合理交叉。300个长条尾延长，对齐总数2→302；相似段应用3项。新增45个逻辑轻击，密度规则仅为高强度启发式，未训练语义高潮检测器。未开25旧测试或127隔离，217三流缓存复用。

Snow Song Show：1127逻辑物件保持，三连19→97、四连17→26，换轨75.82%→48.51%，19个长条补齐，相似段镜像2项，新增点击0。琵琶行：758→759逻辑物件，三连19→37、四连12→14，换轨68.98%→52.33%，8个长条补齐，相似段应用0，68.723秒新增1个有声学证据轻击。这两首未用于阈值搜索，规则安全修复后统一回归。

r1.1修复收尾发现的两个边界：已经对齐的长条不再延到下一个点击；镜像保护逐个已对齐长条身份，不能以总数量不降替代。原r1产物/源码快照保留。全量后端559项通过（边界修复前），修复后12项针对性测试通过；22验证、两曲规则/BMS回读、打包ZIP与资源SHA、媒体原样复用、两个预览及6个音频HTTP206通过。final audit核验v1两checkpoint与全部绑定源码SHA保持。未运行新训练、未作用户听感或自动游戏通过结论。

独立试玩包在E:/下载：Snow Song Show_mikee_MDAC2试玩_模型v1_编排r1.1.mdm（SHA eaf5dd9f453710b10d777d5fb2518328c5d24a6b69a6593bd161817bd7d3e27d）；琵琶行_mikee_MDAC2试玩_模型v1_编排r1.1.mdm（SHA 8673aeacc2f9e14624fff1d643203dd79743d329032c2e0c1bbf2cf6c6297df1）。mikee、scene_01、9.5标签沿用，原music/demo/cover逐字节一致。新试听http://127.0.0.1:8781/snow-r1.1/index.html和pipa-r1.1/index.html，实际原版/新版事件对照，无参考谱混淆。服务PID与路径见review-server.json，不重复启动。

最终入口run-prior-r1.1.json、validation-r1.1、snow-r1.1、pipa-r1.1、对应-package-r1.1、review/*-r1.1；中间final/release/r1目录不是最终交付。completion-r1.1.json含实际参数列表、源码/权重/数据身份、测试与验收；source-r1.1保留源码快照。工程通过，客观分类未显著提升，听感/实机待本轮人工反馈，不更换model registry或发布新tag。监督自动化保持暂停，无后台训练。

附故障记录：最终审计第一次因GPU解释器缺tzdata在写时间戳时失败，所有校验已跑过；改用显式UTC+08:00后重新完整审计通过，原有源码快照先校验SHA再复用，不覆盖其他实验。


## LOG-20261017-001 v2冻结与独立人声优先采音 r1

用户报告编排r1.1的《琵琶行》真人感很好，《Snow Song Show》有个别奇怪之处暂缓，要求保存当前为v2、探索检测到人声则优先人声且连续2秒无活动才退回伴奏。本轮实际先于2026-10-16 23:31冻结发行v2，后于10-17完成实验。v2=原v1神经权重+编排r1.1，models/registry.json选择v2；模型两权重及先验本地复制，策略/源码/原包SHA记录models/v2/manifest.json，原v1和v0.1.0 tag不变。两首“模型v2”MDM只是已认可r1.1的逐字节副本，包内历史标题保留。没有把本次人声实验包含在v2或默认桌面流程中。

人声优先r1为独立推理选择层，没有重训。固定分离人声的能量/相对原曲占比连续3个10ms单元确认活动，频谱正变化局部峰提供起音支持，不把长音音量峰当节拍。人声模式从联合模型0.12局部峰选有±40ms人声起音证据的点，近邻优先人声证据；最后活动单元结束后整整2秒才回到原0.3采音，短停顿保持、再次确认活动接管。初始未检测到人声时使用伴奏模式；跨窗不重置。高强度后处理也门控，不能补回被排除的乐器点。冻结排键整曲重新推理，再执行原r1.1编排，相似段镜像仍最后。策略及限制见设计19。

仅复用217三流缓存与既有预测，无重分轨/重新分割/训练。172训练22验证25已用测试127隔离、manifest34add12…eb71不变；社区没有人声来源真值。历史8窗消融已表明只留人声召回6.53%，因此未直接将伴奏输入归零。当前规则一致性不能当作学到了声源归因。

22验证v2→人声r1：采音precision71.4703%→77.7973%，recall80.8786%→71.7109%，F175.8839%→74.6302%，FP4395→2786、FN2603→3851，时间平均误差10.101→10.256ms。五类macro12.8813%→14.8827%，micro24.4159%→24.5041%；轻击29.7953→29.8868%、重击7.8475→5.5909%、特重6.0606→13.3333%（TP仅1→2）、Gemini13.0545→14.2004%、长条头7.6486→11.4023%、尾4.4680→5.0992%、头尾联合1.9690→3.0453%。采音F1和召回下降，不判整体质量优于v2。原始候选15612→12672，移除4279、新增1339。人声模式内自身证据支持61.85%→100%是规则结果，不是独立人声准确率。没有用两首用户歌曲调参。

最初validation误将新解码拒绝审计也用于v2汇总，已仅修正元数据来源并另存validation-final；保留原证据，22曲逐事件、全部分类/采音分数和策略复查完全一致。没有把该修正视为算法收益。完整命令、源码与数据/checkpoint身份、实际分项、24曲门控和BMS检查见实验completion.json。

琵琶行750→596采音落点，v2的759→601逻辑物件，轻489/重13/Gemini76/长条23/特重0；Snow1099→844落点，1127→866物件，轻697/重28/Gemini62/长条79/特重0。采音变化使排键历史和类别变化，并非仅筛掉原谱的几个点。琵琶行实验仍有38.983–41.453、134.164–137.973秒无操作片段，后者附近存在短促分离活动，多次启动等待可能使空档增加，需听评；不能简单补点遮盖。分离残留、弱声/平滑音节、原网络完全无响应、等待期舍弃合理伴奏仍是局限。

569项后端测试通过，24首完整生成的两秒状态、证据门控与规则/BMS回读通过；两MDM ZIP/资源SHA/音频零偏移继承检查通过，2页/10音频HTTP206通过。新预览http://127.0.0.1:8782/pipa/index.html及snow/index.html含v2/实验/原曲/分离人声对照和模式时间条。新包E:/下载/琵琶行_mikee_MDAC2试玩_人声优先实验r1.mdm（SHA bcdd43df9b7d50b33a53870ed2d92bca612f851b0ca6770577aba4cbbc17cf24），Snow Song Show_mikee_MDAC2试玩_人声优先实验r1.mdm（SHA 9f25cd4f8fb7b8bcb1d2c7f8f0705b13e8bf3a7d76eef7403ced002f4fa514e7）；mikee、scene_01、9.5标签不变，音乐资源逐字节继承原包。

实施由root完成，未委派Sol；没有后台训练。原监督保持暂停，不创建新定时任务。工程执行通过；整体采音有取舍且F1退步，人声真实性未经标注验证；本次实验听感与实机待用户反馈，不替换已选v2。完整记录在tools/private/community/vocal-priority-r1-20261016/实施记录.md和completion.json。


## LOG-20261017-002 人声r2与网格对齐需求更新（计划，未实施）

用户反馈r1改善但仍丢字；要求门限更低，采音向后看一秒确认真实人声，从实际起点进入人声模式，当前及未来一秒没有人声就立即恢复普通采音，替代两秒无人声保持。随后要求长条与note（同/异轨）、长条同起/同收/一开一关对齐；最新明确100ms仅作大致候选方向，实际吸附网格，原250ms补尾暂时舍弃。新r2须替换旧补尾步骤，不能叠加；发行v2及r1冻结文件不改。

计划见设计20。当前试玩链路没有真实歌曲节拍网格；six_export的固定BPM120仅为毫秒事件编码，不是节拍估计。已询问网格指歌曲节拍还是固定绝对时间网格，答案前不实施量化，也不默认把10ms分析步长当拍网格。note离格与长条共同锚点策略尚需随网格明确。此轮只核对代码及记录计划，无新模型训练、无r2产物、无新测试验收。


## LOG-20261017-003 固定240/5ms网格、人声一秒前瞻与长条对齐 r2 已完成

用户明确固定BPM240、每拍250ms/50个存储位置、5ms自动吸附，不重新引入歌曲BPM估计。本轮将前述人声与长条要求实现为独立r2，设计20已从待确认计划更新为实际契约。发行v2、旧r1、原始素材/准备批次/缓存/旧包不改；模型注册仍v2，未接管桌面默认生成流程。执行者root，未委派Sol，无新训练，原监督保持暂停。

新增vocal_priority_r2、fixed_grid_r2、six_grid_export_r2与独立推理/对照/打包/预览工具。人声活动按未来一秒覆盖、频谱占用与弱同源残留排除确认，起点立即生效、末次活动后立即退出；去掉2秒保持，片段内部仅连接≤80ms短空隙。已确认区域模型门限.12→.08、起音支持.22→.16、候选间距80→60ms；不确定区恢复v2 .3候选，不能称原曲专用采音模型。高潮候选受同样证据门控，排键继续冻结v6权重整曲自主历史生成。分离特征与规则不是声源真值，频谱约束可能漏判弱人声。

事件先5ms就近落格，原共同时间成组移动；普通偏移≤2.5ms，若取整新增占用冲突可相邻格修复至≤7.5ms，本次24曲均未触发此分支、最大量化偏移2ms。长条与同/异轨note以note锚定，长条头头/尾尾/尾头以平均时间落格，100ms为关联跨度，不允许链式无限合并；保留最短30ms、同轨占用、特重排他、长按加点击与Gemini/双重击区别。旧250ms补尾完全停用于r2，相似段最后镜像且不得改时间。BMS与info均写240，每个1秒小节200格，无变速/歌曲拍位估计。

22验证同去齿轮/掩码整链：v2/r1/r2采音F1=75.8839/74.6302/76.3144%，precision=71.4703/77.7973/71.2864%，recall=80.8786/71.7109/82.1053%；误点4395/2786/4502、漏点2603/3851/2436。r2匹配平均误差10.548ms，地面/天空F1=44.2714/42.0753%；轻30.7738、重7.8001、特重23.5294（TP4/FP9）、Gemini13.6159、长条头9.0767、尾4.7731、联合2.5039%；五类macro16.9592%、micro25.4729%。对r1召回改善但误点增多、长条分项退步；不能宣称所有维度提升。

门限单因素对照在同22验证原始起点/无掩码口径：只替换前瞻保留旧门限F1=76.7183%、recall79.7271%，完整r2降门限F1=76.2001%、recall82.6326%；r2是预先指定的偏召回探索，不是最佳F1配置。网格阶段消融复现最终事件完全一致：采音F1量化前76.4095→仅量化76.3754→对齐76.3170→最终镜像76.3144%；长条联合F1量化前2.8974→对齐2.5039%。结构对齐未改善参考匹配，不能以5ms代替听感验收，也未定位旧卡顿原因。172train/22val/25旧测试/127隔离与217缓存不变，反复验证不是独立测试，未用琵琶行或Snow选参。

琵琶行750→741采音，749逻辑物件（轻632/重33/Gemini65/长条18/特重1），对齐17根（延11/缩6）；Snow1099→1058采音，1076物件（轻892/重26/Gemini70/长条88/特重0），对齐69根（延45/缩24）。确认人声模式分别39.9/119.99秒，短于r1的104.8/196.26秒；存在真实人声被拒而回退普通采音的风险，召回改善不能等同人声来源更准确。22验证共533根长条调整（延325/缩206/等长平移2），13150个时间位置量化。

交付E:/下载/琵琶行_mikee_MDAC2试玩_人声前瞻r2_5ms.mdm（SHA2d2ff22e61a2df2f223c883b202eec2697bd155517e48eca51eaa79f282d51d3）和Snow Song Show_mikee_MDAC2试玩_人声前瞻r2_5ms.mdm（SHA9c4f6ddb1eacd9f746e207ec92193df9579173249426f18d6632892be0bf5d8e）。scene_01、mikee、9.5标签保留，music/demo/cover逐字节继承已审计原包，零音频偏移。新包内BMS127145/223030字节，相同事件旧格式657534/1278774字节，压缩存储不等于已证实游戏不卡。

独立预览http://127.0.0.1:8783/pipa/index.html及snow/index.html含v2/r1/r2实际事件、原曲/分离人声/击打对照，旧产物保留。593项回归全部通过；24曲前瞻状态、证据门控、5ms/长条/占用/BMS回读通过；两包ZIP、资源SHA、原音频和冻结权重/绑定源码身份通过；2页/12音频/24个HTTP206范围字节比对、页面脚本语法检查通过。初次用普通.venv生成预览缺torch，未产生输出；用指定pretraining-py313重跑成功。所有实际命令、身份、分项、源码快照、故障见tools/private/community/vocal-grid-r2-20261017/completion.json、source-final与实施记录.md。

工程已完成；r2人声真实性未验证，听感与实机待本轮反馈，未自动替换v2、发布模型或tag。


## LOG-20261017-004 用户选定 R1，区分 V 基线与 R 制谱方案

用户反馈当前人声前瞻r2/5ms版本“体感很不错”，明确定为R1，并要求记录部分后处理也适用于V2。本轮注册当前组合V2＋R1：models/registry.json升为schema2，current_model保留v2，新增current_recipe=R1与独立recipes列表。V系列为冻结模型基线及其原默认配置；R系列为依赖指定V的完整制谱方案，包括采音选择、后处理、时基和导出。V2历史已包含编排r1.1，不把V定义成裸网络，不把连续点击/三四连、高强度候选、相似段镜像都归作R1新增。R1共享这些模块，增加人声前瞻与密度门控、5ms网格，替换旧250ms补尾。不得把完整R1叠加到已跑完V2后处理的谱上。

执行 .venv/Scripts/python.exe tools/freeze_recipe_r1.py。清单recipes/R1/manifest.json SHA7fa440d87a3de4aa5bab8679a1c3c47af241c516db066dea85acfeeec4ecb1f7，冻结14份源码及既有评价身份，权重引用models/v2/weights，不产生新神经模型。V2清单及权重/源码/先验SHA不变。用户反馈限定当前版本，未推断两首均通过或泛化验收；历史报告pending保持，最新反馈独立记入R1清单。

生成E:/下载/琵琶行_mikee_MDAC2试玩_R1.mdm和Snow Song Show_mikee_MDAC2试玩_R1.mdm，与已认可r2包逐字节相同，SHA分别2d2ff22e…d51d3、9c4f6ddb…f5d8e，包内历史标题保留，未重新推理/修改谱。注册/源码快照/评价/两副本身份核验通过，证据tools/private/releases/R1-20261017/verification.json。未重复跑无改动的593项算法回归，未训练、发布新应用tag或替换桌面生成入口。设计21、models/README、recipes/R1/README、工程README及CHANGELOG已更新。

## LOG-20261017-005 R1两处长条截图反馈定位，未改冻结谱

用户随后指出Snow末段长条提前结束、长条内note希望间距>500ms，以及171–176秒交错长条影响游玩，要求先查看现状。实际空中长条252.565–256.310，下一点击259.725，中间3.415秒；原模型时长3.744591秒，尾256.309，后处理只取整1ms，没有截断。模型直接回归时长，100ms近邻对齐不处理数秒持续音边界。尾后伴奏特征仍有能量，但不能据能量证明同一声源或正确长条尾。

图2原模型已有地空交错0.5–0.7秒长条，R1端点对齐将地面7根的6个间隙全部压到0，形成171.995–178.340连续占用但仍7个事件；天空亦有3处零间隙。172–176秒双轨同时占用2.934→3.440秒，同轨硬重叠为0；规则只验合法，不验零间隙重新起按/交错长条负担。原预览矩形无头尾标记掩盖了独立边界。只读诊断页补明确头尾，不修改原预览。长条内部同轨相邻note≤500ms扫描：Pipa0对，Snow1对8.295/8.645、350ms；此口径不涵盖长条头尾重按，图2不能靠点击换轨规则修复。

命令 tools/.deps/pretraining-py313/Scripts/python.exe tools/diagnose_hold_feedback_r1.py --output tools/private/community/r1-feedback-holds-20261017。诊断JSON、标头尾HTML及用户截图副本保存，R1补充反馈见recipes/R1/feedback/20261017-holds.json，设计22记录原因与待修边界。后续候选需分开持续音尾检测、同轨延续/重按、跨轨换手与点击过密的语义，保留Gemini/双重击。当前未应用新规则、未生成或命名R2、未改变已冻结R1。

## LOG-20261017-006 试玩交付目录与长条反馈范围更新

用户指定今后的试玩包直接交付 E:/OneDrive - stu.sau.edu.cn/MuseDash/Custom_Albums，已确认该目录存在，并将约定写入工程根 AGENTS.md，后续打包使用该交付目录和独立版本文件名。本次没有搬迁旧包，也没有生成新包。

图1进一步明确为长条时长／结束点预测偏早，不是对齐问题；不能用扩大100ms吸附范围或无条件延长到下一note代替持续音结束判断。图2用户要求暂时搁置，停止进一步调查与修改，保留既有证据，待用户恢复。长条内点击>500ms的独立反馈仍登记为未实施。设计22与R1补充反馈JSON已同步；未改冻结R1算法、权重、谱面或包。

## LOG-20261017-007 独立长条补尾与内部点击间距候选

用户授权解决图1提前结束及长条内点击过密，图2继续搁置。新增hold_refinement_rc1.py/refine_holds_rc1.py，当前策略hold-feedback-rc2，未改冻结R1。声学稳定模板核验延续到明确边界，再接原5ms/100ms对齐；同轨前接长条链暂不处理。内部点击目标505ms，过密普通点击优先换轨，同位置同类型兼容合并；Gemini/双重击不拆，无法处理留原因。相似段仍最后编辑，若新增过密则拒绝镜像。

有效实验tools/private/community/hold-feedback-rc2-20261017：24曲均精确重放R1后再处理，规则/网格/BMS回读通过，48测试通过。Snow仅末段尾256.310→259.725（声学候选259.780）和8.645秒轻击地→空两处改动；图2 171–179秒逐项未改；Pipa未改，未重复出包。22验证1补尾7点击处理，剩余过密对0；采音F1 76.31435→76.31696%，轻击30.77378→30.78352%，长条头尾联合均未提升，不宣称总体预测改善。图1设计利用用户反馈，不作为独立泛化测试。

Snow Song Show_mikee_R1长条修订候选.mdm已交付E:/OneDrive - stu.sau.edu.cn/MuseDash/Custom_Albums，SHA6853836cbd48cf7144b8d63c825b1c366f7ca715f90ff0971f000f26c7623309。默认场景/mikee/原音频保持，音频封面逐字节一致。候选待听评/实机，未晋升R2或改注册表。预览review-snow/index.html已生成，浏览器安全策略拒绝file://自动打开，未绕过；媒体帧数检查通过，页面播放未检验。

首次rc1误改图2链末尾被断言阻止，没有投递；失败证据原目录保留，修正链退出规则后完整运行rc2。GPU解释器无pytest，测试改用既有.venv。详细配置、命令、身份及限制见设计23、实验identity/report/source、package-report与review audit。

## LOG-20261017-008 《アイ》V2 / R2待定双版试玩交付

用户提供E:/CloudMusic/DECO27,初音ミク - アイ.mp3，要求分别用V2和R2待定制谱。完整270.5866667秒，源SHA d792fe91e7c6b582ed1185b2177079ba456591f4429dfaad872b52314666021b。新缓存tools/private/playtests/20261017-ai-v2-r2；固定分离器仅运行一次，两版共用三流特征与采音网络预测。未重训、未按本曲调参，未触及数据分割/锁定测试/旧素材与既有版本。

V2使用冻结V2原方案（.3采音、编排r1.1、原250ms尾对齐及原导出时基）；R2待定使用R1人声前瞻+5ms+hold-feedback-rc2，不是正式R2注册或新权重。每版根据自己的采音落点重新运行完整排键历史。V2物件1051（轻710/Gemini156/长条164/重20/特重1）；R2待定1032（轻710/Gemini151/长条150/重20/特重1）。R2人声优先118.09秒，本曲没有触发新增声学补尾或内部点击换轨/合并，最终长条内部过密对0；不能把两版区别归因于最新两项修复。

已交付Custom_Albums中的アイ_mikee_MDAC2试玩_V2.mdm（SHA9efd9a7edd87d130f292caeb1b9b5b11b7ff2b15d7d3ff82784fa88c78111309）及アイ_mikee_MDAC2试玩_R2待定.mdm（SHA3e80267e437775297e742442d6da8284d8099025f0c8e1fdc1a196f9b4f216f2）。游戏内名称“アイ - V2 原方案”和“アイ - R2 待定”；mikee、默认scene_01、9.5标签。两包music/demo/cover逐字节一致，OGG波形偏移0样本、长度误差0；两谱占用/导出回读、ZIP资源及交付SHA通过。听感/实机待用户反馈，注册表未改，未自动晋升。

入口tools/generate_v2_r2_playtest.py，以指定GPU解释器分别运行--stage prepare和--stage infer，--source为上述MP3，--output tools/private/playtests/20261017-ai-v2-r2，--title アイ。随后分别调用package_predicted_playtest.py和package_grid_playtest_r2.py，prediction分别v2/r2-pending、parent为本批缓存、output为v2-package/r2-package、delivery为上述游戏目录。完整命令和身份见prepare-status/infer-status/identity/generation-source及两份package-report，最终复核completion.json。未改变冻结推理模块，只增加双版生成编排入口。

## LOG-20261017-009 全局1ms规则与V2人声增强试玩

用户否定R2整体体感，要求新制谱恢复1ms，将100ms端点对齐、持续音补尾、505ms内部点击间距设为全局后处理；再明确整个长条内及结束后500ms的note全部移到对向轨，Gemini改对向单轨。已新增global_postprocess.py与generation-defaults.json，新入口generate_current_playtest.py统一执行；AGENTS和registry.active_generation_defaults记录默认，冻结V2/R1/R2历史不改。桌面手动ChartIR流程未接神经模型，未改手工导出，不声称全局规则已接工作台按钮；本轮无web。

V2迭代为独立v2_vocal_bias.py推理偏好：确认人声起音帧候选odds最多×2，原0.3阈值、不压伴奏分数，局部30ms峰竞争，重新整曲排键。未训练/改权重或重分轨。共同后处理与人声策略分离，未来新自动方案必须共用前者。1ms与100ms端点平均、Gemini例外、500ms尾后含边界、两轨受限与双重击保护已有测试；相关59项通过。

实验global-1ms-vocal-v1-20261017的22验证+2试听工程、1ms/BMS回读通过。验证采音F1 75.8839→76.2457%，召回80.8786→82.5608%，误点4395→4629；长条联合1.9690→2.1588%，类别有升降，非泛化验收。806处区域调整/78 Gemini转单轨，过密对0，98项事件×轨道偏好未满足例外保留原因，硬冲突0。

琵琶行750→753落点/761物件，25处区域调整/1 Gemini转换，2处两轨受限；Snow1099→1124/1151物件，48处区域调整/3 Gemini转换/2声学补尾，5处受限或双重击保护。两包已直接交付Custom_Albums，文件名分别琵琶行_mikee_V2人声增强_全局1ms.mdm、Snow Song Show_mikee_V2人声增强_全局1ms.mdm；SHA f7c90140fc77868ecc6ed16cdb0794c996e18c5b4fc82e0148b92f7c393ffb73、f8b16738531aa9b56e9678243f0fe0600e1eba3dc34e2bffa9f193bd3ee6c603。原音频/封面、默认场景/mikee保持，待听评/实机，不发布新模型或tag。详细命令、规则、全部限制见设计24和实验identity/report/completion。


## LOG-20261018-001 全局严格转移/丢弃与特重优先

已按用户最新决定更新global-postprocess-1ms-v2：长条头/内部/尾后500ms能安全转移就转移，否则丢弃；原始目标note优先，20ms冲突邻域不再合并。Gemini→对向轻击，双重击→对向重击，区域外不降级。特重截断在途长条并清除冲突点击，截断不足30ms丢弃，之后不恢复。补尾/100ms对齐后避让，相似段最后且不得引入冲突。停止原505ms合并阶段，保留最终检查。AGENTS/默认配置/设计24已同步，冻结V2/R1/R2不改。

73测试通过；从原保存的神经输出重放22验证+2歌曲（不重训、不重分轨、不重推理），24谱规则/1ms/BMS审计通过，区域残留0、过密对0。22验证最终谱落点F1 76.2457→75.9984%，召回82.5608→82.0613%，为规则删除带来的取舍；采音输出未变，不能宣称模型提升。实际谱未触发特重截条，只在构造用例验证。

琵琶行761→757，删除4颗（64.704/117.184/117.594/117.803）；Snow1151→1142，删除8颗并将78.073双重击变为空中重击。旧7处例外均解决，两首长条几何未变。已交付Custom_Albums中的“琵琶行_mikee_V2人声增强_全局避让修订1.mdm”和“Snow Song Show_mikee_V2人声增强_全局避让修订1.mdm”，原包保留、无web。听感/实机待评。

实验tools/private/community/global-1ms-conflicts-v2-20261017（跨午夜完成），重放入口tools/replay_global_postprocess.py；.venv/Scripts/python.exe执行--parent tools/private/community/global-1ms-vocal-v1-20261017 --output tools/private/community/global-1ms-conflicts-v2-20261017/replay。逐项删除/转换在report与comparison，完整身份/命令/包SHA见identity和package-report；详细规则与限制见设计24最新修订。不是新模型发布，不恢复旧自动化。


## LOG-20261018-002 用户指定当前组合为V3，保留V2

用户明确“更改为v3，保留v2”。已执行tools/freeze_model_v3.py登记models/v3/manifest.json，V3=同V2神经网络权重+v2-vocal-soft-bias-v1人声偏好+global-postprocess-1ms-v2严格避让。V3并非重训；weights存独立副本，完整执行源码/配置和验证证据保存在source/evidence。V2整个目录逐文件SHA与集合复核不变，旧包保留。应用仍v0.1.0，未创建tag或推送。

registry.current_model=v3，current_recipe=null（V3自带默认方案）；last_selected_recipe=R1保留历史指针，R1的base_model仍为v2。generation-defaults.base_model=v3，generate_current_playtest.py按V3清单核验并加载；已实际完成CUDA加载检查，排键10,993,122参数，未重推理/重制谱。记录见tools/private/community/global-1ms-conflicts-v2-20261017/v3-selection-smoke/result.json。

Custom_Albums新增“琵琶行_mikee_MDAC2试玩_V3.mdm”（SHA baff063d10c1cfa8b4aeb568ebb9bf1889cff1496d89d026c55e2f828df49d96）和“Snow Song Show_mikee_MDAC2试玩_V3.mdm”（SHA cae76878ef06556ff3f64edd8e181281f70615002617d9fcc2d2c797a28b33a7）。只将info/BMS标题改为歌曲名 - V3，BMS事件语义、音频和其他资源保持修订1相同，原包逐字节不变。mikee/默认场景继承，未生成web。

完整冻结命令及V2目录摘要见models/v3/registration-audit.json。V3是用户授权命名，最新严格删除修订尚无单独实机反馈，不继承前版正向反馈为全面验收；工程73测试/24谱通过记录保留，规则删点造成参考分数略降仍如实记录。


## 2026-10-18 V3产品入口交接

当前用户已认可V3可玩并要求完善工作台，范围转为产品工作流；不恢复旧训练自动化。入口start.bat，五步AutoStudio，GPU适配tools/studio_generate.py，默认mikee/V3，显式可选V2历史。LLM配置右上角，参考RAG-ULTRA显式协议/API Root，不复制凭据，DPAPI本地密钥，仅歌曲名发送。预览微调占位。开场提示map2.talk，游戏位置待测；完整实现、身份和实测见设计25。

验收runtime/studio-qa-20261018：琵琶行V3=757事件，完整事件语义与原冻结包一致，音频长度/偏移0；验收包已交付Custom_Albums。V2选择=759事件；新12秒音频无缓存路径分离/特征/生成通过。取消重试、服务重启、持久化和备份恢复验证通过。未训练/未改冻结模型源码或权重/未创建tag。真实LLM待用户配置，页面视觉检查受浏览器URL工具策略阻止，游戏提示待实机。继续前先读设计25，不把这些未验收项标为通过。


### 2026-10-18产品入口后续修订

用户要求无口令、精简界面、根目录chart、项目管理弹窗与三语设置，已实现。覆盖前节runtime默认和安装令牌描述。当前8010服务使用chart；start.bat默认8000同样使用chart，运行目录锁避免双开。旧三个自动验收项目已复制迁移，源保留。删除可恢复；封面内嵌提取/上传/裁剪/实际打包SHA验证通过。联网找图未接入，预览微调继续占位。137后端/40前端逻辑及构建通过；页面视觉仍未验收。见设计25末尾与LOG-20261018-004。
