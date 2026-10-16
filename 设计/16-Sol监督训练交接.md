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
