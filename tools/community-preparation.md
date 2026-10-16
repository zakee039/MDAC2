# 社区谱本地准备

最初阶段复制、解压、盘点并筛选；后续已增加六类标签、角色隔离、分轨特征、监督张量和前向检查。所有入口都不训练、不改原文件、不发布模型。当前入口见文末和 `设计/16-Sol监督训练交接.md`；下文早期统计保留作历史记录。

```powershell
.venv/Scripts/python.exe tools/prepare_community.py --source "E:/OneDrive - stu.sau.edu.cn/MuseDash/Custom_Albums" --target "E:/project/MDAC2/tools/private/community/20260926-preparation" --ffmpeg "E:/project/MDAC2/tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe" --ffprobe "E:/project/MDAC2/tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin/ffprobe.exe"
```

目标必须不存在，不覆盖旧准备结果。重复准备使用新目录；原目录 OneDrive 按正常文件读取，未改变同步设置。解压失败会保留 staging 和清单错误；只有 CRC 校验完成后才出现成功展开目录。

查看 `inventory.json` 的 summary/errors，确认 extracted_packages 与全部包数一致；`selected-9-10.json` 的 charts 为候选清单，各 audio_path/chart_path/raw_label_path 对应准备目录内的原件和来源。明确待复核的等级冲突、解析问题、家族关系与对象语义；角色未冻结、training_eligible=false 表示当前等待架构确认，不表示永久不可训练。

完整架构提案在 `设计/11-社区谱数据与训练架构待确认.md`。真实批次验收统计另记于施工日志。

可直接打开 `候选谱面清单.md` 浏览全部 320 张筛选结果和原始 BMS 链接，`准备验收.md` 为批次摘要。再次校验副本/展开内容：

```powershell
.venv/Scripts/python.exe tools/verify_community.py tools/private/community/20260926-preparation
```

本轮 503/503 包展开成功，515 个副本文件、2,722 个展开文件通过哈希复核；572 张原谱中筛选 320 张，310 份不同规范 PCM、约 13.89 小时。112 张候选存在等级来源冲突，保留提示；原始 BMS 解析错误为零。解码和校验成功不等于全部对象语义已经支持。

## 当前范围：8.5～10.5，含两端

上述是首次准备的历史结果。用户于2026-09-27调整范围，当前采用 `selections/rating-8.5-10.5-v2/selection.json`。在全量572张清单上重新筛选，不重复复制、解压或覆盖旧清单：

```powershell
.venv/Scripts/python.exe tools/select_community.py --root tools/private/community/20260926-preparation --output tools/private/community/20260926-preparation/selections/rating-8.5-10.5-v2 --minimum 8.5 --maximum 10.5 --ffmpeg tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe --ffprobe tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin/ffprobe.exe
```

该输出目录已存在，重跑请使用新的版本目录；工具拒绝覆盖。等级优先级继续为 modern→legacy→BMS，边界以 Decimal 判断，不先取整；来源冲突记录待审查。

当前结果：346张独立原谱，346/346主音频核验成功，338份不同规范PCM，约15.23小时，336个临时家族；新增33张、移除7张；135张等级来源冲突，0原始语法解析错误、0音频核验失败。仍未划分训练／验证／测试集，training_eligible=false，未应用归并表。

首次 v1 重筛显露两个旧问题：两个包提供 music.mp3，而原工具只搜索 music.ogg。现准备和筛选工具都按唯一主文件 music（OGG/MP3/WAV/FLAC/M4A，忽略大小写）定位，排除 demo；存在多个主文件时报错，不猜选。实际两个 MP3 均已完整解码并登记身份，新清单把旧问题移入 resolved_preparation_issues，原 inventory 不改写。v1保留作排查记录，v2为当前结果。

筛选只在解码工具身份一致时复用旧PCM身份，同时重查歌曲文件hash；本次新增解码34份压缩音频。各谱列出 audio_status；错误写入 errors，命令返回非零。候选清单、等级冲突提示和逐通道原始物件计数同时生成；后者不是归并后的学习事件统计。

当前标签规则见 `设计/12-训练标签映射规范.md` 和 `设计/13-社区物件归并表草案.md`。下一步仍是确认架构和映射，再实现时基／标签转换；这条准备命令不会运行分离器或训练器。

## 已落地的训练前流水线

当前使用 `training-preparation-v3/manifest.json`，不是早期的selection清单。六类标签归并及整曲角色已固定，219张自动可用／127张隔离，训练172／验证22／锁定测试25。原选择清单和raw-labels保持不变。详细环境、接口、工程默认值及尚未进行的人工验收见 `设计/15-联合制谱模型与训练数据契约.md`、`设计/16-Sol监督训练交接.md`。

构建新版本标签用 `tools/build_training_dataset.py --help` 查询入口，输出必须选择新目录。当前v3和监督张量不覆盖重建。全曲分轨／特征可中断后复用已完成缓存；它会重新核验身份及文件hash，失败staging保留供检查：

```powershell
tools/.deps/pretraining-py313/Scripts/python.exe tools/prepare_training_features.py --manifest tools/private/community/training-preparation-v3/manifest.json --output tools/private/community/pretraining-feature-cache-v1
```

只准备217份可用音频；默认不对隔离谱额外计算。推理禁网络、使用本地固定权重。缓存元信息记录实现hash，生成中不要修改audio_features.py；缓存不能跨不同实现冒用。

生成监督张量时用新的output目录（现有tensors已生成219份，不重复执行覆盖）：

```powershell
.venv/Scripts/python.exe tools/build_training_tensors.py --manifest tools/private/community/training-preparation-v3/manifest.json --output tools/private/community/training-preparation-v3/tensors
```

特征全部完成后运行CPU／CUDA前向和 `tools/check_pretraining_ready.py --help` 对应的全量工程检查。正式loader默认拒绝未完成缓存及锁定测试；调试选项不得带入训练配置。工具只生成随机初始化网络的前向报告，不提供训练循环、训练checkpoint或质量成绩。全部实际入口与下一步交给Sol的事项见16号交接。
