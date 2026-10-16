# T039 BeatThis 本地真实推理实验

日期：2026-09-21。结论：允许接入显式 CPU 实验能力，保留主模型 final0 为分析候选、小模型 small0 为显式低资源选项；没有实曲质量或 GPU 验收结论，不自动批准时间轴。

## 环境与复现

用户提供 `D:\TORCH` 中的 cp313 Windows wheel，安装到 `tools/.deps/beat-this-py313`，Python 3.13.12。核心 `.venv` 不变。torch 2.14.0+cpu、torchaudio 2.11.0+cpu、beat-this 1.1.0；完整本地文件 URL、SHA256 和依赖见 `tools/experiments/beat_this/requirements-py313-cpu.lock`。这组版本号不一致，故仅按已实测的 MelSpectrogram/BeatThis 路径放行，不泛化为全部 torchaudio 原生功能兼容。

`pip check` exit 0；导入与 MelSpectrogram 运行成功。CPU float32，2 threads，minimal postprocessor。没有安装 torchvision，也没有启用 GPU。原 3.12 实验环境保留，不混装 cp313 wheel。

```powershell
tools/.deps/beat-this-py313/Scripts/python.exe tools/experiments/beat_this/matrix.py
tools/.deps/beat-this-py313/Scripts/python.exe tools/experiments/beat_this/verify_local.py
.venv/Scripts/python.exe -m pytest tests/contract/test_beat_this.py tests/unit/test_timing_fit.py tests/unit/test_review.py -q
```

三条命令 exit 0；六次真实推理、两种坏权重拒绝、17 项核心契约测试通过。matrix 使用调用它的解释器，避免悄悄跑回另一环境。

## 来源、许可和身份

[上游仓库与许可说明](https://github.com/CPJKU/beat_this) 于本日核对；代码与发布权重 MIT。训练素材权利单独处理。本次只用 `samples.py` 固定种子生成的 18 秒双声道 PCM16，含鼓击与音高循环，无用户歌曲、无训练集下载。合成 fixture 不是独立真实音乐基准。

模型来源、长度、下载字节 hash 保存在本地 `tools/private/beat-this/download-manifest.json`；没有上游签名保证。

| 模型 | SHA256 |
|---|---|
| final0 | 8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331 |
| small0 | 6074be2c4d490c5f6101fcc374a1ec72ae93456e23bb6019783b849f5dc7d47b |

每次真实推理保存观测、logits、模型和音频 hash、精确依赖、已安装 inference/preprocessing 源码 hash 至该目录的 `.observation.json` / `.metrics.json`。FFmpeg 规范化产物与模型输入 PCM 由同一生成器提供，样例 manifest 不进入默认源代码仓库。

## 结果

F1 使用 mir_eval 0.8.2、70 ms 容差，包含开头/结尾，不进行首五秒裁剪。耗时是模型加载与推理调用，未计 Python 冷启动、FFmpeg 导入和后续指标计算；内存是进程累计 peak working set。

| 模型 / 样例 | beat F1 | downbeat F1 | 耗时秒 | 峰值 MiB | 拟合 p95 / max ms |
|---|---:|---:|---:|---:|---:|
| final0 / 固定 4/4 | .9577 | 1.0000 | .904 | 462.3 | 0 / 20 |
| final0 / 弱起 4/4 | .9859 | 1.0000 | .765 | 463.7 | 0 / 0 |
| final0 / 6/8 | .9787 | .7742 | .778 | 461.3 | 10 / 20 |
| small0 / 固定 4/4 | .9577 | 1.0000 | .369 | 369.0 | 19.44 / 20 |
| small0 / 弱起 4/4 | .9859 | 1.0000 | .351 | 369.3 | 0 / 0 |
| small0 / 6/8 | .9787 | 1.0000 | .348 | 369.2 | 5 / 5 |

全部六组被固定速度拟合接受，但这并不说明拍号或强拍正确。final0 在 6/8 把部分中间 pulse 也报成 downbeat（19 个 vs 12 个参考）；固定 4/4 两模型在 0 秒及尾部静音继续报点；弱起观测统一约 +10 ms。6/8 小模型的结果更好只适用于该样例，不据此更换默认模型。拟合残差衡量观测内部一致性，不是相对真值的偏移；未调用听检也未声称人工批准。

## 本地失败行为与接线条件

本地 loader 使用已打开的文件对象和 `torch.load(weights_only=True)`，不走会自动下载的上游短名 loader。六次 observe 调用期间禁用 socket；missing/corrupt 探针均明确拒绝，网络尝试 0，实测分别约 0.09 / 1.34 ms。模型注册表在推理前后验证 hash，缺失时不触发 backend。

T040 可集成这条已验证 CPU 路径，但必须保留候选、来源、失败、人工复核、局部覆盖与已有谱面迁移，不自动推断强拍起点。真实歌曲、设备精度比较及游戏验收仍未完成。
