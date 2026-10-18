# 模型发行 V3

2026-10-18用户指定当前组合为V3，V2完整保留。

V3 = V2原神经权重 + 柔性人声采音偏好 + 全局1ms严格避让后处理。不是重新训练的网络。

weights/onset.pt、arrangement.pt与V2对应SHA完全相同，run-prior.json也是冻结副本。source保存本次完整执行源码，manifest记录配置、评价与包身份。

新监督制谱入口tools/generate_current_playtest.py读取V3；桌面手工ChartIR流程不变。冻结R1仍依赖V2，不自动组合到V3。应用仍为v0.1.0，不新增Git tag。

最新两份V3包仅更改游戏内标题，谱面/音频与全局避让修订1相同；该修订待用户实机反馈。
