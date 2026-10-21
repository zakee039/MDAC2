# 模型基线与制谱方案

V1/V2/V3 的冻结权重现已通过 Git LFS 发布，路径保持为 `models/<版本>/weights/`；V2/V3 的 `run-prior.json` 也在仓库中。三版共用相同的两个网络检查点，LFS 按内容去重。历史 manifest 内的“本地权重不发布”文字为冻结时的记录；当前发布方式以本说明为准，manifest 和权重内容未修改。

安装 Git LFS 后克隆仓库；已有克隆可在工程目录执行 `git lfs install` 和 `git lfs pull` 获取完整权重。不要把 GitHub 源码 ZIP 中可能出现的 LFS 指针文件当作权重。Python 环境、FFmpeg 及分轨/测拍的第三方模型仍按工程安装说明准备。

当前工作台首选策略：**V4**（2026-10-21）；应用版本 **v0.5.0**。V4 复用 V3 权重、独立处理排键与节拍，不再标为候选；v0.5.0 追加可选 **V5**（同样复用 V3 权重，独立检测长条与小 Boss，经用户实机试玩后发布）。V2/V3 冻结产物与 R1 完整保留。

- V5 清单与策略说明见 [v5/README.md](v5/README.md)，历史修订快照保留在 `models/v5/revisions/`。
- V3冻结当前人声柔性增强＋全局1ms严格避让方案，两份网络权重与V2相同，没有重训。清单见[v3/manifest.json](v3/manifest.json)，本地权重位于`models/v3/weights/`。
- `registry.json.current_strategy=v4`、`current_model=v3`（复用权重）；以下 V3 批处理入口保持历史语义：`current_recipe=null`表示使用V3自带默认方案，无R覆盖。R1仍明确依赖V2，不能自动叠加到V3。新监督制谱通过`tools/generate_current_playtest.py`读取`recipes/generation-defaults.json`。
- 两份V3试玩包已放入指定Custom_Albums目录；仅替换修订1包的显示标题，事件与音频相同。V2目录/权重/源码/清单与旧包未改动。

以下为历史V2/R1登记说明：

- V 系列冻结模型基线及登记时的默认配置。V2 包含 V1 的采音／排键权重和编排 r1.1，不能称为裸网络。
- R 系列冻结基于明确 V 版本的完整制谱方案，包括候选选择、共享及替换的后处理、时间网格和导出。R1 复用 V2 权重，未重训。
- [registry.json](registry.json) 是唯一当前选择入口：`current_model` 为 `v2`，`current_recipe` 为 `R1`，清单分别位于 [V2](v2/manifest.json) 和 [R1](../recipes/R1/manifest.json)。schema 2 保留模型字段，新增独立方案字段；R1没有放入模型列表。

登记与桌面默认流程接入分开：本轮保留已验证的版本化 CLI 入口，没有替换桌面默认生成。冻结版本不能热改；完整继承、替换和实验命名关系见 [版本规范](../设计/21-V系列与R系列版本规范.md)。
