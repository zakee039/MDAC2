# 模型发行 v2

用户认可的当前组合：v1 两份神经网络权重 + 编排后处理 r1.1。权重未重新训练；发行编号与历史训练实验编号分开。

weights/onset.pt、weights/arrangement.pt、weights/run-prior.json 是本地冻结副本，不进入 Git。manifest.json 记录完整 SHA、策略、源码快照与试听范围。

《琵琶行》获得正向实机反馈；《Snow Song Show》仍有用户报告的个别奇怪之处。客观分类质量未显著提高，不能扩大为所有歌曲验收。

人声优先采音属于后续独立实验，不包含在 v2。保留 v1、旧包和 v0.1.0 tag，桌面默认生成入口尚未替换。
