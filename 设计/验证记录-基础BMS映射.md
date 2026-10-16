# T027 基础 BMS 格式证据（2026-09-20）

状态：真实模板与固定加载器源码已核对；尚未在游戏中导入，不是 T033 的替代验收。

- 模板：[MDBMSC Linux 包 map2.bms](https://github.com/kokolab/mdbmsc-installer-linux/blob/ea3585b4f95b824a911450546aa08e4d44814409/app/template/map2.bms)。实际下载字节 SHA256：47b2ca5ca91faba1ddd9a0ac7cc0e4daf0cc46a3bd7ee4f118bcd4362014adc9。模板声明 Small=01、Gemini=0E、Hold=0F、music=10、LNTYPE=1。
- 加载器：[CustomAlbums Data/Bms.cs](https://github.com/MDMods/CustomAlbums/blob/bdfc7e6c2f4be0c8b5c5b405ba26b287397f956f/Data/Bms.cs)。实际字节 SHA256：66d74ae2df356289dda2d5847b23f6c5f4fee05a6a2c6f4a60a18275204a2552。Channels/路径映射：13/53 为 air，14/54 为 ground；Hold 按同通道同对象的下一标记配尾。
- [BmsLoader.cs](https://github.com/MDMods/CustomAlbums/blob/bdfc7e6c2f4be0c8b5c5b405ba26b287397f956f/BmsLoader.cs) 的 ProcessGeminis 配对同刻、异轨 Gemini；WAV 文本被忽略，对象 ID 才是游戏语义。

旧 MDAC 的 note_defs.py 将地空通道倒置，不沿用。新 profile 标识 customalbums-bdfc7e6-basic-v1；旧文件不修改，未来旧谱导入必须显式选择源 profile 再迁移 IR，不能复用不明通道含义。

首版范围：单轨普通键、单轨 Hold、成对 Gemini；拒绝 accent（未证明其与加载器血量/对象大小的语义等价）、双轨普通/Hold、障碍、Ghost、Masher、Boss、方向/速度/视觉效果。不以猜测或静默降级补齐。

模板是空谱，因此 golden/basic.bms 是 MDAC2 自建的格式样例，不能称为真实实玩曲。第三方完整证据仅下载到忽略目录 tools/private/profile-evidence，不重新分发其音乐/图像/音效或运行安装器。没有下载权重。

时间表达与游戏延迟行为需 T030/T033 单独验证。当前 profile.game_verified=False；不能声称所有游戏/加载器版本兼容。
