# 本地音频解码 profile 1

本轮私有安装位置：`tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin`，不修改系统 PATH，目录已忽略。

来源为 [FFmpeg 官方下载页](https://ffmpeg.org/download.html) 指向的 [Gyan Windows builds](https://www.gyan.dev/ffmpeg/builds/)。下载 `ffmpeg-release-essentials.zip`，与同站 `.sha256` 比对通过，实测版本 9.0.1；归档 SHA256 `fec81ae03971d9dd4be3ebe02e263bd2ec1d789483f931bdba5f5715e65da2e9`。保留随包许可；第三方构建是 GPLv3，当前仅私有开发使用，不表示已完成应用再分发许可验收。

`AudioDecoder(ffmpeg_path, ffprobe_path)` 只接受显式本地工具，缺项失败，不自动下载。真实集成测试当前指向上述锁定位置；新机器需提供相同工具或显式更新测试配置和身份记录。

支持格式白名单 wav/mp3/flac/ogg/aac/mov（实际容器必须可通过 pipe 解码）；一次仅一条音轨，mono/stereo，8–192 kHz。默认规范化 48 kHz signed PCM16LE，保留声道，不自动下混、响度归一或静音裁剪。gain 显式记录且溢出拒绝。其他格式/多声道/多音轨不静默降级。

以原生解码样本作为 transform 源；裁剪→重采样（ceil 长度，不符拒绝）→增益→补零。压缩编码器的延迟处理由锁定 FFmpeg 完成，不把容器 packet 编号冒充原始 sample。原始字节与规范 WAV 作为不可变 audio Artifact 的两个 sidecar 保存。

请求使用 pipe-only 输入协议与格式白名单，禁止 URL/播放列表/外部文件引用；字节、PCM、时长、每次子进程超时有限额，单线程解码。错误不返回 stderr 或文件路径。本适配器应在工作线程/作业中执行，不能阻塞 ASGI 事件循环。当前不宣称敌对本机程序安全沙箱。

2026-09-15 定向真实验证：`pytest tests/integration/test_audio_io.py -q` 10 passed，包括无损 PCM 对照、立体声、单声道重采样、裁剪原点、FLAC/MP3、坏文件/播放列表、截断、时长/字节/增益拒绝。超时异常路径为注入测试；原始用户音乐未使用。
