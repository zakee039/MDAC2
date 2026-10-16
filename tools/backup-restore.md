# 一致性备份与恢复

本版提供显式 **整个工作区** 备份，包含所有工程、历史修订、引用音频和已登记产物。`backup.json` 记录工程清单与逐文件长度/SHA-256；不包括模型、队列传输库、staging 或源码。选择目标前确认它适合存放整个工作区；没有按工程筛选的导出功能。

```powershell
.venv/Scripts/python.exe -m mdac2.application.backup backup --runtime E:/project/MDAC2/runtime --target F:/MDAC2-backups/snapshot-001
.venv/Scripts/python.exe -m mdac2.application.backup restore --backup F:/MDAC2-backups/snapshot-001 --target E:/project/MDAC2/restored-runtime
```

路径为示例，程序不假定 F: 存在，也不自动建立远端备份。备份及恢复目标必须不存在；失败保留独立 `.staging-<uuid>` 诊断目录，不覆盖原件。SQLite 使用在线 backup 读取 WAL 一致性快照，随后复制不可变引用闭包；快照之后的新修订不属于本次备份。

恢复校验文件、数据库、引用闭合、谱面时间绑定及规范 WAV；合法静音不会当全零损坏。源中正在运行的作业仅在副本内标 interrupted/exit_confirmed，不能继承源进程；queued 重新置为待投递。恢复不会接触原实例的进程或数据。可用 build_app 的 runtime_root 指向恢复目录；启动器支持 --runtime 指定目录。

这是本地恢复演练能力。正式灾备需另一个故障域，不能将同物理磁盘的目录复制称为已完成灾备。没有实测断电持久性。
