# T013 — Windows SqliteHuey 隔离实验

2026-09-15；CPython 3.12.14；Huey 3.4.0；结论 **adopt（仅传输后端）**。

命令：`tools/experiments/huey/.venv/Scripts/python.exe tools/experiments/huey/probe.py`。
修正 SQLite 显式 close 后连续两次 exit 0。隔离依赖见同目录 requirements.lock，未加入核心环境。

| 实测项 | 结果 |
|---|---|
| Windows 单线程 consumer 子进程启动/优雅停止 | 成功，所有子进程 wait 回收 |
| 已排队 priority 10/1 | 10 先执行 |
| 首次异常，retries=1 | 两次 attempt，一次 effect |
| 投递后 outbox ACK 前故障的重复投递 | 两次消息，业务条件更新仅一次 effect |
| 尚未运行的 revoke | 不执行业务 |
| 已运行的 revoke | 继续执行，不能用作强取消 |
| kill consumer，重启 | 排队任务仍执行；已出队的在途任务没有自动重投 |

采用原因：本地持久化优先级/消息传输可复用，Windows thread consumer 可运行。适配成本明确：业务 SQLite 保留唯一 Job/Run 权威；outbox 至少一次投递；worker 条件领取；新 run 重试；租约过期必须先确认旧进程退出。运行函数只派发隔离计算，Huey 自身重试不替代业务 Run 重试。不得采用其结果库作为业务成功权威；不得让 consumer thread 直接执行模型重计算。

原型 synthetic ledger 仅证明跨库重复投递的条件去重模式；不是生产完整 outbox、lease 或模型集成验收。生产只保留 Huey 一种队列实现，T015 实现后另验收。不把 crash 后不重投解释成业务任务丢失可接受；T016/T017 必须恢复。

首跑所有行为断言通过，但临时目录清理 exit 1：sqlite3 connection 上下文只提交/回滚，并不关闭句柄。已改为 finally close，并关闭 Huey storage 后复跑。首次失败的临时测试目录可能残留，未扫描或删除其他临时目录。

上游依据：[Huey 官方 guide](https://huey.readthedocs.io/en/latest/guide.html)、[PyPI Huey 元数据](https://pypi.org/pypi/huey/json)（本轮核对）；实际 API 以隔离安装的 3.4.0 源码与上述运行结果为准。
