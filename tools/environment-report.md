# T001 环境清点报告

检查日期：2026-09-15 10:57–10:58（Asia/Shanghai）。工程：`E:\project\MDAC2`。参考：设计 01、10 及 T001 卡片。

仅执行本机只读清点；没有安装依赖、联网下载、导入模型库、执行推理或扫描私人文件。ready 仅表示本项查询成功，不代表业务或兼容性验收完成；missing 限定在已检查范围，unknown 表示尚未验证。磁盘、内存和显存是瞬时快照。

## 1. 系统、硬件与命令

| 项目 | 状态 | 实际结果 |
|---|---|---|
| OS | ready | Windows 11 专业版，64 位，10.0.26200，build 26200 |
| CPU | ready | Intel Core Ultra 7 255H；16 核 / 16 逻辑处理器（CIM 返回值） |
| RAM | ready | 总计 32961912 KiB；可用 8400712 KiB |
| 工作盘 E: | ready | NTFS；总计 1524102066176 bytes，可用 1304049016832 bytes；未做写入/性能测试 |
| Node | ready | v24.14.0；`C:\Program Files\nodejs\node.exe` |
| npm | ready | 11.9.0；使用同目录 `npm.cmd`，不依赖 npm.ps1 执行策略 |
| Git 命令 | ready | 2.50.1.windows.1；`C:\Users\mamin\scoop\shims\git.exe` |
| MDAC2 Git 仓库 | missing | `git status --short` 返回 128：not a git repository；未初始化仓库 |
| FFmpeg / ffprobe | missing | `Get-Command` 未找到，执行 `-version` 均 CommandNotFoundException；精确版本 unknown，不代表整机未安装 |
| uv / nvcc | missing | PATH 未找到；版本 unknown；不要求 T002 安装 CUDA Toolkit |
| NVIDIA GPU | ready | GeForce RTX 5060 Laptop GPU；驱动 616.92；compute capability 12.0；显存总计 8151 MiB、可用 5647 MiB |
| NVIDIA-SMI / NVML | ready | 均为 616.92；CUDA UMD 13.4 是驱动报告值，不是已安装 Toolkit 或 torch CUDA runtime 版本 |
| 其他显示适配器 | ready | Intel Arc 140T GPU (16GB)，驱动 32.0.101.6874；GameViewer Virtual Display Adapter，驱动 15.6.5.199 |
| CUDA / 模型实际运行 | unknown | 未导入 torch、未分配 GPU、未跑算子；不能由硬件查询推出模型兼容性 |

## 2. Python 与限定的包元数据清点

四套解释器均实际运行标准库探针成功（exit 0），不能混用默认命令。

| 入口 | 精确版本 | 解释器路径 |
|---|---|---|
| `python` | 3.13.12，Anaconda，MSC v.1942 AMD64 | `D:\anaconda\python.exe` |
| `python3` | 3.11.9，MSC v.1938 AMD64 | `C:\Users\mamin\scoop\apps\python\current\python.exe` |
| `py` 默认 / `py -0p` 的 3.14 | 3.14.3，MSC v.1944 AMD64 | `C:\Python314\python.exe` |
| `py -0p` 的 Astral/CPython3.12.14 | 3.12.14，MSC v.1944 AMD64 | `C:\Users\mamin\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe` |

以下版本来自 `importlib.metadata`，只证明 distribution 元数据存在，不证明可导入/可运行；“missing”仅指该解释器没有对应元数据。

| 包 | Python 3.13.12 | Python 3.11.9 | Python 3.14.3 | Python 3.12.14 |
|---|---|---|---|---|
| pip | 26.0.1 | 24.0 | 25.3 | 26.2.1 |
| pytest | 8.4.2 | missing | missing | missing |
| setuptools | 80.10.2 | 65.5.0 | missing | missing |
| fastapi | 0.141.1 | missing | missing | missing |
| torch | 2.12.0 | missing | missing | missing |
| demucs | 4.1.0 | missing | missing | missing |

设计候选 `beat-this`、`librosa`、`mir_eval`、`audio-separator`、`huey`、`allin1`、`basic-pitch`、`ortools` 在上述四套解释器均 missing。RoFormer 后端、权重、许可和哈希 unknown，未搜索权重文件。Vue/TypeScript/Vite/wavesurfer.js 的项目本地依赖 missing：工程开始时仅有 `设计/`，未建立 web 或锁文件；全局 npm 包不在检查范围。

## 3. 实际检查命令与失败面

执行目录为 `E:\project\MDAC2`，PowerShell。下列只读命令是本轮实际执行语句（循环中的探针针对上表四个解释器分别执行）。

```powershell
Get-ChildItem -Force
rg --files -g AGENTS.md
git status --short
Get-Command python,python3,py,node,npm,ffmpeg,ffprobe,nvidia-smi,nvcc,uv,git -ErrorAction SilentlyContinue
python --version
py -0p
python3 --version
node --version
npm.cmd --version
git --version
ffmpeg -version
ffprobe -version
nvcc --version
uv --version
Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,OSArchitecture,TotalVisibleMemorySize,FreePhysicalMemory
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='E:'" | Select-Object DeviceID,FileSystem,Size,FreeSpace
Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,compute_cap --format=csv
nvidia-smi --version
```

`Get-Command` 实际按名字循环，以 JSON 输出找到的路径、显式记录 missing；CIM 输出也经 JSON 格式化。`rg` exit 1（没有 AGENTS.md 匹配）；Git 仓库检查失败；四个缺失命令未启动子进程，因此没有子进程退出码。初次批量版本命令最终 exit 0 仅来自末尾 Git，不能掩盖前述缺项。随后单独记录 Node/npm/Git 与 python3/NVIDIA-SMI 版本命令 exit 0；CIM 查询成功、GPU CSV 查询成功。

包检查使用 `-B -c`（不写 pyc），核心探针如下；没有调用包的导入逻辑，也没有用 pip 发起联网查询：

```python
import sys, json, importlib.metadata as m
names = ["pip", "pytest", "setuptools", "fastapi", "torch", "beat-this",
         "librosa", "mir_eval", "demucs", "audio-separator", "huey",
         "allin1", "basic-pitch", "ortools"]
installed = {d.metadata["Name"].lower().replace("_", "-"): d.version
             for d in m.distributions() if d.metadata["Name"]}
print(json.dumps({"executable": sys.executable, "version": sys.version,
                  "packages": {n: installed.get(n.lower().replace("_", "-"), "missing")
                               for n in names}}, ensure_ascii=False))
```

## 4. 交接结论

T001 的只读清点可验收，缺项没有被当作成功能力。下一任务 T002：显式选定解释器和隔离策略，再验证包导入与最小测试；现有 Anaconda 的 pytest 只是可用候选，未替工程冻结版本，也不授权改动共享环境。T003 有 Node/npm 命令基础，但依赖安装、版本兼容与构建需在其任务验收。音频任务前需解决 FFmpeg/ffprobe 的显式可执行路径；模型任务另行验证驱动/runtime/算子/权重兼容。无业务代码可回归；未执行真实音频、模型、前端构建或游戏验收。
