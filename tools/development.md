# 核心开发环境（T002）

验证基线为 Windows x64 / CPython 3.12.14，项目版本唯一来源为 `src/mdac2/__init__.py` 的 `__version__`。`requires-python >=3.12` 是语法下限，并非所有新版本或模型栈已验证。核心无运行依赖；不复用 Anaconda 的模型环境。

在工程根使用显式解释器创建 `.venv`（禁止覆盖已有环境），之后无需激活：

```powershell
& 'C:/Users/mamin/AppData/Roaming/uv/python/cpython-3.12.14-windows-x86_64-none/python.exe' -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements-dev.lock
./.venv/Scripts/python.exe -m pip install --no-build-isolation --no-deps -e .
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe -I -c "import mdac2; print(mdac2.__version__)"
```

开发锁文件记录本次实际解析的完整测试/构建依赖版本；不是带哈希的跨平台供应链锁。更新依赖须重新验收，不安装 FastAPI/torch。尚未提供业务命令、模型或制谱能力。
