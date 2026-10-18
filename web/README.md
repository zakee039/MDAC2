# MDAC2 前端工作室

Vue 3 + TypeScript。当前首页为V3自动制谱四步流程：导入歌曲、确认信息、自动制谱、导出。右上角配置LLM，默认创作者mikee；项目列表可重开。

## 启动完整应用

在 `E:/project/MDAC2` 双击 `start.bat`。启动器检查本项目环境、构建前端，然后启动 MDAC2 的同源本地服务。打开终端输出的地址（默认 http://127.0.0.1:8000），打开即用，不需要口令。项目存放在工程根目录chart。

如果默认端口被占用，可在根目录运行 `./start.bat --port 8001`。Python 环境、服务依赖与 FFmpeg 的准备方法见 [本机工作台](../tools/start-workstation.md)。不要使用旧 MDAC 的启动器。

## 前端开发与验证

验证环境 Node 24.14.0 / npm 11.9.0。

```powershell
cd E:/project/MDAC2/web
npm.cmd ci
npm.cmd test
npm.cmd run build
npm.cmd run dev
```

Vite 开发服务仅用于界面开发，未配置 API 代理。完整业务联调请构建后使用 MDAC2 本地同源服务；无需放宽 Host、Origin 或令牌检查。

## 界面结构

- App.vue：框架、首页、导航、使用指南和右上角LLM入口。
- pages/projects/AutoStudio.vue：持久化项目、LLM建议与配置、V3/V2选择、异步任务轮询/取消/重试、下载与交付。
- studio.css：已有浅色工作室布局与共享样式。
- Projects.vue / Workstation.vue 与旧编辑面板：历史实现保留，当前首页不挂载；预览微调后续独立接入。

API仅本机访问，检查同源Host/Origin与跨站请求；没有用户口令。LLM密钥不回显、不写浏览器存储。编辑信息尚未确认时不能直接进入生成/导出；确认更改后使当前输出失效，旧尝试保留。轮询在切换项目或离开页面时失效。

应用版本仍v0.1.0。本轮typecheck/build与40项既有前端逻辑测试通过；浏览器视觉自动验收被工具URL策略阻止，未冒称视觉验收完成。详见设计25。


2026-10-18精简界面：App.vue只保留文件夹按钮/居中品牌/设置按钮；StudioDialog.vue使用原生dialog（焦点隔离、Escape关闭）。AutoStudio支持项目搜索、新建/载入/删除/恢复与封面选择、预览裁剪。studioI18n.ts提供zh/en/ja，设置保存至服务端并在启动读取。删除为软删除；当前模型不变。


2026-10-18项目流程更新：默认导入页，项目管理直接显示全部项目，右上角新建返回导入；删除永久清除该项目本地文件夹及登记，不再提供回收站。切换/重新导入先保存当前信息草稿和封面裁剪，重新导入建立独立项目。导入常见音频/视频自动抽取第一音轨为44.1kHz双声道float32 WAV，导出OGG。完整约束见设计25。
