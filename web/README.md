# MDAC2 前端脚手架

仅 T003 基础页面，无编辑、播放、API 或导出能力。验证环境 Node 24.14.0 / npm 11.9.0。

```powershell
cd E:/project/MDAC2/web
npm.cmd ci
npm.cmd run typecheck
npm.cmd run build
npm.cmd run dev
```

开发/预览仅监听 loopback；不代表后续 API 写认证已经完成。package-lock.json 固定解析结果；不提交 node_modules/dist。TypeScript 暂用 5.9.3，未直接跟进新主版本，兼容性以本任务实测为准。

配置依据：[Vite 入门](https://vite.dev/guide/)、[Vue TypeScript](https://vuejs.org/guide/typescript/overview.html)，2026-09-15 核对。类型检查单独用 vue-tsc，build 同时执行检查。
