<script setup lang="ts">
import { ref, shallowReactive } from 'vue'
import { ProjectsApi } from '../../services/projects.ts'
import { ProjectSession } from './ProjectSession.ts'
import Workstation from './Workstation.vue'
const token = ref(''), title = ref(''), id = ref(''), merged = ref(''), parseError = ref('')
const session = shallowReactive(new ProjectSession(new ProjectsApi(() => token.value)))
const workDirty=ref(false)
function merge() {
  try { session.merge(JSON.parse(merged.value)); parseError.value = '' } catch { parseError.value = '请输入审阅后的 ChartIR JSON，保存时由服务端严格校验。' }
}
</script>
<template>
  <section aria-labelledby="projects-title">
    <h2 id="projects-title">工程</h2>
    <label>本机安装令牌 <input v-model="token" type="password" autocomplete="off" /></label>
    <p>令牌仅保存在当前页面内存。通过同源本地服务连接。</p>
    <fieldset :disabled="session.busy || !token">
      <form @submit.prevent="!workDirty && session.create(title)"><label>工程名称 <input v-model="title" maxlength="256" required /></label> <button :disabled="workDirty">新建工程</button></form>
      <form @submit.prevent="!workDirty && session.open(id)"><label>工程 ID <input v-model="id" required /></label> <button :disabled="workDirty">打开工程</button></form>
      <p v-if="session.busy" role="status">正在处理…</p>
      <p v-if="session.error" role="alert">{{ session.error }}</p>
      <template v-if="session.project">
        <h3>{{ session.project.title }}</h3>
        <p>工程 ID：{{ session.project.project_id }}</p>
        <p>当前修订：{{ session.project.head_revision ?? '尚未保存谱面' }} · {{ session.dirty ? '有未保存修改' : '已同步' }}</p>
        <button :disabled="!session.dirty || session.conflict" @click="session.save()">保存谱面</button>
        <button :disabled="!session.dirty" @click="session.discard()">放弃本地草稿</button>
        <p v-if="!session.draft">工程已建立；请在谱面编辑器生成草稿后保存。</p>
        <Workstation :key="session.project.project_id" :project-id="session.project.project_id" :token="token" @dirty="workDirty=$event" @saved="session.open(session.project!.project_id)" />
        <section v-if="session.conflict" aria-label="保存冲突">
          <h3>保存冲突：本地草稿已保留</h3>
          <button @click="session.compare()">获取最新版本并比较</button>
          <template v-if="session.remote">
            <div class="comparison"><div><h4>本地草稿</h4><pre>{{ JSON.stringify(session.draft, null, 2) }}</pre></div><div><h4>远端修订 {{ session.remote.head_revision }}</h4><pre>{{ JSON.stringify(session.remote.chart, null, 2) }}</pre></div></div>
            <button @click="session.useRemote()">明确放弃草稿，使用远端版本</button>
            <label>人工合并后的谱面 JSON <textarea v-model="merged" rows="8" /></label>
            <button @click="merge">采用合并草稿，等待保存</button>
            <p v-if="parseError" role="alert">{{ parseError }}</p>
          </template>
        </section>
      </template>
    </fieldset>
  </section>
</template>
<style scoped>
fieldset { border: 0; padding: 0; } form { margin: 1rem 0; } input, textarea, button { font: inherit; padding: .5rem; } button { cursor: pointer; margin: .3rem; } label { display: block; } textarea { display: block; width: 90%; } .comparison { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; } pre { overflow: auto; max-height: 20rem; } [role=alert] { color: #ffb4ac; }
</style>
