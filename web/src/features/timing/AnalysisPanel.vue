<script setup lang="ts">
import {ref,reactive,onMounted,onBeforeUnmount,computed} from 'vue'
import {AnalysisSession,reviewWindows} from './AnalysisSession.ts'
import type {JsonApi} from './AnalysisSession.ts'
const props=defineProps<{api:JsonApi;projectId:string;base:string|null;blocked:boolean}>()
const emit=defineEmits<{applied:[result:any];listen:[start:number,end:number];stopListen:[];busy:[value:boolean]}>()
const session=reactive(new AnalysisSession(props.api,props.projectId,props.base))
const models=ref<any[]>([]),model=ref('final0'),meter=ref('4/4'),pulse=ref('1/4'),first=ref('0'),reason=ref(''),policy=ref('keep_audio_time'),error=ref(''),busy=ref(false)
const windows=computed(()=>session.candidate?reviewWindows(session.candidate):[])
async function act(fn:()=>Promise<void>){if(busy.value)return;busy.value=true;emit('busy',true);error.value='';try{await fn()}catch(e){error.value=(e as Error).message}finally{busy.value=false;emit('busy',false)}}
onMounted(()=>void act(async()=>{models.value=(await props.api.json('timing/models')).models;await session.history()}))
onBeforeUnmount(()=>{session.dispose();emit('busy',false)})
async function start(){if(props.blocked)return;await act(()=>session.start({model_id:model.value,pulse_unit:pulse.value,first_q:first.value,meter:meter.value.split('/').map(Number)}))}
async function adopt(){if(props.blocked)return;await act(async()=>emit('applied',await session.adopt(reason.value)))}
</script>
<template>
  <section aria-label="自动时间分析">
    <h3>自动时间分析 · CPU</h3>
    <p>结果先作为候选保留。拍号、脉冲单位和首拍位置由你选择，模型强拍仅供参考。</p>
    <p v-if="error" role="alert">{{error}}</p><p v-if="busy" role="status">正在读取或提交…</p>
    <fieldset :disabled="busy || blocked">
      <label>模型 <select v-model="model"><option v-for="m in models" :key="m.model_id" :value="m.model_id" :disabled="!m.available">{{m.model_id==='final0'?'主模型':'小模型'}}{{m.available?'':'（本地不可用）'}}</option></select></label>
      <label>拍号 <select v-model="meter"><option>4/4</option><option>3/4</option><option>6/8</option></select></label>
      <label>观测脉冲单位 <select v-model="pulse"><option>1/4</option><option>1/8</option><option>3/8</option></select></label>
      <label>首个观测的四分拍坐标（弱起可为负数）<input v-model="first" /></label>
      <button :disabled="!models.some(m=>m.model_id===model && m.available) || ['queued','running'].includes(session.job?.state)" @click="start">分析音频时间轴</button>
      <p v-if="blocked">请先保存或处理当前草稿和迁移。</p>
      <div v-if="session.job"><p>分析作业：{{session.job.state}} {{session.job.error_code??''}}</p><button @click="act(()=>session.refresh())">刷新分析状态</button><button v-if="['queued','running'].includes(session.job.state)" @click="act(()=>session.cancel())">取消分析</button></div>
      <template v-if="session.candidate">
        <p v-if="!session.current">这是旧谱面版本的分析，只供查看；请重新分析后采纳。</p>
        <p>拟合：{{session.candidate.fit.accepted?'通过（仍需听检）':session.candidate.fit.reason}}；p95 / 最大残差：{{(session.candidate.fit.p95_seconds*1000).toFixed(2)}} / {{(session.candidate.fit.max_seconds*1000).toFixed(2)}} ms</p>
        <p>观测覆盖：{{session.candidate.fit.coverage_seconds.join(' – ')}} 秒；覆盖之外不自动外推。</p>
        <p v-for="item in session.candidate.quality?.items??[]" :key="item.name">{{item.name}}：{{item.status}} {{item.value??''}}</p>
        <button v-for="w in windows" :key="w.label" @click="emit('listen',w.start,w.end)">循环听检：{{w.label}}</button>
        <button @click="emit('stopListen')">结束循环听检</button>
        <details><summary>原始节拍 / 强拍观测（秒）</summary><pre>{{session.candidate.observation.payload.beat_times}}</pre><pre>{{session.candidate.observation.payload.downbeat_times}}</pre></details>
        <label>已有音符 <select v-model="policy" @change="session.migration=null"><option value="keep_audio_time">保持秒位置</option><option value="keep_musical_position">保持拍位置</option></select></label>
        <button :disabled="!session.adoptable" @click="act(()=>session.preview(policy))">预览采纳影响</button>
        <div v-if="session.migration"><pre>{{session.migration.diffs}}</pre><pre>{{session.migration.conflicts}}</pre>
          <label>听检结论与采纳理由 <input v-model="reason" /></label>
          <button :disabled="!reason.trim() || session.migration.conflicts.length>0 || !session.adoptable" @click="adopt">确认采纳时间轴</button>
          <button @click="session.migration=null">取消本次采纳</button>
        </div>
      </template>
    </fieldset>
  </section>
</template>
