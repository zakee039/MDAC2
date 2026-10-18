<script setup lang="ts">
import {computed,onBeforeUnmount,onMounted,reactive,ref,watch} from 'vue'
import {OnsetSession} from './OnsetSession.ts'
import type {JsonApi} from '../timing/AnalysisSession.ts'
import {numberOf} from '../../services/manual.ts'
const props=defineProps<{api:JsonApi;projectId:string;base:string;blocked:boolean;duration:number}>()
const emit=defineEmits<{applied:[result:any];listen:[start:number,end:number];stopListen:[];busy:[value:boolean]}>()
const session=reactive(new OnsetSession(props.api,props.projectId,props.base))
const busy=ref(false),error=ref(''),difficulty=ref('normal'),reason=ref(''),page=ref(0)
watch(()=>props.blocked,value=>session.setBlocked(value),{immediate:true})
const candidates=computed(()=>session.candidate?.onsets.candidates??[])
const visible=computed(()=>candidates.value.slice(page.value*100,(page.value+1)*100))
const decisions=computed(()=>new Map<string,any>((session.candidate?.decisions??[]).map((d:any)=>[d.candidate_id,d])))
const labels:Record<string,string>={ACOUSTIC_SUPPORT_WITHIN_BUDGET:'强度和密度符合当前策略',NEEDS_MANUAL_QUANTIZATION:'偏离拍格，需手工定点',BELOW_SCORE_THRESHOLD:'强度低于阈值',GRID_COMPLEXITY_LIMIT:'细分超出难度',MINIMUM_GAP:'与其他落点过近',LOCAL_DENSITY_LIMIT:'局部密度超限'}
function explanation(id:string){return (decisions.value.get(id)?.reasons??[]).map((r:string)=>labels[r]??r).join('；')}
async function act(fn:()=>Promise<void>){if(busy.value)return;busy.value=true;emit('busy',true);error.value='';try{await fn()}catch(e){error.value=(e as Error).message}finally{busy.value=false;emit('busy',false)}}
function toggle(id:string,event:Event){try{session.toggle(id,(event.target as HTMLInputElement).checked)}catch(e){error.value=(e as Error).message}}
function clear(){try{session.select([])}catch(e){error.value=(e as Error).message}}
function listen(seconds:string){const t=numberOf(seconds);emit('listen',Math.max(0,t-.6),Math.min(props.duration,t+1.2))}
onMounted(()=>void act(()=>session.history()))
onBeforeUnmount(()=>{session.dispose();emit('busy',false)})
</script>
<template>
  <section aria-label="采音候选">
    <h3>采音 · 混音起音基线</h3>
    <p>先生成候选，再选择落点。现有人工修改和锁定会保留；确认后仍可在下方编辑器修订。</p>
    <p v-if="error" role="alert">{{error}}</p>
    <fieldset :disabled="busy || blocked">
      <label>难度 <select v-model="difficulty"><option value="easy">简单</option><option value="normal">普通</option><option value="hard">困难</option></select></label>
      <button :disabled="['queued','running'].includes(session.job?.state)" @click="act(async()=>{page=0;await session.start(difficulty)})">生成采音候选</button>
      <p v-if="blocked">请先保存或处理当前草稿和时间迁移。</p>
      <div v-if="session.job"><p>采音作业：{{session.job.state}} {{session.job.error_code??''}}</p>
        <button @click="act(()=>session.refresh())">刷新采音状态</button>
        <button v-if="['queued','running'].includes(session.job.state)" @click="act(()=>session.cancel())">取消采音</button>
      </div>
      <template v-if="session.candidate">
        <p v-if="!session.current">这是旧谱面版本的候选，仅供查看；重新生成后才能采纳。</p>
        <p>共 {{candidates.length}} 个候选，选中 {{session.selection.length}} 个。分数是规则强度，不代表音乐准确率；未量化的点请在编辑器手工添加。</p>
        <button :disabled="!session.editable" @click="clear">清空本次选留</button>
        <button @click="emit('stopListen')">结束循环听检</button>
        <table><thead><tr><th>选留</th><th>秒 / 拍格</th><th>强度 / 量化偏移</th><th>原始选留理由</th><th>听检</th></tr></thead>
          <tbody><tr v-for="c in visible" :key="c.candidate_id">
            <td><input type="checkbox" :aria-label="'选择 '+numberOf(c.seconds).toFixed(3)+' 秒候选'" :checked="session.selection.includes(c.candidate_id)" :disabled="!session.editable || c.quantized_q===null" @change="toggle(c.candidate_id,$event)" /></td>
            <td>{{numberOf(c.seconds).toFixed(3)}} 秒 / {{c.quantized_q??'待定'}} 拍</td>
            <td>{{c.score.toFixed(2)}} / {{c.quantization_error_seconds===null?'—':(numberOf(c.quantization_error_seconds)*1000).toFixed(1)+' ms'}}</td>
            <td>{{explanation(c.candidate_id)}}</td><td><button @click="listen(c.seconds)">听此处</button></td>
          </tr></tbody>
        </table>
        <button :disabled="page===0" @click="page--">上一页</button><span>第 {{page+1}} 页</span><button :disabled="(page+1)*100>=candidates.length" @click="page++">下一页</button>
        <button :disabled="!session.editable || session.selection.length>2048" @click="act(()=>session.preview())">预览采音差异</button>
        <section v-if="session.diff">
          <p>新增 {{session.diff.changes.filter((c:any)=>!c.before).length}}，移除 {{session.diff.changes.filter((c:any)=>!c.after).length}}，修改 {{session.diff.changes.filter((c:any)=>c.before&&c.after).length}}；保护 {{session.diff.preserved_ids.length}} 个已有事件。</p>
          <p v-if="session.diff.reasons.length">{{session.diff.reasons.length}} 个选点因人工修改、删除或锁定保护未重复加入。</p>
          <details><summary>逐音符差异</summary><ul><li v-for="c in session.diff.changes" :key="c.event_id">{{c.before?'原 '+c.before.q+' 拍':'新增'}} → {{c.after?c.after.q+' 拍 / '+c.after.lane_mask.join('、'):'移除'}}</li></ul></details>
          <label>检查结论与采纳理由 <input v-model="reason" maxlength="2000" /></label>
          <button :disabled="!session.editable || !reason.trim()" @click="act(async()=>{const result=await session.adopt(reason);if(result)emit('applied',result)})">确认写入新谱面版本</button>
          <button @click="session.diff=null">取消本次预览</button>
        </section>
      </template>
    </fieldset>
  </section>
</template>
