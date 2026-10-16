<script setup lang="ts">
import {computed,onBeforeUnmount,onMounted,reactive,ref,watch} from 'vue'
import {StructureSession} from './StructureSession.ts'
import SimilarityPanel from './SimilarityPanel.vue'
import type {JsonApi} from '../timing/AnalysisSession.ts'
import type {Timing} from '../timing/TimingEditor.ts'
import type {Proposal} from './SimilaritySession.ts'
import {numberOf} from '../../services/manual.ts'
const props=defineProps<{api:JsonApi;projectId:string;base:string;blocked:boolean;timing:Timing}>()
const emit=defineEmits<{applied:[result:any];listen:[start:number,end:number];stopListen:[];busy:[value:boolean]}>()
const session=reactive(new StructureSession(props.api,props.projectId,props.base))
const busy=ref(false),error=ref(''),choice=ref(0),mode=ref('arrangement'),sourceHold=ref('reject'),targetHold=ref('reject'),locked=ref('reject'),confirmFull=ref(false)
watch(()=>props.blocked,value=>session.setBlocked(value),{immediate:true})
watch([mode,sourceHold,targetHold,locked],()=>{session.invalidateDiff();confirmFull.value=false})
const selected=computed<Proposal[]>(()=>{
 const p=session.candidate?.proposals[choice.value];if(!p)return []
 return [{id:p.proposal_id,fingerprint:p.fingerprint,timingFingerprint:p.timing_fingerprint,
  source:{start:numberOf(p.source.start_q),end:numberOf(p.source.end_q)},target:{start:numberOf(p.target.start_q),end:numberOf(p.target.end_q)},
  scores:Object.fromEntries(p.similarity_by_source),warnings:[...p.warnings,...p.alignment.reasons]}]
})
const permitted=computed(()=>!!session.review&&session.review.group.targets.every((t:any)=>t.allowed_modes.includes(mode.value)))
const alignmentFailed=computed(()=>session.review&&Object.keys(session.review.alignment_errors).length>0)
async function act(fn:()=>Promise<void>){if(busy.value)return;busy.value=true;emit('busy',true);error.value='';try{await fn()}catch(e){error.value=(e as Error).message}finally{busy.value=false;emit('busy',false)}}
function options(){return {target_ids:session.review.group.targets.map((t:any)=>t.target_id),mode:mode.value,source_hold:mode.value==='full_pattern'?sourceHold.value:'reject',target_hold:mode.value==='full_pattern'?targetHold.value:'reject',locked:mode.value==='full_pattern'?locked.value:'reject'}}
function choose(){session.invalidateReview();confirmFull.value=false;emit('stopListen')}
onMounted(()=>void act(()=>session.history()))
onBeforeUnmount(()=>{session.dispose();emit('busy',false)})
</script>
<template>
 <section aria-label="相似段工作流">
  <h3>结构与复用 · 混音频谱基线</h3><p>检测结果仅作建议。先听检并确认关系，再预览谱面变化；不会自动复制音符。</p>
  <p v-if="error" role="alert">{{error}}</p>
  <fieldset :disabled="busy || blocked">
   <p v-if="!session.available">本地分析环境暂不可用。</p>
   <button :disabled="!session.available || ['queued','running'].includes(session.job?.state)" @click="act(async()=>{choice=0;await session.start()})">检测相似片段</button>
   <button v-if="session.operation?.operation_id" @click="act(async()=>{const result=await session.undo();if(result)emit('applied',result)})">撤销上次结构复用</button>
   <div v-if="session.job"><p>结构作业：{{session.job.state}} {{session.job.error_code??''}}</p><button @click="act(()=>session.refresh())">刷新结构状态</button><button v-if="['queued','running'].includes(session.job.state)" @click="act(()=>session.cancel())">取消结构分析</button></div>
   <template v-if="session.candidate">
    <p v-if="!session.current">旧版本检测只供查看；请基于当前谱面重新检测。</p>
    <p v-if="!session.candidate.proposals.length">本次没有符合基线条件的相似段，不补造关系。</p>
    <template v-else>
     <label>候选段对 <select v-model.number="choice" @change="choose"><option v-for="(p,i) in session.candidate.proposals" :key="p.proposal_id" :value="i">{{Number(i)+1}}：{{p.source.start_q}}–{{p.source.end_q}} 拍 → {{p.target.start_q}}–{{p.target.end_q}} 拍</option></select></label>
     <fieldset :disabled="!session.current">
      <SimilarityPanel :key="session.candidate.candidate_id+':'+choice" :timing="timing" :timing-fingerprint="session.candidate.timing_fingerprint" :proposals="selected" @changed="session.invalidateReview();confirmFull=false" @confirmed="value=>act(()=>session.confirm(value))" @listen="(a,b)=>emit('listen',a,b)" @stop-listen="emit('stopListen')" />
     </fieldset>
     <section v-if="session.review">
      <p>关系确认已保存。{{alignmentFailed?'当前边界无法可靠对齐，仅保留标注，不能复用。':'仍需单独确认以下应用差异。'}}</p>
      <label>复用方式 <select v-model="mode"><option value="arrangement">仅编排参考（保留落点）</option><option value="mirror">镜像编排（保留落点）</option><option value="full_pattern">完整替换（会增删目标音符）</option></select></label>
      <p v-if="!permitted">此方式未获得目标段许可，请先修改并重新确认关系。</p>
      <template v-if="mode==='full_pattern'">
       <label>主段跨界长条 <select v-model="sourceHold"><option value="reject">冲突时停止</option><option value="skip">明确跳过跨界长条</option><option value="truncate">明确截断至边界</option></select></label>
       <label>目标跨界长条 <select v-model="targetHold"><option value="reject">冲突时停止</option><option value="preserve">保留（仍检查碰撞）</option><option value="truncate">明确保留边界外部分</option></select></label>
       <label>目标锁定 <select v-model="locked"><option value="reject">冲突时停止</option><option value="skip_target">明确跳过整个目标</option></select></label>
      </template>
      <button :disabled="!session.editable || !permitted || alignmentFailed" @click="act(()=>session.preview(options()))">预览结构复用差异</button>
      <section v-if="session.diff">
       <p>新增 {{session.diff.changes.filter((c:any)=>!c.before).length}}、删除 {{session.diff.changes.filter((c:any)=>!c.after).length}}、修改 {{session.diff.changes.filter((c:any)=>c.before&&c.after).length}}；冲突 {{session.diff.conflicts.length}}。</p>
       <ul><li v-for="c in session.diff.changes" :key="c.event_id">{{c.before?c.before.q+' 拍 / '+c.before.lane_mask.join('、'):'新增'}} → {{c.after?c.after.q+' 拍 / '+c.after.lane_mask.join('、'):'移除'}}</li></ul>
       <details v-if="session.diff.conflicts.length"><summary>冲突原因</summary><pre>{{session.diff.conflicts}}</pre></details>
       <label v-if="mode==='full_pattern'"><input v-model="confirmFull" type="checkbox" />我已检查目标音符增删，确认完整替换</label>
       <button :disabled="!session.editable || session.diff.conflicts.length>0 || mode==='full_pattern'&&!confirmFull" @click="act(async()=>{const result=await session.apply(confirmFull);if(result)emit('applied',result)})">应用结构复用并保存</button>
       <button @click="session.invalidateDiff()">取消结构预览</button>
      </section>
     </section>
    </template>
   </template>
  </fieldset>
 </section>
</template>
<style scoped>fieldset{border:0;padding:0}label{display:block;margin:.4rem 0}button,input,select{font:inherit;padding:.3rem;margin:.2rem}pre,ul{max-height:18rem;overflow:auto}[role=alert]{color:#ffb4ac}</style>
