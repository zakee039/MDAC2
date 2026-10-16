<script setup lang="ts">
import {ref,computed,watch} from 'vue'
import {SimilaritySession} from './SimilaritySession.ts'
import type {Proposal,Relation,ReuseMode,Span} from './SimilaritySession.ts'
import type {Timing} from '../timing/TimingEditor.ts'
const props=defineProps<{timing:Timing;timingFingerprint:string;proposals:Proposal[]}>()
const emit=defineEmits<{confirmed:[value:ReturnType<SimilaritySession['confirmation']>];listen:[start:number,end:number];stopListen:[];changed:[]}>()
const session=ref(new SimilaritySession(props.timing,props.timingFingerprint,props.proposals))
const error=ref(''),reason=ref(''),editId=ref('primary'),unit=ref('q'),start=ref(session.value.primary.start),end=ref(session.value.primary.end)
const labels:Record<Relation,string>={unknown:'尚未确定',exact_repeat:'近乎完全重复',variation:'变奏',rhythmic_similar:'节奏相似'}
const rows=computed(()=>[{id:'primary',span:session.value.primary},...session.value.targets])
watch(()=>props.timingFingerprint,f=>{session.value.invalidate(f);emit('stopListen')})
watch(()=>props.proposals,()=>{session.value.stale=true;emit('stopListen')})
function act(fn:()=>void){error.value='';try{fn()}catch(e){error.value=(e as Error).message}}
function select(id:string,span:Span){editId.value=id;unit.value='q';start.value=span.start;end.value=span.end}
function edit(){act(()=>{session.value.edit(editId.value,unit.value==='seconds'?session.value.fromSeconds(start.value,end.value):{start:start.value,end:end.value});emit('changed');emit('stopListen')})}
function changeUnit(){const row=rows.value.find(r=>r.id===editId.value)!;const values=unit.value==='seconds'?session.value.listen(row.id):[row.span.start,row.span.end];start.value=values[0]!;end.value=values[1]!}
function configure(id:string,enabled:boolean,relation:Relation,modes:ReuseMode[]){act(()=>{session.value.configure(id,enabled,relation,modes);emit('changed')})}
function listen(id:string){act(()=>{const [a,b]=session.value.listen(id);emit('listen',a,b)})}
function seconds(id:string){try{return session.value.listen(id).map(v=>v.toFixed(3)).join(' – ')}catch{return '已过期'}}
</script>
<template>
 <section aria-label="相似段听检与确认">
  <h3>相似片段</h3><p>频谱相似仅供听检。确认关系后，复用谱面仍需单独预览和操作。</p>
  <p v-if="error" role="alert">{{error}}</p><p v-if="session.stale" role="status">候选或时间轴已更新，请重新打开候选。</p>
  <fieldset :disabled="session.stale">
   <ol><li v-for="row in rows" :key="row.id" :class="row.id==='primary'?'primary':'target'">
    <strong>{{row.id==='primary'?'A · 主段':'B · 目标'}}</strong>：{{row.span.start}} – {{row.span.end}} 四分拍；{{seconds(row.id)}} 秒
    <button @click="listen(row.id)">循环听检 {{row.id==='primary'?'A':'B'}}</button><button @click="select(row.id,row.span)">修改边界</button>
   </li></ol>
   <button @click="emit('stopListen')">结束循环</button>
   <div><label>边界单位 <select v-model="unit" @change="act(changeUnit)"><option value="q">四分拍</option><option value="seconds">秒</option></select></label>
    <label>起点 <input v-model.number="start" type="number" step="any" /></label><label>终点 <input v-model.number="end" type="number" step="any" /></label>
    <button @click="edit">更新{{editId==='primary'?'主段':'目标'}}边界（需重新确认）</button>
   </div>
   <article v-for="(t,index) in session.targets" :key="t.id">
    <label><input type="checkbox" :checked="t.enabled" @change="configure(t.id,!t.enabled,t.relation,t.modes)" />启用目标 {{index+1}}</label>
    <label>关系 <select :value="t.relation" @change="configure(t.id,t.enabled,($event.target as HTMLSelectElement).value as Relation,[])"><option v-for="(label,key) in labels" :key="key" :value="key">{{label}}</option></select></label>
    <label v-for="mode in (['arrangement','mirror','full_pattern'] as ReuseMode[])" :key="mode"><input type="checkbox" :checked="t.modes.includes(mode)" :disabled="t.relation==='unknown'||mode==='full_pattern'&&t.relation!=='exact_repeat'" @change="configure(t.id,t.enabled,t.relation,t.modes.includes(mode)?t.modes.filter(m=>m!==mode):[...t.modes,mode])" />{{({arrangement:'允许编排参考',mirror:'允许镜像编排',full_pattern:'允许完整替换预览'})[mode]}}</label>
    <button :disabled="!t.enabled" @click="act(()=>session.confirm(t.id))">{{t.confirmed?'已确认此目标':'确认此目标关系'}}</button>
    <details><summary>检测证据与限制</summary><pre>{{session.proposals.find(p=>p.id===t.id)?.scores}}</pre><p>{{session.proposals.find(p=>p.id===t.id)?.warnings.join('；')}}</p></details>
   </article>
   <label>听检结论 <input v-model="reason" /></label><button @click="act(()=>emit('confirmed',session.confirmation('local-user',reason)))">保存关系确认</button>
  </fieldset>
 </section>
</template>
<style scoped>fieldset{border:0;padding:0}label{display:block;margin:.4rem 0}button,input,select{font:inherit;padding:.3rem;margin:.2rem}.primary{border-left:4px solid #68b8ef}.target{border-left:4px solid #e4b96b}li,article{padding:.5rem;margin:.5rem 0}[role=alert]{color:#ffb4ac}</style>
