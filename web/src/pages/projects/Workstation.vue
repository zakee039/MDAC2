<script setup lang="ts">
import { ref, shallowRef, onBeforeUnmount, watch } from 'vue'
import { ManualApi, fraction, viewTiming, storedTiming, numberOf, multiply } from '../../services/manual.ts'
import { Transport } from '../../features/audio/Transport.ts'
import TransportView from '../../features/audio/Transport.vue'
import Waveform from '../../features/audio/Waveform.vue'
import TimingView from '../../features/timing/TimingEditor.vue'
import AnalysisPanel from '../../features/timing/AnalysisPanel.vue'
import OnsetPanel from '../../features/onsets/OnsetPanel.vue'
import StructurePanel from '../../features/similarity/StructurePanel.vue'
import { secondsAt } from '../../features/timing/TimingEditor.ts'
import type { Timing } from '../../features/timing/TimingEditor.ts'
import ChartView from '../../features/chart/ChartEditor.vue'
import type { EditableChart } from '../../features/chart/ChartEditor.ts'
const props=defineProps<{projectId:string;token:string}>()
const emit=defineEmits<{saved:[];dirty:[value:boolean]}>()
const api=new ManualApi(()=>props.token)
const busy=ref(false),error=ref(''),asset=ref<any>(null),stored=ref<any>(null),chart=shallowRef<EditableChart|null>(null),timing=shallowRef<Timing|null>(null),base=ref<string|null>(null)
const dirty=ref(false),bpm=ref(120),transport=shallowRef<Transport|null>(null),peaks=ref<number[][]>([]),position=ref(0),migration=ref<any>(null),policy=ref('keep_audio_time'),job=ref<any>(null),exportId=ref(''),analysisBusy=ref(false)
const onsetBusy=ref(false),structureBusy=ref(false)
let media:HTMLAudioElement|null=null,objectUrl='',unsubscribe=()=>{},generation=0
const path=()=>`projects/${encodeURIComponent(props.projectId)}`
watch(dirty,v=>emit('dirty',v))
function release(){unsubscribe();transport.value?.destroy();transport.value=null;media=null;if(objectUrl)URL.revokeObjectURL(objectUrl);objectUrl=''}
async function act(fn:()=>Promise<void>){if(busy.value)return;busy.value=true;error.value='';try{await fn()}catch(e){error.value=(e as Error).message}finally{busy.value=false}}
async function load(){await act(async()=>{
  const epoch=++generation;release();asset.value=null;chart.value=null;timing.value=null;migration.value=null;exportId.value='';job.value=null
  const snapshot=await api.json(path()+'/manual');if(epoch!==generation)return
  stored.value=snapshot.timing;chart.value=snapshot.chart;base.value=snapshot.chart?.revision_id??null
  if(stored.value)timing.value=viewTiming(stored.value)
  try{asset.value=await api.json(path()+'/audio')}catch(e){if((e as any).status===404){dirty.value=false;return}throw e}
  const wave=await api.json(path()+'/peaks');peaks.value=wave.peaks
  const response=await api.request(asset.value.audio_url.replace('/api/v1/',''));const blob=await response.blob();if(epoch!==generation)return
  objectUrl=URL.createObjectURL(blob);media=new Audio(objectUrl);media.preload='auto'
  transport.value=new Transport(media);unsubscribe=transport.value.subscribe(s=>position.value=s.position);dirty.value=false
})}
watch(()=>props.projectId,()=>{void load()},{immediate:true})
onBeforeUnmount(()=>{generation++;release()})
async function upload(event:Event){const file=(event.target as HTMLInputElement).files?.[0];if(!file)return
  await act(async()=>{await api.request(path()+'/audio',file,true)});if(!error.value)await load()
}
function initialize(){
  if(!asset.value||!Number.isFinite(bpm.value)||bpm.value<=0){error.value='请输入正 BPM。';return}
  const duration=asset.value.canonical.sample_count/asset.value.canonical.sample_rate,id=crypto.randomUUID()
  stored.value={revision_id:id,review:'pending',parent_fingerprint:null,downbeat_q:['0'],mapping:{schema_version:1,anchors:[{q:'0',t:'0'},{q:fraction(duration*bpm.value/60),t:fraction(duration)}],audio_duration:fraction(duration),pulse_unit:'1/4',bar_origin:'0',extrapolation_seconds:'0',meter_segments:[{start_q:'0',numerator:4,denominator:4}]}}
  // Audio duration is the exact sample ratio, including non-terminating decimal lengths.
  const reduce=(n:number,d:number)=>{let a=n,b=d;while(b){const r=a%b;a=b;b=r}return `${n/a}${d/a===1?'':'/'+d/a}`}
  stored.value.mapping.audio_duration=reduce(asset.value.canonical.sample_count,asset.value.canonical.sample_rate)
  stored.value.mapping.anchors[1].t=stored.value.mapping.audio_duration
  stored.value.mapping.anchors[1].q=multiply(stored.value.mapping.audio_duration,fraction(bpm.value),60)
  timing.value=viewTiming(stored.value)
  chart.value={revision_id:crypto.randomUUID(),timing_revision:id,events:[],onset_revision:null,arrangement_policy_hash:null,semantics_policy_hash:null,user_patch_ids:[],schema_version:1};dirty.value=true
}
async function persist(nextChart:EditableChart,nextTiming:any,confirmation:any=null){
  const result=await api.json(path()+'/manual',{expected_revision:base.value,command_id:crypto.randomUUID(),chart:{...nextChart,revision_id:crypto.randomUUID()},timing:nextTiming,position_policy:confirmation?policy.value:null,preview_fingerprint:confirmation?.fingerprint??null})
  chart.value=result.chart;stored.value=result.timing;timing.value=viewTiming(result.timing);base.value=result.chart.revision_id;dirty.value=false;migration.value=null;exportId.value='';job.value=null;emit('saved')
}
async function changeTiming(view:Timing){
  if(!chart.value||!stored.value)return
  const next=storedTiming(view,stored.value)
  exportId.value='';job.value=null
  if(!base.value || !chart.value.events.length){stored.value=next;timing.value=view;chart.value={...chart.value,timing_revision:view.id};dirty.value=true;return}
  if(dirty.value){error.value='请先保存谱面，再预览时间迁移。';timing.value=viewTiming(stored.value);return}
  await act(async()=>{const preview=await api.json(path()+'/manual/preview',{expected_revision:base.value,timing:next,position_policy:policy.value});migration.value={...preview,timing:next}})
}
function time(q:string){if(!timing.value)throw new Error('缺少时基');return secondsAt(timing.value,numberOf(q))}
function appliedAnalysis(result:any){chart.value=result.chart;stored.value=result.timing;timing.value=viewTiming(result.timing);base.value=result.chart.revision_id;dirty.value=false;migration.value=null;exportId.value='';job.value=null;emit('saved')}
async function listen(start:number,end:number){await act(async()=>{transport.value!.setLoop([start,end]);transport.value!.seek(start);await transport.value!.play()})}
async function exportChart(){await act(async()=>{job.value=await api.json(path()+'/exports',{expected_revision:base.value});exportId.value=''})}
async function refreshJob(){if(!job.value)return;await act(async()=>{const id=job.value.job_id;job.value={...await api.json('jobs/'+id),job_id:id};if(job.value.state==='succeeded'){const result=await api.json('jobs/'+id+'/export');if(result.manifest.chart_revision===base.value && !dirty.value && !migration.value)exportId.value=result.export_id}})}
async function download(){await act(async()=>{const response=await api.request('exports/'+exportId.value+'/download');const url=URL.createObjectURL(await response.blob()),a=document.createElement('a');a.href=url;a.download='chart.mdm';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)})}
</script>
<template>
  <section aria-label="手工制谱工作台">
    <h2>制谱工作台</h2><p v-if="error" role="alert">{{error}}</p><p v-if="busy" role="status">正在处理…</p>
    <AnalysisPanel v-if="asset && transport" :key="projectId+':'+base" :api="api" :project-id="projectId" :base="base" :blocked="busy || onsetBusy || structureBusy || dirty || !!migration" @applied="appliedAnalysis" @listen="listen" @stop-listen="transport.setLoop(null);transport.pause()" @busy="analysisBusy=$event" />
    <OnsetPanel v-if="asset && transport && base && stored?.review==='accepted'" :key="'onsets:'+projectId+':'+base" :api="api" :project-id="projectId" :base="base" :duration="asset.canonical.sample_count/asset.canonical.sample_rate" :blocked="busy || analysisBusy || structureBusy || dirty || !!migration" @applied="appliedAnalysis" @listen="listen" @stop-listen="transport.setLoop(null);transport.pause()" @busy="onsetBusy=$event" />
    <p v-else-if="timing">采音候选需要先确认并保存时间轴。</p>
    <StructurePanel v-if="asset && transport && base && timing && stored?.review==='accepted'" :key="'structure:'+projectId+':'+base" :api="api" :project-id="projectId" :base="base" :timing="timing" :blocked="busy || analysisBusy || onsetBusy || dirty || !!migration" @applied="appliedAnalysis" @listen="listen" @stop-listen="transport.setLoop(null);transport.pause()" @busy="structureBusy=$event" />
    <fieldset :disabled="busy || analysisBusy || onsetBusy || structureBusy || !token">
      <button @click="load">重新打开（放弃本地工作台草稿）</button>
      <label v-if="!asset">导入音频 <input type="file" accept="audio/*" @change="upload" /></label>
      <template v-if="asset && !timing"><label>初始四分拍 BPM <input v-model.number="bpm" type="number" min="1" /></label><button @click="initialize">建立手工时间轴</button></template>
      <template v-if="transport"><TransportView :key="projectId" :transport="transport" /><Waveform :transport="transport" :peaks="peaks" :duration="asset.canonical.sample_count/asset.canonical.sample_rate" :regions="[]" /></template>
      <template v-if="timing && chart && transport">
        <label>修改时间轴时 <select v-model="policy" :disabled="!!migration"><option value="keep_audio_time">保持音频位置（秒）</option><option value="keep_musical_position">保持音乐位置（拍）</option></select></label>
        <TimingView :key="timing.id" :timing="timing" :position="position" @change="changeTiming" @seek="transport.seek($event)" />
        <section v-if="migration"><h3>时间迁移影响（头尾均列出）</h3><pre>{{migration.diffs}}</pre><pre>{{migration.conflicts}}</pre>
          <button :disabled="migration.conflicts.length>0" @click="act(()=>persist(migration.chart,migration.timing,migration))">确认并保存迁移</button><button @click="migration=null; timing=viewTiming(stored)">取消迁移</button>
        </section>
        <ChartView v-else :chart="chart" :timing-id="timing.id" :time="time" :transport="transport" @change="chart=$event;dirty=true;exportId='';job=null" />
        <button :disabled="!dirty || !!migration" @click="act(()=>persist(chart!,stored))">保存当前谱面</button>
        <p>{{dirty?'有未保存修改':'已保存'}} · 游戏加载尚待实际验收；导出不支持的语义会明确失败。</p>
        <button :disabled="dirty || !base || !!migration || !chart.events.length" @click="exportChart">构建 MDM 资源包</button>
        <div v-if="job"><p>导出作业：{{job.state}}</p><button @click="refreshJob">刷新作业状态</button><button v-if="['queued','running'].includes(job.state)" @click="act(async()=>{await api.json('jobs/'+job.job_id+'/cancel',{run_id:job.run_id})})">取消作业</button></div>
        <button v-if="exportId && !dirty && !migration" @click="download">下载已校验资源包</button>
      </template>
    </fieldset>
  </section>
</template>
<style scoped>
section { margin-top: 1.5rem; } fieldset {border:0;padding:0} label {display:block;margin:.5rem 0} button,input,select{font:inherit;padding:.35rem;margin:.2rem} pre{max-height:20rem;overflow:auto}[role=alert]{color:#ffb4ac}
</style>
