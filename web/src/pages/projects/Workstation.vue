<script setup lang="ts">
import { computed, ref, shallowRef, onBeforeUnmount, watch } from 'vue'
import StudioIcon from '../../components/StudioIcon.vue'
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
const emit=defineEmits<{saved:[];dirty:[value:boolean];blocked:[value:boolean]}>()
const api=new ManualApi(()=>props.token)
const busy=ref(false),error=ref(''),asset=ref<any>(null),stored=ref<any>(null),chart=shallowRef<EditableChart|null>(null),timing=shallowRef<Timing|null>(null),base=ref<string|null>(null)
const dirty=ref(false),bpm=ref(120),transport=shallowRef<Transport|null>(null),peaks=ref<number[][]>([]),position=ref(0),migration=ref<any>(null),policy=ref('keep_audio_time'),job=ref<any>(null),exportId=ref(''),analysisBusy=ref(false)
const onsetBusy=ref(false),structureBusy=ref(false)
const stage=ref(1)
const fileInput=ref<HTMLInputElement|null>(null)
const fileName=ref('')
const stageNames=['导入音频','时间与信息','采音候选','轨道编排','音符语义','结构与复用','资源导出']
const unavailable=computed(()=>busy.value || analysisBusy.value || onsetBusy.value || structureBusy.value || !props.token)
function available(index:number) {
  if(index===1)return true
  if(index===2)return !!asset.value && !!transport.value
  if(index===3 || index===6)return !!base.value && stored.value?.review==='accepted' && !!transport.value
  return !!timing.value && !!chart.value && !!transport.value
}
async function receiveFile(file?:File) {
  if(!file || unavailable.value || asset.value)return
  const accepted=file.type.startsWith('audio/') || /\.(wav|mp3|flac|ogg|m4a|aac|webm|wma)$/i.test(file.name)
  if(!accepted){error.value='请选择音频文件，例如 WAV、MP3、FLAC 或 OGG。';return}
  await act(async()=>{await api.request(path()+'/audio',file,true)})
  if(!error.value){fileName.value=file.name;await load();if(asset.value)stage.value=2}
}
function dropped(event:DragEvent){void receiveFile(event.dataTransfer?.files[0])}
let media:HTMLAudioElement|null=null,objectUrl='',unsubscribe=()=>{},generation=0
const path=()=>`projects/${encodeURIComponent(props.projectId)}`
watch(dirty,v=>emit('dirty',v))
watch(()=>busy.value || analysisBusy.value || onsetBusy.value || structureBusy.value || !!migration.value, value=>emit('blocked',value), {immediate:true})
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
async function upload(event:Event){const input=event.target as HTMLInputElement;const file=input.files?.[0];input.value='';await receiveFile(file)}
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
  <section id="workflow" class="workstation panel" aria-label="手工制谱工作台">
    <div class="workstation-heading section-title"><div><span class="eyebrow">CHART WORKSPACE</span><h2>让每一拍，恰到好处。</h2></div><span class="work-status">{{dirty ? '草稿未保存' : busy ? '正在处理' : '工作台就绪'}}</span></div>
    <nav class="stage-nav" aria-label="制谱阶段"><button v-for="(name,index) in stageNames" :key="name" :class="{selected:stage===index+1}" :disabled="!available(index+1)" :aria-current="stage===index+1?'step':undefined" @click="stage=index+1"><span>0{{index+1}}</span>{{name}}</button></nav>
    <p v-if="error" role="alert">{{error}}</p><p v-if="busy" role="status">正在处理，请稍候…</p>
    <div class="editor-surface">
      <fieldset :disabled="unavailable">
        <div v-show="stage===1" class="stage-content">
          <div class="section-title"><div><h3>一切，从喜欢的音乐开始。</h3><p>导入后会生成音频波形，再进入时间轴确认。</p></div></div>
          <template v-if="!asset"><input ref="fileInput" type="file" accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.aac,.webm,.wma" class="hidden-file" tabindex="-1" aria-hidden="true" @change="upload" /><button class="audio-drop" @click="fileInput?.click()" @dragover.prevent @drop.prevent="dropped"><span class="drop-symbol"><StudioIcon name="upload" :size="30" /></span><strong>把你的音乐拖到这里</strong><span>或点击选择本地音频文件</span><small>MP3 / WAV / FLAC / OGG</small></button></template>
          <div v-else class="audio-ready"><StudioIcon name="wave" :size="28" /><div><h3>{{fileName || '项目音频已载入'}}</h3><p>{{(asset.canonical.sample_count/asset.canonical.sample_rate).toFixed(1)}} 秒 · {{(asset.canonical.sample_rate/1000).toFixed(1)}} kHz</p></div><button class="primary" @click="stage=2">查看时间轴<StudioIcon name="arrow" :size="16" /></button></div>
          <details class="reload-options"><summary>重新读取与草稿</summary><p>重新读取会放弃尚未保存的工作台修改，恢复服务端已保存的版本。</p><button @click="load">重新打开（放弃本地工作台草稿）</button></details>
        </div>
        <div v-if="transport" class="player-panel"><TransportView :key="projectId" :transport="transport" /><Waveform :transport="transport" :peaks="peaks" :duration="asset.canonical.sample_count/asset.canonical.sample_rate" :regions="[]" /></div>
        <div v-show="stage===2" class="stage-content">
          <div v-if="asset && !timing" class="initialize-panel"><h3>建立初始时间轴</h3><p>输入四分拍 BPM 创建手工时间轴，也可在下方获取自动分析候选。</p><label>初始四分拍 BPM <input v-model.number="bpm" type="number" min="1" /></label><button class="primary" @click="initialize">建立手工时间轴</button></div>
          <template v-if="timing && chart && transport"><label>修改时间轴时 <select v-model="policy" :disabled="!!migration"><option value="keep_audio_time">保持音频位置（秒）</option><option value="keep_musical_position">保持音乐位置（拍）</option></select></label><TimingView :key="timing.id" :timing="timing" :position="position" @change="changeTiming" @seek="transport.seek($event)" /></template>
        </div>
        <section v-if="migration" class="migration-panel"><h3>时间迁移影响（头尾均列出）</h3><pre>{{migration.diffs}}</pre><pre>{{migration.conflicts}}</pre><button class="primary" :disabled="migration.conflicts.length>0" @click="act(()=>persist(migration.chart,migration.timing,migration))">确认并保存迁移</button><button @click="migration=null; timing=viewTiming(stored)">取消迁移</button></section>
        <div v-show="stage===4 || stage===5" class="stage-content"><ChartView v-if="timing && chart && transport && !migration" :chart="chart" :timing-id="timing.id" :time="time" :transport="transport" @change="chart=$event;dirty=true;exportId='';job=null" /></div>
        <div v-show="stage===7" class="stage-content export-panel"><div class="export-symbol"><StudioIcon name="layers" :size="30" /></div><h3>把这段节奏，带进游戏。</h3><p>基于当前已保存的谱面构建 MDM 资源包。构建完成并通过校验后即可下载。</p><p v-if="!chart?.events.length">当前为空谱，请先在轨道或音符编辑器中添加音符。</p><button class="primary" :disabled="dirty || !base || !!migration || !chart?.events.length" @click="exportChart">构建 MDM 资源包<StudioIcon name="arrow" :size="16" /></button><div v-if="job"><p>导出作业：{{job.state}}</p><button @click="refreshJob">刷新作业状态</button><button v-if="['queued','running'].includes(job.state)" @click="act(async()=>{await api.json('jobs/'+job.job_id+'/cancel',{run_id:job.run_id})})">取消作业</button></div><button v-if="exportId && !dirty && !migration" class="primary" @click="download">下载已校验资源包</button><p class="export-note">游戏加载尚待实际验收；导出不支持的语义会明确失败。</p></div>
      </fieldset>
      <div v-show="stage===2" class="analysis-area"><AnalysisPanel v-if="asset && transport" :key="projectId+':'+base" :api="api" :project-id="projectId" :base="base" :blocked="busy || onsetBusy || structureBusy || dirty || !!migration || !token" @applied="appliedAnalysis" @listen="listen" @stop-listen="transport.setLoop(null);transport.pause()" @busy="analysisBusy=$event" /><p v-if="timing && stored?.review!=='accepted'" class="review-hint">采音与结构复用需要先确认并保存时间轴。</p></div>
      <div v-show="stage===3"><OnsetPanel v-if="asset && transport && base && stored?.review==='accepted'" :key="'onsets:'+projectId+':'+base" :api="api" :project-id="projectId" :base="base" :duration="asset.canonical.sample_count/asset.canonical.sample_rate" :blocked="busy || analysisBusy || structureBusy || dirty || !!migration || !token" @applied="appliedAnalysis" @listen="listen" @stop-listen="transport.setLoop(null);transport.pause()" @busy="onsetBusy=$event" /></div>
      <div v-show="stage===6"><StructurePanel v-if="asset && transport && base && timing && stored?.review==='accepted'" :key="'structure:'+projectId+':'+base" :api="api" :project-id="projectId" :base="base" :timing="timing" :blocked="busy || analysisBusy || onsetBusy || dirty || !!migration || !token" @applied="appliedAnalysis" @listen="listen" @stop-listen="transport.setLoop(null);transport.pause()" @busy="structureBusy=$event" /></div>
    </div>
    <div v-if="timing && chart && transport" class="save-bar"><span><i :class="{dirty}" />{{dirty?'有未保存修改':'当前谱面已保存'}}<small>{{chart.events.length}} 个音符事件</small></span><button class="primary" :disabled="unavailable || !dirty || !!migration" @click="act(()=>persist(chart!,stored))">保存当前谱面<StudioIcon name="check" :size="16" /></button></div>
  </section>
</template>
<style scoped>
.workstation { padding: 26px; }.workstation-heading { margin-bottom: 25px; }.workstation-heading h2 { margin-top: 9px; font-size: 20px; }.work-status { font-size: 10px; padding: 7px 10px; border-radius: 5px; border: 1px solid #dce5db; color: #526859; background: #f3f7f0; }.stage-nav { display: flex; gap: 7px; overflow-x: auto; margin-bottom: 27px; padding-bottom: 5px; }.stage-nav button { flex: 1; min-width: 92px; padding: 13px 8px; font-size: 11px; border-color: #dce5db; background: #f3f7f0; flex-direction: column; gap: 8px; }.stage-nav button>span { font: 10px var(--mono); color: #61756a; }.stage-nav button.selected { color: var(--accent); background: #eaf3e4; border-color: #dce5db; }.stage-nav button.selected>span { color: var(--accent); }.stage-content { padding-bottom: 14px; }.section-title h3 { margin-top: 0; }.section-title>button { font-size: 11px; }.hidden-file { position: absolute; width: 1px; height: 1px; opacity: 0; pointer-events: none; }.editor-surface .audio-drop { display: flex; flex-direction: column; gap: 12px; width: 100%; min-height: 240px; margin: 12px 0; border: 1px dashed #dce5db; background: #f3f7f0; border-radius: 12px; }.audio-drop:hover { background: #f3f7f0; }.drop-symbol { display: grid; place-items: center; width: 58px; height: 58px; background: #eaf3e4; border: 1px solid #dce5db; border-radius: 15px; color: var(--accent); margin-bottom: 5px; }.audio-drop strong { font-size: 16px; font-weight: 500; }.audio-drop>span:not(.drop-symbol) { color: #61756a; font-size: 12px; }.audio-drop small { color: #61756a; font: 9px var(--mono); letter-spacing: 2px; margin-top: 12px; }.audio-ready { display: flex; align-items: center; gap: 16px; padding: 23px; border-radius: 10px; border: 1px solid #dce5db; background: #f3f7f0; color: var(--accent); }.audio-ready h3 { margin: 0; overflow-wrap: anywhere; }.audio-ready p { margin: 6px 0 0; font: 11px var(--mono); }.audio-ready button { margin-left: auto; }.player-panel { margin: 0 0 22px; border-bottom: 1px solid var(--border); }.initialize-panel { padding: 22px; background: #f3f7f0; border: 1px solid #dce5db; border-radius: 10px; }.analysis-area:has(section) { padding-top: 24px; border-top: 1px solid var(--border); }.review-hint { padding: 12px 16px; border: 1px solid #ead9ad; border-radius: 8px; background: #fff9e9; color: #866d2e; }.migration-panel { padding: 18px; border: 1px solid #dce5db; border-radius: 10px; }.export-panel { text-align: center; padding: 30px 15px; }.export-symbol { margin: 0 auto 18px; display: grid; place-items: center; width: 64px; height: 64px; border: 1px solid #dce5db; border-radius: 17px; background: #f3f7f0; color: var(--accent); }.export-panel h3 { font-size: 21px; }.export-note { font-size: 11px; margin-top: 23px; }.save-bar { display: flex; justify-content: space-between; align-items: center; gap: 16px; border-top: 1px solid var(--border); margin-top: 20px; padding-top: 20px; }.save-bar>span { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; color: #526859; font-size: 12px; }.save-bar i { width: 6px; height: 6px; border-radius: 50%; background: #a8caa4; }.save-bar i.dirty { background: #d9ecce; }.save-bar small { color: #61756a; margin-left: 8px; font-size: 10px; }
@media(max-width:680px) { .workstation { padding: 20px 16px; }.stage-nav button { min-width: 78px; font-size: 10px; }.audio-ready { flex-wrap: wrap; padding: 18px; }.audio-ready button { width: 100%; margin-left: 0; }.save-bar { flex-wrap: wrap; }.save-bar button { margin-left: auto; }.workstation-heading h2 { font-size: 18px; }.work-status { font-size: 9px; } }
</style>
