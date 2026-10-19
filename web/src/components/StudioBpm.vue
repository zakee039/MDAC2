<script setup lang="ts">
import {ref,computed,watch,onBeforeUnmount} from 'vue'
import {t} from '../studioI18n'
import {estimateTaps,type BeatSettings} from '../studioBpm'
import {DrumClock} from '../studioMetronome'
import {visibleSegments,type Trim} from '../studioTimeline'
import StudioTimeline from './StudioTimeline.vue'
const props=defineProps<{modelValue:BeatSettings;trim:Trim;duration:number;audioUrl:string;active:boolean;blocked:boolean;status:string;analysis:any}>()
const emit=defineEmits<{'update:modelValue':[value:BeatSettings];detect:[index:number];playing:[]}>()
const audio=ref<HTMLAudioElement>(),current=ref(0),taps=ref<Record<string,number[]>>({}),tapError=ref('')
const drumEnabled=ref(true),drumVolume=ref(1),musicVolume=ref(.65),drumError=ref(false)
const detectionIndex=ref<number|null>(null)
function detect(index:number){detectionIndex.value=index;emit('detect',index)}
const clipDuration=computed(()=>Math.max(0,props.trim.end_ms-props.trim.start_ms))
const sections=computed(()=>visibleSegments(props.modelValue,clipDuration.value))
const markers=computed(()=>sections.value.slice(1).map((s,i)=>({id:String(s.index),time:s.start_ms/1000,label:t('tempoPoint')+' '+(i+1)})))
const localCurrent=computed(()=>Math.max(0,Math.min(clipDuration.value/1000,current.value-props.trim.start_ms/1000)))
const isPlaying=ref(false)
function clock(seconds:number){const s=Math.max(0,seconds);return `${Math.floor(s/60)}:${(s%60).toFixed(1).padStart(4,'0')}`}
const clipTime=computed(()=>clock(current.value-props.trim.start_ms/1000)+' / '+clock((props.trim.end_ms-props.trim.start_ms)/1000))
function toggle(){if(!audio.value)return;if(audio.value.paused)void audio.value.play().catch(()=>{});else pause()}
let drum:DrumClock|undefined,playing=false,frame=0
function clockSettings(){const shift=props.trim.start_ms;return {...props.modelValue,offset_ms:props.modelValue.offset_ms+shift,segments:props.modelValue.segments?.map(s=>({...s,start_ms:s.start_ms+shift,offset_ms:s.offset_ms+shift})),volume:drumVolume.value,end_time:props.trim.end_ms/1000}}
function stopDrum(){playing=false;drum?.stop();cancelAnimationFrame(frame)}
function pause(){audio.value?.pause();stopDrum()}
function seek(time:number){if(audio.value)audio.value.currentTime=Math.max(props.trim.start_ms/1000,Math.min(props.trim.end_ms/1000,time));current.value=audio.value?.currentTime||0}
function seekLocal(time:number){seek(props.trim.start_ms/1000+time)}
function tick(){if(!audio.value)return;current.value=audio.value.currentTime;if(current.value<props.trim.start_ms/1000){seek(props.trim.start_ms/1000);current.value=props.trim.start_ms/1000}if(current.value>=props.trim.end_ms/1000){pause();seek(props.trim.end_ms/1000);return}frame=requestAnimationFrame(tick)}
async function prepareDrum(){
 emit('playing');if(audio.value&&(audio.value.currentTime<props.trim.start_ms/1000||audio.value.currentTime>=props.trim.end_ms/1000))seek(props.trim.start_ms/1000)
 if(!props.active||!audio.value||!drumEnabled.value)return
 try{drum??=new DrumClock(new AudioContext(),audio.value,clockSettings);await drum.unlock()}catch{drumError.value=true}
}
async function playDrum(){
 if(!props.active||!audio.value||audio.value.paused||audio.value.readyState<3)return
 playing=true;audio.value.volume=musicVolume.value;cancelAnimationFrame(frame);tick()
 if(!drumEnabled.value)return
 try{drum??=new DrumClock(new AudioContext(),audio.value,clockSettings);await drum.start();drumError.value=false}catch{drumError.value=true}
}
function resetDrum(){drum?.stop();if(playing)void playDrum()}
watch([()=>props.modelValue,()=>props.trim,drumEnabled,drumVolume],resetDrum,{deep:true})
watch(musicVolume,value=>{if(audio.value)audio.value.volume=value})
watch(()=>props.active,active=>{if(!active)pause()})
function set(index:number,key:string,value:number){if(!Number.isFinite(value))return;const config=props.modelValue;if(index<0)emit('update:modelValue',{...config,[key]:value});else emit('update:modelValue',{...config,segments:config.segments!.map((s,i)=>i===index?{...s,[key]:value}:s)})}
function number(index:number,key:string,e:Event,add=0){set(index,key,(e.target as HTMLInputElement).valueAsNumber+add)}
function reset(index?:number){if(index===undefined)taps.value={};else taps.value[index]=[];tapError.value=''}
function applyTaps(index:number){try{const s=sections.value.find(s=>s.index===index)!;const fit=estimateTaps(taps.value[index]||[],s.value.tap_latency_ms);const update={bpm:fit.bpm,offset_ms:fit.offset_ms-props.trim.start_ms};emit('update:modelValue',index<0?{...props.modelValue,...update}:{...props.modelValue,segments:props.modelValue.segments!.map((p,i)=>i===index?{...p,...update}:p)});tapError.value=''}catch{tapError.value=t('tapInvalid')}}
function tap(index:number){const s=sections.value.find(s=>s.index===index)!;if(!audio.value||audio.value.paused){tapError.value=t('tapPlayFirst');return}const time=audio.value.currentTime,local=time*1000-props.trim.start_ms;if(local<s.start_ms||local>=s.end_ms){tapError.value=t('tapSegment');return}drumEnabled.value=false;stopDrum();const list=taps.value[index]||[];if(list.length&&time-list.at(-1)!<.12)return;taps.value[index]=[...list,time].slice(-32);if(taps.value[index]!.length>=4)applyTaps(index)}
function add(){const ms=Math.round(current.value*1000)-props.trim.start_ms;if(ms<=0||ms>=props.trim.end_ms-props.trim.start_ms){tapError.value=t('seekInside');return}const points=props.modelValue.segments||[];if(points.length>=64||points.some(s=>Math.abs(s.start_ms-ms)<1))return;const s=sections.value.find(s=>ms>=s.start_ms&&ms<s.end_ms)!;emit('update:modelValue',{...props.modelValue,segments:[...points,{start_ms:ms,bpm:s.value.bpm,offset_ms:ms,tap_latency_ms:s.value.tap_latency_ms}].sort((a,b)=>a.start_ms-b.start_ms)});reset()}
function move(id:string,time:number){if(id==='in'||id==='out')return;const index=Number(id),points=[...(props.modelValue.segments||[])],s=points[index]!;const low=Math.max(1,(points[index-1]?.start_ms??-1)+1),high=Math.min(clipDuration.value-1,(points[index+1]?.start_ms??Infinity)-1);const ms=Math.max(low,Math.min(high,Math.round(time*1000)));points[index]={...s,start_ms:ms,offset_ms:s.offset_ms+ms-s.start_ms};emit('update:modelValue',{...props.modelValue,segments:points});reset()}
function remove(index:number){emit('update:modelValue',{...props.modelValue,segments:props.modelValue.segments?.filter((_,i)=>i!==index)});reset()}
watch(()=>props.audioUrl,()=>{pause();drum?.dispose();drum=undefined;reset()})
onBeforeUnmount(()=>{pause();drum?.dispose()});defineExpose({pause})
</script>
<template>
 <div class="bpm-module">
  <label class="enable"><input type="checkbox" :checked="modelValue.enabled" :disabled="blocked" @change="emit('update:modelValue',{...modelValue,enabled:($event.target as HTMLInputElement).checked})"/>{{t('bpmEnable')}}</label>
  <p class="hint">{{t('bpmHint')}}</p>
  <div class="toolbar"><button :disabled="blocked" @click="add">{{t('addTempo')}}</button><button @click="seek(trim.start_ms/1000)">{{t('jumpClipStart')}}</button><span>{{t('tempoHint')}}</span></div>
  <audio ref="audio" :key="audioUrl" :src="audioUrl" preload="metadata" @play="isPlaying=true;prepareDrum()" @playing="playDrum" @pause="isPlaying=false;stopDrum()" @waiting="stopDrum" @ended="isPlaying=false;stopDrum()" @seeking="stopDrum();reset()" @seeked="playDrum" @timeupdate="current=audio?.currentTime||0" @ratechange="resetDrum"/>
  <div class="clip-player"><button class="primary" :aria-label="t(isPlaying?'pauseClip':'playClip')" @click="toggle">{{isPlaying?'❚❚':'▶'}} {{t(isPlaying?'pauseClip':'playClip')}}</button><span>{{clipTime}}</span></div>
  <StudioTimeline :url="audioUrl.replace(/\/audio$/,'/waveform')" :duration="clipDuration/1000" :range-start="0" :range-end="clipDuration/1000" :source-start="trim.start_ms/1000" :source-end="trim.end_ms/1000" :current="localCurrent" :markers="markers" :blocked="blocked" :zoom-controls="true" @move="move" @seek="seekLocal"/>
  <div class="drum-controls">
   <label class="enable"><input v-model="drumEnabled" type="checkbox"/>{{t('drumEnabled')}}</label>
   <label>{{t('drumVolume')}} · {{Math.round(drumVolume*100)}}%<input v-model.number="drumVolume" type="range" min="0" max="1" step=".05"/></label>
   <label>{{t('musicVolume')}} · {{Math.round(musicVolume*100)}}%<input v-model.number="musicVolume" type="range" min="0" max="1" step=".05"/></label>
  </div>
  <p class="hint">{{t('drumHint')}}</p><p v-if="drumError" role="alert">{{t('drumUnavailable')}}</p>
  <p v-if="tapError" role="alert">{{tapError}}</p>
  <fieldset v-for="(s,n) in sections" :key="s.index" :disabled="blocked" class="segment">
   <div class="segment-heading"><h4>{{t('segment')}} {{n+1}}</h4><span>{{(s.start_ms/1000).toFixed(3)}} – {{(s.end_ms/1000).toFixed(3)}} s</span><button @click="seekLocal(s.start_ms/1000)">{{t('jumpSegment')}}</button><button v-if="n>0" @click="remove(s.index)">{{t('deleteTempo')}}</button></div>
   <div class="toolbar"><button @click="detect(s.index)">{{t('autoBpm')}}</button><span role="status">{{detectionIndex===s.index?status:''}}</span><button @click="tap(s.index)" @keydown.space.prevent="!$event.repeat&&tap(s.index)">{{t('tapBpm')}}</button><span>{{t('tapCount')}} {{taps[s.index]?.length||0}} / 4</span><button @click="reset(s.index)">{{t('resetTaps')}}</button></div>
   <div class="bpm-fields">
    <label>BPM<input type="number" :value="s.value.bpm" min="30" max="400" step="0.001" @input="number(s.index,'bpm',$event)"/><span><button :disabled="s.value.bpm/2<30" @click="set(s.index,'bpm',s.value.bpm/2)">÷ 2</button><button :disabled="s.value.bpm*2>400" @click="set(s.index,'bpm',s.value.bpm*2)">× 2</button></span></label>
    <label>{{t('beatOffset')}}<input type="number" :value="Math.round((s.value.offset_ms-s.start_ms)*1000)/1000" min="-600000" max="600000" step="1" @input="number(s.index,'offset_ms',$event,s.start_ms)"/><small>{{t('segmentOffsetHint')}}</small></label>
    <label>{{t('tapLatency')}}<input type="number" :value="s.value.tap_latency_ms" min="-1000" max="1000" step="1" @input="number(s.index,'tap_latency_ms',$event)"/><small>{{t('tapLatencyHint')}}</small><button v-if="(taps[s.index]?.length||0)>=4" @click="applyTaps(s.index)">{{t('reapplyTaps')}}</button></label>
   </div>
  </fieldset><p class="hint">{{t('tapHint')}}</p>
 </div>
</template>
<style scoped>
.drum-controls{display:flex;align-items:center;gap:24px;flex-wrap:wrap}.drum-controls label{min-width:140px}.drum-controls input[type=range]{width:100%;accent-color:#23856d}
.clip-player{display:flex;align-items:center;gap:16px;margin:12px 0 0;padding:10px 16px;border-radius:999px;background:#f0f5ef}.clip-player button{min-width:96px;border-radius:999px}.clip-player span{font-size:13px;color:#355b4c;font-variant-numeric:tabular-nums}
.enable{display:flex;align-items:center;gap:10px;font-weight:600}.hint,small{color:#61756a;font-size:12px;line-height:1.7}audio{width:100%;margin:10px 0 5px}.toolbar,.segment-heading{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:8px 0}.toolbar span,.segment-heading span{font-size:12px;color:#61756a}.segment-heading h4{margin:0}.segment-heading button:first-of-type{margin-left:auto}.bpm-fields{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px;margin-top:22px}label{display:flex;flex-direction:column;gap:8px;font-size:13px}.enable{flex-direction:row}input[type=number]{padding:11px;width:100%;border:1px solid #dce5db;border-radius:7px;background:white}label>span{display:flex;gap:8px}button{font-size:12px}fieldset{border:1px solid #dce5db;padding:20px;min-width:0;border-radius:10px;margin-top:18px}p[role=alert]{color:#bd453c}@media(max-width:650px){.bpm-fields{grid-template-columns:1fr}}
</style>
