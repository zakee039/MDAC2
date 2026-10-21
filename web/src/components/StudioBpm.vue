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
const drumEnabled=ref(true),drumVolume=ref(1),musicVolume=ref(.65),drumError=ref(false),playbackError=ref(false)
const detectionIndex=ref<number|null>(null)
function detect(index:number){detectionIndex.value=index;emit('detect',index)}
const clipDuration=computed(()=>Math.max(0,props.trim.end_ms-props.trim.start_ms))
const sections=computed(()=>visibleSegments(props.modelValue,clipDuration.value))
const markers=computed(()=>sections.value.slice(1).map((s,i)=>({id:String(s.index),time:s.start_ms/1000,label:t('tempoPoint')+' '+(i+1)})))
const localCurrent=computed(()=>Math.max(0,Math.min(clipDuration.value/1000,current.value-props.trim.start_ms/1000)))
const isPlaying=ref(false)
let drum:DrumClock|undefined,frame=0,playEpoch=0,wantPlay=false
const pendingPlay=ref(false)
function unlockDrum(){
 if(!props.active||!audio.value||!drumEnabled.value)return Promise.resolve()
 try{drum??=new DrumClock(new AudioContext(),audio.value,clockSettings);return drum.unlock()}
 catch{return Promise.reject(new Error('audio unavailable'))}
}
async function toggle(){
 const media=audio.value
 if(!media||!props.active)return
 if(pendingPlay.value||!media.paused){pause();return}
 const epoch=++playEpoch;wantPlay=true;pendingPlay.value=true;playbackError.value=false
 prepareDrum();media.volume=musicVolume.value
 // Both calls originate in the click gesture. A suspended Web Audio promise
 // must never prevent the song from starting or resurrect a cancelled play.
 void unlockDrum().catch(()=>{if(epoch===playEpoch)drumError.value=true})
 try{
  await media.play()
  if(!props.active||audio.value!==media||!wantPlay)media.pause()
 }catch{if(epoch===playEpoch){playbackError.value=true;pause()}}
 finally{if(epoch===playEpoch)pendingPlay.value=false}
}
function clockSettings(){const shift=props.trim.start_ms;return {...props.modelValue,offset_ms:props.modelValue.offset_ms+shift,segments:props.modelValue.segments?.map(s=>({...s,start_ms:s.start_ms+shift,offset_ms:s.offset_ms+shift})),volume:drumVolume.value,end_time:props.trim.end_ms/1000}}
function stopDrum(){drum?.stop();cancelAnimationFrame(frame)}
function pause(){++playEpoch;wantPlay=false;pendingPlay.value=false;audio.value?.pause();stopDrum();isPlaying.value=false}
function mediaError(){playbackError.value=true;pause()}
function seek(time:number){if(audio.value)audio.value.currentTime=Math.max(props.trim.start_ms/1000,Math.min(props.trim.end_ms/1000,time));current.value=audio.value?.currentTime||0}
function seekLocal(time:number){seek(props.trim.start_ms/1000+time)}
function tick(){if(!audio.value)return;current.value=audio.value.currentTime;if(current.value<props.trim.start_ms/1000){seek(props.trim.start_ms/1000);current.value=props.trim.start_ms/1000}if(current.value>=props.trim.end_ms/1000){pause();seek(props.trim.end_ms/1000);return}frame=requestAnimationFrame(tick)}
function prepareDrum(){
 emit('playing');if(audio.value&&(audio.value.currentTime<props.trim.start_ms/1000||audio.value.currentTime>=props.trim.end_ms/1000))seek(props.trim.start_ms/1000)
}
async function playDrum(){
 if(!props.active||!audio.value||audio.value.paused||audio.value.readyState<3)return
 audio.value.volume=musicVolume.value;cancelAnimationFrame(frame);tick()
 if(!drumEnabled.value)return
 try{drum??=new DrumClock(new AudioContext(),audio.value,clockSettings);await drum.start();drumError.value=false}catch{drumError.value=true}
}
function resetDrum(){drum?.stop();if(props.active&&audio.value&&!audio.value.paused)void playDrum()}
watch([()=>props.modelValue,()=>props.trim,drumEnabled,drumVolume],resetDrum,{deep:true})
watch(musicVolume,value=>{if(audio.value)audio.value.volume=value})
watch(()=>props.active,active=>{if(!active)pause()})
function set(index:number,key:string,value:number){if(!Number.isFinite(value))return;const config=props.modelValue;if(index<0)emit('update:modelValue',{...config,[key]:value});else emit('update:modelValue',{...config,segments:config.segments!.map((s,i)=>i===index?{...s,[key]:value}:s)})}
function number(index:number,key:string,e:Event,add=0){set(index,key,(e.target as HTMLInputElement).valueAsNumber+add)}
function reset(index?:number){if(index===undefined)taps.value={};else taps.value[index]=[];tapError.value=''}
function applyTaps(index:number){try{const s=sections.value.find(s=>s.index===index)!;const fit=estimateTaps(taps.value[index]||[],s.value.tap_latency_ms);const update={bpm:fit.bpm,offset_ms:fit.offset_ms-props.trim.start_ms};emit('update:modelValue',index<0?{...props.modelValue,...update}:{...props.modelValue,segments:props.modelValue.segments!.map((p,i)=>i===index?{...p,...update}:p)});tapError.value=''}catch{tapError.value=t('tapInvalid')}}
function tap(index:number){const s=sections.value.find(s=>s.index===index)!;if(!audio.value||audio.value.paused){tapError.value=t('tapPlayFirst');return}const time=audio.value.currentTime,local=time*1000-props.trim.start_ms;if(local<s.start_ms||local>=s.end_ms){tapError.value=t('tapSegment');return}drumEnabled.value=false;drum?.stop();const list=taps.value[index]||[];if(list.length&&time-list.at(-1)!<.12)return;taps.value[index]=[...list,time].slice(-32);if(taps.value[index]!.length>=4)applyTaps(index)}
function add(){const ms=Math.round(current.value*1000)-props.trim.start_ms;if(ms<=0||ms>=props.trim.end_ms-props.trim.start_ms){tapError.value=t('seekInside');return}const points=props.modelValue.segments||[];if(points.length>=64||points.some(s=>Math.abs(s.start_ms-ms)<1))return;const s=sections.value.find(s=>ms>=s.start_ms&&ms<s.end_ms)!;emit('update:modelValue',{...props.modelValue,segments:[...points,{start_ms:ms,bpm:s.value.bpm,offset_ms:ms,tap_latency_ms:s.value.tap_latency_ms}].sort((a,b)=>a.start_ms-b.start_ms)});reset()}
function move(id:string,time:number){if(id==='in'||id==='out')return;const index=Number(id),points=[...(props.modelValue.segments||[])],s=points[index]!;const low=Math.max(1,(points[index-1]?.start_ms??-1)+1),high=Math.min(clipDuration.value-1,(points[index+1]?.start_ms??Infinity)-1);const ms=Math.max(low,Math.min(high,Math.round(time*1000)));points[index]={...s,start_ms:ms,offset_ms:s.offset_ms+ms-s.start_ms};emit('update:modelValue',{...props.modelValue,segments:points});reset()}
function remove(index:number){emit('update:modelValue',{...props.modelValue,segments:props.modelValue.segments?.filter((_,i)=>i!==index)});reset()}
watch(()=>props.audioUrl,()=>{pause();drum?.dispose();drum=undefined;playbackError.value=false;drumError.value=false;reset()})
onBeforeUnmount(()=>{pause();drum?.dispose()});defineExpose({pause})
</script>
<template>
 <div class="bpm-module">
  <label class="enable"><input type="checkbox" :checked="modelValue.enabled" :disabled="blocked" @change="emit('update:modelValue',{...modelValue,enabled:($event.target as HTMLInputElement).checked})"/>{{t('bpmEnable')}}</label>
  <p class="hint">{{t('bpmHint')}}</p>
  <audio ref="audio" :key="audioUrl" :src="audioUrl" hidden preload="metadata" @error="mediaError" @play="isPlaying=true;prepareDrum()" @playing="playDrum" @pause="isPlaying=false;stopDrum()" @waiting="stopDrum" @ended="isPlaying=false;stopDrum()" @seeking="stopDrum();reset()" @seeked="playDrum" @timeupdate="current=audio?.currentTime||0" @ratechange="resetDrum"/>
  <StudioTimeline :url="audioUrl.replace(/\/audio$/,'/waveform')" :duration="clipDuration/1000" :range-start="0" :range-end="clipDuration/1000" :source-start="trim.start_ms/1000" :source-end="trim.end_ms/1000" :current="localCurrent" :markers="markers" :blocked="blocked" :zoom-controls="true" :playing="isPlaying||pendingPlay" :playback-disabled="!active" @toggle="toggle" @move="move" @seek="seekLocal" :hide-time="true" :hint-above="true">
<template #controls>  <div class="playback-controls">
   <div class="drum-controls">
    <label class="enable"><input v-model="drumEnabled" type="checkbox" @change="unlockDrum().catch(()=>drumError=true)"/>{{t('drumEnabled')}}</label>
    <label class="volume"><span>{{t('drumVolume')}}<output>{{Math.round(drumVolume*100)}}%</output></span><input v-model.number="drumVolume" :aria-label="t('drumVolume')" :style="{'--volume':`${drumVolume*100}%`}" type="range" min="0" max="1" step=".01"/></label>
    <label class="volume"><span>{{t('musicVolume')}}<output>{{Math.round(musicVolume*100)}}%</output></span><input v-model.number="musicVolume" :aria-label="t('musicVolume')" :style="{'--volume':`${musicVolume*100}%`}" type="range" min="0" max="1" step=".01"/></label>
   </div>
   <div class="tempo-controls"><div class="toolbar"><button :disabled="blocked" @click="add">{{t('addTempo')}}</button><button @click="seek(trim.start_ms/1000)">{{t('jumpClipStart')}}</button></div></div>
  </div>
</template>
</StudioTimeline>
<p v-if="drumError" role="alert">{{t('drumUnavailable')}}</p>
  <p v-if="playbackError" role="alert">{{t('playbackUnavailable')}}</p>
  <p v-if="tapError" role="alert">{{tapError}}</p>
  <fieldset v-for="(s,n) in sections" :key="s.index" :disabled="blocked" class="segment">
   <div class="segment-heading"><h4>{{t('segment')}} {{n+1}}</h4><span>{{(s.start_ms/1000).toFixed(3)}} – {{(s.end_ms/1000).toFixed(3)}} s</span>   <div class="toolbar segment-tools"><button @click="detect(s.index)">{{t('autoBpm')}}</button><span role="status">{{detectionIndex===s.index?status:''}}</span><button @click="tap(s.index)" @keydown.space.prevent="!$event.repeat&&tap(s.index)">{{t('tapBpm')}}</button><span>{{t('tapCount')}} {{taps[s.index]?.length||0}} / 4</span><button @click="reset(s.index)">{{t('resetTaps')}}</button></div><button class="segment-jump" @click="seekLocal(s.start_ms/1000)">{{t('jumpSegment')}}</button><button v-if="n>0" @click="remove(s.index)">{{t('deleteTempo')}}</button></div>

   <div class="bpm-fields">
    <label>BPM<div class="bpm-value"><input type="number" :value="s.value.bpm" min="30" max="400" step="0.001" @input="number(s.index,'bpm',$event)"/><span><button :disabled="s.value.bpm/2<30" @click="set(s.index,'bpm',s.value.bpm/2)">÷ 2</button><button :disabled="s.value.bpm*2>400" @click="set(s.index,'bpm',s.value.bpm*2)">× 2</button></span></div></label>
    <label>{{t('beatOffset')}}<input type="number" :value="Math.round((s.value.offset_ms-s.start_ms)*1000)/1000" min="-600000" max="600000" step="1" @input="number(s.index,'offset_ms',$event,s.start_ms)"/><small>{{t('segmentOffsetHint')}}</small></label>
    <label>{{t('tapLatency')}}<input type="number" :value="s.value.tap_latency_ms" min="-1000" max="1000" step="1" @input="number(s.index,'tap_latency_ms',$event)"/><small>{{t('tapLatencyHint')}}</small><button v-if="(taps[s.index]?.length||0)>=4" @click="applyTaps(s.index)">{{t('reapplyTaps')}}</button></label>
   </div>
  </fieldset>
 </div>
</template>
<style scoped>
.playback-controls{display:flex;align-items:center;justify-content:space-between;gap:24px;flex-wrap:wrap;margin:14px 0}
.drum-controls{display:flex;align-items:center;gap:24px;flex-wrap:wrap}
.drum-controls label{margin:0}
.drum-controls .volume{display:flex;flex-direction:column;gap:6px;width:172px;min-width:0}
.volume>span{display:flex;align-items:baseline;justify-content:space-between;gap:12px}
.volume output{font-variant-numeric:tabular-nums;color:#61756a}
.drum-controls input[type=range]{display:block;box-sizing:border-box;appearance:none;width:100%;height:20px;padding:0;margin:0;border:0;border-radius:0;background:transparent;cursor:pointer;accent-color:#23856d}
.drum-controls input[type=range]::-webkit-slider-runnable-track{height:5px;border-radius:9px;background:linear-gradient(to right,#23856d var(--volume),#dce5db var(--volume))}
.drum-controls input[type=range]::-webkit-slider-thumb{appearance:none;width:14px;height:14px;border-radius:50%;margin-top:-4.5px;background:#23856d;box-shadow:0 0 0 2px white}
.drum-controls input[type=range]::-moz-range-track{height:5px;border-radius:9px;background:#dce5db}
.drum-controls input[type=range]::-moz-range-progress{height:5px;border-radius:9px;background:#23856d}
.drum-controls input[type=range]::-moz-range-thumb{width:14px;height:14px;border:2px solid white;border-radius:50%;background:#23856d}
.drum-controls input[type=range]:focus-visible{outline:2px solid #23856d;outline-offset:3px;border-radius:3px}
.tempo-controls{max-width:360px}.tempo-controls .toolbar{justify-content:flex-end;margin:0 0 4px;gap:10px}.tempo-controls button{margin:0}.tempo-controls>.hint{display:block;text-align:right}
.enable{display:flex;align-items:center;gap:10px;font-weight:600}.hint,small{color:#61756a;font-size:12px;line-height:1.7}audio{width:100%;margin:10px 0 5px}.toolbar,.segment-heading{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:8px 0}.toolbar span,.segment-heading span{font-size:12px;color:#61756a}.segment-heading h4{margin:0}.segment-heading button:first-of-type{margin-left:auto}.bpm-fields{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px;margin-top:22px}label{display:flex;flex-direction:column;gap:8px;font-size:13px}.enable{flex-direction:row}input[type=number]{padding:11px;width:100%;border:1px solid #dce5db;border-radius:7px;background:white}label>span{display:flex;gap:8px}button{font-size:12px}fieldset{border:1px solid #dce5db;padding:20px;min-width:0;border-radius:10px;margin-top:18px}p[role=alert]{color:#bd453c}@media(max-width:650px){.bpm-fields{grid-template-columns:1fr}}

.playback-controls{flex:1;gap:16px;margin:0;flex-wrap:wrap}.drum-controls{gap:14px}.drum-controls .enable{font-size:12px;white-space:nowrap}.drum-controls .volume{width:115px;font-size:11px;gap:2px}.tempo-controls .toolbar{margin:0;gap:6px}.tempo-controls button{padding:8px 10px;font-size:11px}.segment-heading{gap:12px;margin:0}.segment-heading .segment-tools{margin:0 0 0 8px;gap:8px}.segment-tools button{margin:0;padding:8px 10px}.segment-heading .segment-tools button:first-of-type{margin-left:0}.segment-heading .segment-jump{margin-left:auto}.bpm-value{display:flex;align-items:center;gap:6px}.bpm-value input{min-width:0;flex:1}.bpm-value>span{display:flex;gap:5px}.bpm-value button{padding:9px 10px;white-space:nowrap;margin:0}.bpm-fields label{margin:0;min-width:0;width:100%}.bpm-fields{gap:16px;margin-top:18px}
@media(max-width:1100px){.playback-controls{gap:10px}.drum-controls{gap:10px}.drum-controls .volume{width:100px}}
</style>
