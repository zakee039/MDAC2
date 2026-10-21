<script setup lang="ts">
import {t} from '../studioI18n'
import {ref,watch,onBeforeUnmount} from 'vue'
import type {Trim} from '../studioTimeline'
import StudioTimeline from './StudioTimeline.vue'
const props=defineProps<{modelValue:Trim;duration:number;audioUrl:string;active:boolean;blocked:boolean}>()
const emit=defineEmits<{'update:modelValue':[value:Trim];playing:[]}>()
const audio=ref<HTMLAudioElement>(),current=ref(0),status=ref(''),running=ref(false)
const isPlaying=ref(false),pendingPlay=ref(false),playbackError=ref(false)
let frame=0,controller:AbortController|undefined,playEpoch=0,wantPlay=false
function move(id:string,time:number){if(!Number.isFinite(time))return;const t=Math.round(time*1000),v=props.modelValue;const next=id==='in'?{...v,start_ms:Math.max(0,Math.min(t,v.end_ms-1000))}:{...v,end_ms:Math.min(Math.round(props.duration*1000),Math.max(t,v.start_ms+1000))};emit('update:modelValue',next)}
function seek(t:number){if(audio.value)audio.value.currentTime=Math.max(0,Math.min(props.duration,t));current.value=t}
function tick(){if(!audio.value)return;current.value=audio.value.currentTime;if(current.value<props.modelValue.start_ms/1000){seek(props.modelValue.start_ms/1000);current.value=props.modelValue.start_ms/1000}if(current.value>=props.modelValue.end_ms/1000){audio.value.pause();seek(props.modelValue.end_ms/1000);return}frame=requestAnimationFrame(tick)}
function prepare(){emit('playing');if(audio.value&&(audio.value.currentTime<props.modelValue.start_ms/1000||audio.value.currentTime>=props.modelValue.end_ms/1000))seek(props.modelValue.start_ms/1000)}
function play(){if(!props.active){pause();return}isPlaying.value=true;prepare();cancelAnimationFrame(frame);tick()}
function stopFrame(){cancelAnimationFrame(frame)}
function pause(){++playEpoch;wantPlay=false;pendingPlay.value=false;isPlaying.value=false;stopFrame();audio.value?.pause()}
async function start(){
 const media=audio.value
 if(!media||!props.active)return
 const epoch=++playEpoch;wantPlay=true;pendingPlay.value=true;playbackError.value=false
 prepare()
 try{await media.play();if(!props.active||audio.value!==media||!wantPlay)media.pause()}
 catch{if(epoch===playEpoch){playbackError.value=true;pause()}}
 finally{if(epoch===playEpoch)pendingPlay.value=false}
}
function toggle(){if(pendingPlay.value||audio.value&&!audio.value.paused)pause();else void start()}
async function tail(){seek(Math.max(props.modelValue.start_ms/1000,props.modelValue.end_ms/1000-5));await start()}
async function detect(){if(running.value)return;const sourceUrl=props.audioUrl;running.value=true;status.value=t('trimRunning');controller=new AbortController();const pending=controller;const snapshot={...props.modelValue};try{const r=await fetch(props.audioUrl.replace(/\/audio$/,'/trim/detect'),{method:'POST',signal:controller.signal});if(!r.ok)throw Error(t('trimFailed'));const d=await r.json();if(pending.signal.aborted||sourceUrl!==props.audioUrl)return;if(snapshot.start_ms!==props.modelValue.start_ms||snapshot.end_ms!==props.modelValue.end_ms){status.value=t('trimChanged');return}status.value=d.reason;if(d.candidate_ms>snapshot.start_ms+1000)move('out',d.candidate_ms/1000)}catch(e){if(!pending.signal.aborted)status.value=e instanceof Error?e.message:'检测失败'}finally{if(controller===pending)running.value=false}}
watch(()=>props.audioUrl,()=>{pause();controller?.abort();controller=undefined;running.value=false;status.value='';current.value=0;playbackError.value=false})
watch(()=>props.active,a=>{if(!a)pause()});watch(()=>props.modelValue,()=>{if(audio.value&&!audio.value.paused&&(audio.value.currentTime<props.modelValue.start_ms/1000||audio.value.currentTime>=props.modelValue.end_ms/1000))pause()})
onBeforeUnmount(()=>{pause();controller?.abort()});defineExpose({pause})
</script>
<template><div>
 <p class="hint">{{t('trimHint')}}</p>
 <div class="trim-toolbar"> <div class="actions"><button @click="seek(modelValue.start_ms/1000)">{{t('jumpIn')}}</button><button @click="tail">{{t('auditionTail')}}</button><button :disabled="blocked||running" @click="detect">{{t('autoTrim')}}</button><button :disabled="blocked" @click="emit('update:modelValue',{start_ms:0,end_ms:Math.round(duration*1000)})">{{t('restoreSong')}}</button><span role="status">{{status}}</span></div> <div class="fields"><label>{{t('inPoint')}}（{{t('seconds')}}）<input type="number" :disabled="blocked" :value="modelValue.start_ms/1000" min="0" :max="modelValue.end_ms/1000-1" step=".001" @change="move('in',($event.target as HTMLInputElement).valueAsNumber)"/></label><label>{{t('outPoint')}}（{{t('seconds')}}）<input type="number" :disabled="blocked" :value="modelValue.end_ms/1000" :min="modelValue.start_ms/1000+1" :max="duration" step=".001" @change="move('out',($event.target as HTMLInputElement).valueAsNumber)"/></label><span>{{t('selectedSpan')}} {{((modelValue.end_ms-modelValue.start_ms)/1000).toFixed(3)}} {{t('seconds')}}</span></div></div>
 <audio ref="audio" :key="audioUrl" :src="audioUrl" hidden preload="metadata" @play="play" @pause="isPlaying=false;stopFrame()" @ended="isPlaying=false;stopFrame()" @waiting="stopFrame" @playing="play" @seeking="stopFrame" @seeked="audio&&!audio.paused&&play()" @error="playbackError=true;pause()" @timeupdate="current=audio?.currentTime||0"/>
 <StudioTimeline :url="audioUrl.replace(/\/audio$/,'/waveform')" :duration="duration" :current="current" :blocked="blocked" :playing="isPlaying||pendingPlay" :playback-disabled="!active" :zoom-controls="true" :markers="[{id:'in',time:modelValue.start_ms/1000,label:t('inPoint')},{id:'out',time:modelValue.end_ms/1000,label:t('outPoint')}]" @toggle="toggle" @move="move" @seek="seek"/>
 <p v-if="playbackError" role="alert">{{t('playbackUnavailable')}}</p>

</div></template>
<style scoped>audio{width:100%;margin-top:15px}.hint{font-size:13px;color:#61756a}.actions,.fields{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.actions span{font-size:12px;max-width:450px}.fields label{display:grid;gap:8px;font-size:13px}.fields input{padding:10px;border:1px solid #dce5db;border-radius:7px;max-width:180px}.fields span{font-size:13px}.trim-toolbar{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;flex-wrap:wrap}.trim-toolbar .actions{gap:8px;flex:1}.trim-toolbar button{margin:0;padding:9px 11px;font-size:12px}.trim-toolbar .fields{align-items:end;gap:12px}.trim-toolbar .fields label{margin:0;width:120px;font-size:12px;gap:5px}.trim-toolbar .fields input{width:100%;height:38px;padding:8px}.trim-toolbar .fields>span{display:flex;align-items:center;height:38px;white-space:nowrap;font-size:12px}
</style>
