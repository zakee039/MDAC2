<script setup lang="ts">
import {ref,watch,onBeforeUnmount} from 'vue'
import {t} from '../studioI18n'
import {estimateTaps,type BeatSettings} from '../studioBpm'
import {DrumClock} from '../studioMetronome'
const props=defineProps<{modelValue:BeatSettings;audioUrl:string;active:boolean;blocked:boolean;status:string;analysis:any}>()
const emit=defineEmits<{'update:modelValue':[value:BeatSettings];detect:[]}>()
const audio=ref<HTMLAudioElement|null>(null),taps=ref<number[]>([]),tapError=ref('')
const drumEnabled=ref(true),drumVolume=ref(1),musicVolume=ref(.65),drumError=ref(false)
let drum:DrumClock|undefined,playing=false
function stopDrum(){playing=false;drum?.stop()}
async function prepareDrum(){
 if(!props.active||!audio.value||!drumEnabled.value)return
 try{drum??=new DrumClock(new AudioContext(),audio.value,()=>({...props.modelValue,volume:drumVolume.value}));await drum.unlock()}catch{drumError.value=true}
}
async function playDrum(){
 if(!props.active||!audio.value||audio.value.paused||audio.value.readyState<3)return
 playing=true;audio.value.volume=musicVolume.value
 if(!drumEnabled.value)return
 try{drum??=new DrumClock(new AudioContext(),audio.value,()=>({...props.modelValue,volume:drumVolume.value}));await drum.start();drumError.value=false}catch{drumError.value=true}
}
function resetDrum(){drum?.stop();if(playing)void playDrum()}
watch([()=>props.modelValue.bpm,()=>props.modelValue.offset_ms,drumEnabled,drumVolume],resetDrum)
watch(musicVolume,value=>{if(audio.value)audio.value.volume=value})
watch(()=>props.active,active=>{if(!active){audio.value?.pause();stopDrum()}})
function set(key:keyof BeatSettings,value:number|boolean){emit('update:modelValue',{...props.modelValue,[key]:value})}
function number(key:keyof BeatSettings,e:Event){set(key,(e.target as HTMLInputElement).valueAsNumber)}
function reset(){taps.value=[];tapError.value=''}
function applyTaps(){try{const fit=estimateTaps(taps.value,props.modelValue.tap_latency_ms);emit('update:modelValue',{...props.modelValue,bpm:fit.bpm,offset_ms:fit.offset_ms});tapError.value=''}catch{tapError.value=t('tapInvalid')}}
function tap(){if(!audio.value||audio.value.paused){tapError.value=t('tapPlayFirst');return}const time=audio.value.currentTime;if(taps.value.length&&time-taps.value.at(-1)!<.12)return;taps.value=[...taps.value,time].slice(-32);if(taps.value.length>=4)applyTaps()}
watch(()=>props.audioUrl,()=>{audio.value?.pause();stopDrum();drum?.dispose();drum=undefined;reset()})
onBeforeUnmount(()=>{audio.value?.pause();drum?.dispose()})
</script>
<template>
 <div class="bpm-module">
  <label class="enable"><input type="checkbox" :checked="modelValue.enabled" :disabled="blocked" @change="set('enabled',($event.target as HTMLInputElement).checked)"/>{{t('bpmEnable')}}</label>
  <p class="hint">{{t('bpmHint')}}</p>
  <audio ref="audio" :key="audioUrl" :src="audioUrl" controls preload="none" @play="prepareDrum" @playing="playDrum" @pause="stopDrum" @waiting="stopDrum" @ended="stopDrum" @seeking="stopDrum();reset()" @seeked="playDrum" @ratechange="resetDrum"/>
  <div class="drum-controls">
   <label class="enable"><input v-model="drumEnabled" type="checkbox"/>{{t('drumEnabled')}}</label>
   <label>{{t('drumVolume')}} · {{Math.round(drumVolume*100)}}%<input v-model.number="drumVolume" type="range" min="0" max="1" step=".05"/></label>
   <label>{{t('musicVolume')}} · {{Math.round(musicVolume*100)}}%<input v-model.number="musicVolume" type="range" min="0" max="1" step=".05"/></label>
  </div>
  <p class="hint">{{t('drumHint')}}</p><p v-if="drumError" role="alert">{{t('drumUnavailable')}}</p>
  <fieldset :disabled="blocked">
   <div class="measurement"><button @click="emit('detect')">{{t('autoBpm')}}</button><span role="status" aria-live="polite">{{status}}</span></div>
   <p v-if="analysis" class="hint">{{t('bpmEvidence')}} {{analysis.beat_count}} · P95 {{analysis.p95_ms}} ms <strong v-if="analysis.variable_tempo_warning">{{t('variableBpm')}}</strong></p>
   <div class="tap-row"><button @click="tap" @keydown.space.prevent="!$event.repeat&&tap()">{{t('tapBpm')}}</button><span>{{t('tapCount')}} {{taps.length}} / 4</span><button @click="reset">{{t('resetTaps')}}</button></div>
   <p class="hint">{{t('tapHint')}}</p><p v-if="tapError" role="alert">{{tapError}}</p>
   <div class="bpm-fields">
    <label>BPM<input type="number" :value="modelValue.bpm" min="30" max="400" step="0.001" @input="number('bpm',$event)"/><span><button :disabled="modelValue.bpm/2<30" @click="set('bpm',modelValue.bpm/2)">÷ 2</button><button :disabled="modelValue.bpm*2>400" @click="set('bpm',modelValue.bpm*2)">× 2</button></span></label>
    <label>{{t('beatOffset')}}<input type="number" :value="modelValue.offset_ms" min="-600000" max="600000" step="1" @input="number('offset_ms',$event)"/><small>{{t('beatOffsetHint')}}</small></label>
    <label>{{t('tapLatency')}}<input type="number" :value="modelValue.tap_latency_ms" min="-1000" max="1000" step="1" @input="number('tap_latency_ms',$event)"/><small>{{t('tapLatencyHint')}}</small><button v-if="taps.length>=4" @click="applyTaps">{{t('reapplyTaps')}}</button></label>
   </div>
  </fieldset>
 </div>
</template>
<style scoped>
.drum-controls{display:flex;align-items:center;gap:24px;flex-wrap:wrap}.drum-controls label{min-width:140px}.drum-controls input[type=range]{width:100%;accent-color:#23856d}
.enable{display:flex;align-items:center;gap:10px;font-weight:600}.hint,small{color:#61756a;font-size:12px;line-height:1.7}.hint strong{display:block;color:#916b31}audio{width:100%;margin:10px 0 20px}.measurement,.tap-row{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:8px 0}.measurement span,.tap-row span{font-size:12px;color:#61756a}.bpm-fields{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px;margin-top:22px}label{display:flex;flex-direction:column;gap:8px;font-size:13px}.enable{flex-direction:row}input[type=number]{padding:11px;width:100%;border:1px solid #dce5db;border-radius:7px;background:white}label>span{display:flex;gap:8px}button{font-size:12px}fieldset{border:0;padding:0;min-width:0}@media(max-width:650px){.bpm-fields{grid-template-columns:1fr}}
</style>
