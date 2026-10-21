<script setup lang="ts">
import {t} from '../studioI18n'
import {ref,computed,watch,onBeforeUnmount} from 'vue'
import {markerPosition} from '../studioTimeline'
const props=defineProps<{url:string;duration:number;current:number;markers:{id:string;time:number;label:string}[];blocked?:boolean;rangeStart?:number;rangeEnd?:number;sourceStart?:number;sourceEnd?:number;zoomControls?:boolean;hideTime?:boolean;hintAbove?:boolean;playing?:boolean;playbackDisabled?:boolean}>()
const emit=defineEmits<{move:[id:string,time:number];seek:[time:number];toggle:[]}>()
function clock(seconds:number){const s=Math.max(0,seconds);return `${Math.floor(s/60)}:${(s%60).toFixed(1).padStart(4,'0')}`}
const box=ref<HTMLElement>(),peaks=ref<number[]>([]),error=ref(''),zoom=ref(false),center=ref(0),span=ref(4)
// Visible window; defaults to the full song.
const low=computed(()=>Math.max(0,Math.min(props.duration,props.rangeStart??0)))
const high=computed(()=>Math.max(low.value+.001,Math.min(props.duration,props.rangeEnd??props.duration)))
const start=computed(()=>zoom.value?Math.max(low.value,Math.min(high.value-span.value,center.value-span.value/2)):low.value)
const end=computed(()=>zoom.value?Math.min(high.value,start.value+span.value):high.value)
let controller:AbortController|undefined,delay:ReturnType<typeof setTimeout>|undefined,drag:{id:string;x:number;y:number;original:number;time:number;width:number;fine:boolean}|undefined
function pos(time:number){return 100*(time-start.value)/Math.max(.001,end.value-start.value)}
const visibleMarkers=computed(()=>props.markers.flatMap(m=>{const position=markerPosition(m.time,start.value,end.value);return position===null?[]:[{...m,position}]}))
const wave=computed(()=>peaks.value.map((p,i)=>`M${i*1000/peaks.value.length},${40-p*38}v${p*76}`).join(''))
async function load(){controller?.abort();controller=new AbortController();const signal=controller.signal;const sourceStart=props.sourceStart??start.value,sourceEnd=props.sourceEnd??end.value;try{const r=await fetch(props.url+`?start=${sourceStart}&end=${sourceEnd}`,{signal});if(!r.ok)throw Error();const d=await r.json();if(!signal.aborted){peaks.value=d.peaks;error.value=''}}catch(e){if(!signal.aborted)error.value=t('waveUnavailable')}}
watch([()=>props.url,start,end,()=>props.sourceStart,()=>props.sourceEnd],()=>{clearTimeout(delay);delay=setTimeout(load,100)},{immediate:true})
function down(e:PointerEvent,m:{id:string;time:number}){if(props.blocked)return;e.preventDefault();(e.currentTarget as HTMLElement).focus();(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);drag={id:m.id,x:e.clientX,y:e.clientY,original:m.time,time:m.time,width:box.value!.clientWidth,fine:zoom.value}}
function move(e:PointerEvent){if(!drag)return;const up=drag.y-e.clientY;if(up>30){if(!drag.fine){drag.fine=true;zoom.value=true;center.value=drag.time;drag.x=e.clientX}span.value=Math.max(.5,4/2**((up-30)/55))}const range=drag.fine?span.value:high.value-low.value;drag.time=Math.round(Math.max(low.value,Math.min(high.value,drag.time+(e.clientX-drag.x)*range/drag.width))*1000)/1000;drag.x=e.clientX;emit('move',drag.id,drag.time)}
function changeZoom(direction:number){const full=Math.max(.001,high.value-low.value);const currentSpan=zoom.value?span.value:full;span.value=Math.max(.5,Math.min(full,currentSpan*(direction>0?.5:2)));center.value=Math.max(low.value,Math.min(high.value,props.current));zoom.value=span.value<full-1e-6}
function restoreZoom(){zoom.value=false;center.value=Math.max(low.value,Math.min(high.value,props.current));span.value=Math.max(.5,high.value-low.value)}
function release(){drag=undefined}
function key(e:KeyboardEvent,m:{id:string;time:number}){if(e.key==='Escape'){if(drag)emit('move',drag.id,drag.original);release();zoom.value=false;return}if(props.blocked)return;if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();emit('move',m.id,m.time+(e.key==='ArrowLeft'?-1:1)*(e.shiftKey ? .001 : .01))}}
function seek(e:MouseEvent){if(drag)return;const r=box.value!.getBoundingClientRect();emit('seek',start.value+(e.clientX-r.left)/r.width*(end.value-start.value))}
onBeforeUnmount(()=>{clearTimeout(delay);controller?.abort()})
</script>
<template><div class="timeline-shell">
 <div class="timeline-toolbar">
  <div class="timeline-playback"><button type="button" class="wave-play" :aria-label="t(playing?'pauseClip':'playClip')" :title="t(playing?'pauseClip':'playClip')" :disabled="playbackDisabled" @click="emit('toggle')"><svg viewBox="0 0 20 20" aria-hidden="true"><path v-if="playing" d="M5 3h4v14H5zM12 3h4v14h-4z"/><path v-else d="M5 2l13 8-13 8z"/></svg></button><span v-if="!hideTime">{{clock(current)}} / {{clock(duration)}}</span></div>
  <slot name="controls"/>
  <div class="timeline-zoom"><span>{{t('waveformScale')}}</span><div class="timeline-tools"><button type="button" :aria-label="t('zoomOut')" :title="t('zoomOut')" :disabled="blocked" @click="changeZoom(-1)">−</button><button type="button" :aria-label="t('zoomIn')" :title="t('zoomIn')" :disabled="blocked" @click="changeZoom(1)">+</button><button type="button" :aria-label="t('restoreZoom')" :title="t('restoreZoom')" :disabled="blocked" @click="restoreZoom">{{t('restoreZoom')}}</button></div></div>
 </div>
 <small v-if="hintAbove" class="wave-hint-above">{{error||t('waveHint')}}</small>
 <div ref="box" class="timeline" @click.self="seek">
  <svg viewBox="0 0 1000 80" preserveAspectRatio="none" @click="seek"><path :d="wave" stroke="#91b5a8" stroke-width="1"/><line :x1="pos(current)*10" :x2="pos(current)*10" y1="0" y2="80" stroke="#273f34" stroke-width="2"/></svg>
  <button v-for="m in visibleMarkers" :key="m.id" class="marker" :style="{left:m.position+'%'}" :aria-label="m.label" :title="m.label+' · '+m.time.toFixed(3)+'s'" :disabled="blocked" @click.stop @pointerdown="down($event,m)" @pointermove="move" @pointerup="release" @pointercancel="release" @lostpointercapture="release" @keydown="key($event,m)">{{m.label}}</button>
 </div><div class="scale"><span>{{start.toFixed(3)}} s</span><button v-if="zoom&&!zoomControls" type="button" @click="restoreZoom">{{t('fullWave')}}</button><span>{{end.toFixed(3)}} s</span></div>
 <small v-if="!hintAbove">{{error||t('waveHint')}}</small>
</div></template>
<style scoped>
.timeline-shell{margin:25px 0 16px}.timeline-toolbar{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;margin:0 25px 30px;color:#61756a;font-size:12px}.timeline-playback,.timeline-zoom{display:flex;align-items:center;gap:10px}.timeline-playback{font-variant-numeric:tabular-nums}.timeline-tools{display:flex;gap:5px}.timeline-toolbar button{margin:0}.timeline-playback .wave-play{display:grid;place-items:center;width:34px;height:34px;padding:8px;border-radius:8px;background:#23856d;color:white;border:0}.wave-play svg{width:16px;height:16px;fill:currentColor}.wave-play:focus-visible{outline:2px solid #23856d;outline-offset:3px}.timeline-tools button{min-width:30px;padding:4px 8px;font-size:13px}.timeline{height:100px;position:relative;background:#f0f5ef;border-radius:10px;margin:0 25px;touch-action:none}.timeline svg{width:100%;height:100%;cursor:crosshair}.marker{position:absolute;top:-15px;margin:0;transform:translateX(-50%);padding:5px 9px!important;font-size:12px;white-space:nowrap;touch-action:none;border:1px solid #23856d;background:#fff;color:#156d57;cursor:ew-resize;z-index:1}.marker:after{content:'';position:absolute;top:100%;left:50%;height:90px;border-left:2px solid #23856d;pointer-events:none}.scale{display:flex;align-items:center;justify-content:space-between;margin:8px 25px;font-size:12px}.scale button{padding:3px 10px}small{font-size:12px;color:#61756a}
.wave-hint-above{display:block;margin:-16px 25px 28px}.timeline-zoom{margin-left:auto}
</style>
