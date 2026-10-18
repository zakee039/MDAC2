<script setup lang="ts">
import { ref, onBeforeUnmount, watch } from 'vue'
import { ChartEditor, subscribeOverlay } from './ChartEditor.ts'
import type { EditableChart, Note } from './ChartEditor.ts'
import type { Transport } from '../audio/Transport.ts'
const props=defineProps<{chart:EditableChart;timingId:string;time:(q:string)=>number;transport:Transport}>()
const emit=defineEmits<{change:[chart:EditableChart]}>()
let editor=new ChartEditor(props.chart,props.timingId,props.time)
const current=ref(editor.snapshot()),error=ref(''),selected=ref(''),q=ref('0'),duration=ref('0'),semantic=ref<Note['semantic']>('tap'),lane=ref('ground'),accent=ref(false)
const overlay=ref<ReturnType<ChartEditor['overlay']>>([]),position=ref(0)
let unsubscribe=subscribeOverlay(editor,props.transport,(items,seconds)=>{overlay.value=items;position.value=seconds})
watch(()=>[props.chart,props.timingId] as const,()=>{
  if(props.chart.revision_id===current.value.revision_id && props.timingId===current.value.timing_revision) return
  unsubscribe();editor=new ChartEditor(props.chart,props.timingId,props.time);current.value=editor.snapshot()
  unsubscribe=subscribeOverlay(editor,props.transport,(items,seconds)=>{overlay.value=items;position.value=seconds})
})
watch(()=>props.transport,()=>{unsubscribe();unsubscribe=subscribeOverlay(editor,props.transport,(items,seconds)=>{overlay.value=items;position.value=seconds})})
onBeforeUnmount(()=>unsubscribe())
function act(fn:()=>void){try{fn();current.value=editor.snapshot();overlay.value=editor.overlay(position.value);emit('change',editor.snapshot());error.value=''}catch(e){error.value=(e as Error).message}}
function fields(){return {q:q.value,duration_q:duration.value,semantic:semantic.value,lane_mask:(lane.value==='both'?['ground','air']:[lane.value]) as Note['lane_mask'],accent:accent.value}}
function select(note:Note){selected.value=note.event_id;q.value=note.q;duration.value=note.duration_q;semantic.value=note.semantic;lane.value=note.lane_mask.length===2?'both':note.lane_mask[0]!;accent.value=note.accent}
</script>
<template>
  <section aria-label="谱面编辑器"><h2>谱面 · {{current.events.length}} 个逻辑事件</h2>
    <label>起点 q <input v-model="q" /></label><label>时长 q <input v-model="duration" /></label>
    <label>语义 <select v-model="semantic"><option value="tap">Note</option><option value="hold">Hold</option><option value="gemini">Gemini</option></select></label>
    <label>轨道 <select v-model="lane"><option value="ground">地面</option><option value="air">空中</option><option value="both">双轨</option></select></label><label><input v-model="accent" type="checkbox" />重音</label>
    <button @click="act(()=>editor.add({...fields(),source_kind:'manual',source_id:null,locked:false}))">新增音符</button>
    <button :disabled="!selected" @click="act(()=>editor.update(selected,fields()))">修改所选</button>
    <button :disabled="!selected" @click="act(()=>editor.remove(selected))">删除所选</button>
    <button @click="act(()=>editor.undo())">撤销</button><button @click="act(()=>editor.redo())">重做</button><p v-if="error" role="alert">{{error}}</p>
    <p v-if="!current.events.length">当前为空谱，尚不能作为有效谱面导出。</p>
    <ul><li v-for="note in current.events" :key="note.event_id"><button @click="select(note)">{{note.q}} · {{note.semantic}} · {{note.lane_mask.join('/')}} · {{note.duration_q}}</button><button @click="act(()=>editor.lock(note.event_id,!note.locked))">{{note.locked?'显式解锁':'锁定'}}</button></li></ul>
    <h3>播放叠层 · {{position.toFixed(3)}} 秒</h3>
    <svg viewBox="0 0 600 120" role="img" aria-label="共享播放时钟附近的音符">
      <path d="M0 35H600 M0 85H600 M300 0V120" stroke="currentColor" />
      <g v-for="item in overlay" :key="item.note.event_id">
        <template v-for="physical in item.note.lane_mask" :key="physical">
          <rect :x="300+(item.start-position)*140" :y="physical==='air'?25:75" :width="Math.max(8,(item.end-item.start)*140)" height="20" :fill="item.note.locked?'#b88635':'#32977c'" />
        </template>
      </g>
    </svg>
  </section>
</template>
