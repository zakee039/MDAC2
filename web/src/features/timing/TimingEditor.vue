<script setup lang="ts">
import { ref, watch } from 'vue'
import { TimingEditor } from './TimingEditor.ts'
import type { Timing, TimingEdit } from './TimingEditor.ts'
const props=defineProps<{ timing: Timing; position?: number }>()
const emit=defineEmits<{ change: [timing: Timing]; seek: [seconds: number] }>()
let editor=new TimingEditor(props.timing)
const current=ref(editor.snapshot()), preview=ref<Timing|null>(null), error=ref(''), q=ref(0), seconds=ref(0), latency=ref(0), tapCount=ref(0)
const meterStart=ref(0), numerator=ref(4), denominator=ref(4), grid=ref(1)
watch(()=>props.timing, value=>{ if(value.id!==current.value.id) { editor=new TimingEditor(value); refresh() } })
function refresh() { current.value=editor.snapshot(); preview.value=editor.preview() }
function act(fn:()=>void) { try { fn(); error.value=''; refresh() } catch(e) { error.value=(e as Error).message; refresh() } }
function propose(edit:TimingEdit) { act(()=>editor.propose(edit)) }
function commit() { act(()=>{editor.commit(current.value.id); emit('change',editor.snapshot())}) }
function history(kind:'undo'|'redo') { act(()=>{editor[kind]();emit('change',editor.snapshot())}) }
</script>
<template>
  <section aria-label="手工时间轴">
    <h2>时间轴 · {{ current.review === 'pending' ? '待复核' : current.review }}</h2>
    <p>修改先预览，再确认。q 为四分音符拍坐标；秒为音频位置。</p>
    <label>拍 q <input v-model.number="q" type="number" step="any" /></label>
    <label>秒 <input v-model.number="seconds" type="number" step="any" /></label>
    <button @click="propose({type:'insert',q,t:seconds})">预览插入</button>
    <button @click="propose({type:'drag',q,t:seconds})">预览移动</button>
    <button @click="propose({type:'delete',q})">预览删除</button>
    <button @click="propose({type:'downbeat',q,enabled:true})">预览强拍</button>
    <button @click="propose({type:'downbeat',q,enabled:false})">取消强拍预览</button>
    <button @click="history('undo')">撤销</button><button @click="history('redo')">重做</button>
    <details><summary>拍号与网格</summary>
      <label>拍号起点 q <input v-model.number="meterStart" type="number" step="any" /></label>
      <label>分子 <input v-model.number="numerator" type="number" min="1" /></label><label>分母 <input v-model.number="denominator" type="number" min="1" /></label>
      <button @click="propose({type:'meter',meters:[...current.meters.filter(m=>m.start!==meterStart),{start:meterStart,numerator,denominator}].sort((a,b)=>a.start-b.start)})">预览拍号修改</button>
      <p v-for="m in (preview ?? current).meters" :key="m.start">从 q={{m.start}}：{{m.numerator}}/{{m.denominator}}</p>
      <label>输入网格细分 <select v-model="grid"><option :value=".5">半倍</option><option :value="1">一倍</option><option :value="2">双倍</option></select></label>
      <button @click="q+=1/grid">输入拍前进一格</button><p>仅改变输入步长，不改变音乐速度和现有锚点。</p>
    </details>
    <p v-if="error" role="alert">{{ error }}</p>
    <table><thead><tr><th>拍</th><th>当前秒</th><th>预览秒</th><th>强拍</th></tr></thead><tbody>
      <tr v-for="a in (preview ?? current).anchors" :key="a.q"><td>{{ a.q }}</td><td>{{ current.anchors.find(old=>old.q===a.q)?.t ?? '新增' }}</td><td><button @click="emit('seek',a.t)">{{ a.t.toFixed(3) }}</button></td><td>{{ (preview ?? current).downbeats.includes(a.q) ? '是' : '否' }}</td></tr>
    </tbody></table>
    <p v-if="preview">预览：{{ current.anchors.length }} → {{ preview.anchors.length }} 个锚点。确认后原有批准失效。</p>
    <button :disabled="!preview" @click="commit">确认时间修订</button><button :disabled="!preview" @click="act(()=>editor.cancel())">放弃预览</button>
    <h3>局部分析证据</h3>
    <p v-if="!current.evidence.length">此编辑视图未载入局部分数；自动分析的原始观测和质量报告见候选区。</p>
    <ul><li v-for="(e,i) in current.evidence" :key="i"><button @click="emit('seek',e.start)">{{ e.start }}–{{ e.end }} 秒</button> · 分数 {{ e.score }} · {{ e.reason }}{{ e.score<.5 ? ' · 建议听检' : '' }}</li></ul>
    <label>输入延迟（秒）<input v-model.number="latency" type="number" min="-1" max="1" step=".01" /></label>
    <button :disabled="position===undefined" @click="act(()=>{editor.tap(position!,latency);tapCount=editor.taps.length})">记录 Tap</button> 已记录 {{ tapCount }} 次
  </section>
</template>
