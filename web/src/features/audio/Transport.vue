<script setup lang="ts">
import { onBeforeUnmount, shallowRef } from 'vue'
import type { Transport } from './Transport'
const props = defineProps<{ transport: Transport }>()
const state = shallowRef(props.transport.snapshot())
const unsubscribe = props.transport.subscribe((value) => { state.value = value })
onBeforeUnmount(unsubscribe)
async function toggle() {
  if (state.value.paused) await props.transport.play().catch(() => {})
  else props.transport.pause()
}
</script>

<template>
  <section class="transport" aria-label="音频播放控制">
    <button class="primary play-button" type="button" @click="toggle">{{ state.paused ? '播放' : '暂停' }}</button>
    <label class="seek-control">播放位置 <input type="range" min="0" :max="state.duration" step="0.01" :value="state.position"
      :disabled="state.duration === 0" @input="transport.seek(Number(($event.target as HTMLInputElement).value))" /></label>
    <output>{{ state.position.toFixed(2) }} / {{ state.duration.toFixed(2) }} 秒</output>
    <label>试听速度 <select :value="state.rate" @change="transport.setRate(Number(($event.target as HTMLSelectElement).value))">
      <option :value="0.5">0.5×</option><option :value="1">1×</option><option :value="1.5">1.5×</option><option :value="2">2×</option>
    </select></label>
    <p v-if="state.error" role="alert">{{ state.error }}</p>
  </section>
</template>

<style scoped>
.transport { display: flex; flex-wrap: wrap; align-items: center; gap: 16px; padding: 14px 0; }.transport .play-button { min-width: 76px; margin: 0; }.transport .seek-control { flex: 1; min-width: 130px; margin: 0; font-size: 10px; }.transport input[type=range] { padding: 0; width: 100%; border: 0; }.transport output { font: 11px var(--mono); color: #526859; }.transport label { margin: 0; font-size: 10px; }.transport select { min-height: 32px; padding: 5px 8px; }.transport [role=alert] { flex-basis: 100%; }
@media(max-width:680px) { .transport { gap: 12px; }.transport .seek-control { width: auto; }.transport label:last-of-type { width: auto; margin-left: auto; } }
</style>
