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
  <section aria-label="音频播放控制">
    <button type="button" @click="toggle">{{ state.paused ? '播放' : '暂停' }}</button>
    <label>播放位置 <input type="range" min="0" :max="state.duration" step="0.01" :value="state.position"
      :disabled="state.duration === 0" @input="transport.seek(Number(($event.target as HTMLInputElement).value))" /></label>
    <output>{{ state.position.toFixed(2) }} / {{ state.duration.toFixed(2) }} 秒</output>
    <label>试听速度 <select :value="state.rate" @change="transport.setRate(Number(($event.target as HTMLSelectElement).value))">
      <option :value="0.5">0.5×</option><option :value="1">1×</option><option :value="1.5">1.5×</option><option :value="2">2×</option>
    </select></label>
    <p v-if="state.error" role="alert">{{ state.error }}</p>
  </section>
</template>
