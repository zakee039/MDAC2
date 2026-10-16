<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { Transport } from './Transport'
import { WaveformController } from './Waveform'
import type { AudioRegion } from './Waveform'
import { wavesurferFactory } from './wavesurferAdapter'
const props = defineProps<{ transport: Transport; peaks: number[][]; duration: number; regions: AudioRegion[] }>()
const emit = defineEmits<{ region: [value: AudioRegion] }>()
const container = ref<HTMLElement>()
const error = ref('')
let controller: WaveformController | undefined
function mount() {
  controller?.destroy()
  controller = undefined
  error.value = ''
  if (!container.value) return
  try {
    controller = new WaveformController(props.transport, props.peaks, props.duration, wavesurferFactory(container.value),
      (region) => emit('region', region), (message) => { error.value = message })
    controller.setRegions(props.regions)
  } catch { error.value = '无法载入波形' }
}
onMounted(mount)
watch(() => [props.transport, props.peaks, props.duration], mount)
watch(() => props.regions, (regions) => {
  try { controller?.setRegions(regions) } catch { error.value = '区域边界无效' }
}, { deep: true })
onBeforeUnmount(() => controller?.destroy())
</script>
<template><section aria-label="音频波形"><div ref="container" /><p v-if="error" role="alert">{{ error }}</p></section></template>
