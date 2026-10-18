import WaveSurfer from 'wavesurfer.js'
import Regions from 'wavesurfer.js/dist/plugins/regions.esm.js'
import type { WaveformFactory } from './Waveform'

export function wavesurferFactory(container: HTMLElement): WaveformFactory {
  return (options) => {
    if (!(options.media instanceof HTMLMediaElement)) throw new Error('必须共享 Transport 的 HTML 音源')
    const regions = Regions.create()
    const waveform = WaveSurfer.create({ container, media: options.media, peaks: options.peaks,
      duration: options.duration, height: 100, interact: false, dragToSeek: false, autoplay: false,
      waveColor: '#80b9a6', progressColor: '#28876e', cursorWidth: 0, plugins: [regions] })
    const cursor = document.createElement('div')
    cursor.setAttribute('aria-hidden', 'true')
    cursor.style.cssText = 'position:absolute;top:0;bottom:0;width:2px;background:#b88635;pointer-events:none;z-index:5'
    container.style.position = 'relative'
    container.append(cursor)
    const click = (event: MouseEvent) => {
      if (event.composedPath().some((element) => element instanceof HTMLElement && element.getAttribute('part')?.includes('region'))) return
      const bounds = container.getBoundingClientRect()
      if (bounds.width > 0) options.onSeek(Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)) * options.duration)
    }
    container.addEventListener('click', click)
    let updating = false
    const offRegion = regions.on('region-updated', (region) => {
      if (!updating) options.onRegion({ id: region.id, start: region.start, end: region.end })
    })
    const offError = waveform.on('error', () => options.onError('波形渲染失败'))
    return {
      cursor(seconds) { cursor.style.left = `${100 * seconds / options.duration}%` },
      regions(items) {
        updating = true
        try {
          regions.clearRegions()
          for (const region of items) regions.addRegion({ ...region, drag: true, resize: true, color: '#6cac8644' })
        } finally { updating = false }
      },
      destroy() {
        offRegion(); offError()
        container.removeEventListener('click', click)
        cursor.remove()
        waveform.destroy() // external media is not owned/destroyed by WaveSurfer
      },
    }
  }
}
