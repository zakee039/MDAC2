import type { MediaClock, Transport } from './Transport'

export interface AudioRegion { id: string; start: number; end: number }
export interface WaveformView {
  cursor(seconds: number): void
  regions(regions: readonly AudioRegion[]): void
  destroy(): void
}
export interface WaveformOptions {
  media: MediaClock
  peaks: number[][]
  duration: number
  onSeek(seconds: number): void
  onRegion(region: AudioRegion): void
  onError(message: string): void
}
export type WaveformFactory = (options: WaveformOptions) => WaveformView

export class WaveformController {
  private view: WaveformView
  private unsubscribe: () => void
  private disposed = false
  private duration: number
  constructor(transport: Transport, peaks: readonly (readonly number[])[], duration: number,
    factory: WaveformFactory, onRegion: (region: AudioRegion) => void, onError: (message: string) => void) {
    if (!Number.isFinite(duration) || duration <= 0 || peaks.length < 1 || peaks.length > 2 ||
      peaks.some((channel) => channel.length === 0 || channel.length > 200000 || channel.some((n) => !Number.isFinite(n) || Math.abs(n) > 1))) {
      throw new Error('无效预计算波形，禁止隐式解码大文件')
    }
    this.duration = duration
    this.view = factory({ media: transport.clock, peaks: peaks.map((channel) => [...channel]), duration,
      onSeek: (seconds) => { if (!this.disposed) { try { transport.seek(seconds) } catch { onError('波形位置无效') } } },
      onRegion: (region) => { if (!this.disposed) { try { this.validate(region); onRegion({ ...region }) } catch { onError('区域边界无效') } } },
      onError })
    this.unsubscribe = transport.subscribe((state) => this.view.cursor(state.position))
  }

  private validate(region: AudioRegion) {
    if (!region.id || !Number.isFinite(region.start) || !Number.isFinite(region.end) ||
      region.start < 0 || region.end <= region.start || region.end > this.duration) throw new Error('无效区域')
  }

  setRegions(regions: readonly AudioRegion[]) {
    if (this.disposed) throw new Error('波形已卸载')
    if (new Set(regions.map((region) => region.id)).size !== regions.length) throw new Error('重复区域 ID')
    regions.forEach((region) => this.validate(region))
    this.view.regions(regions.map((region) => ({ ...region })))
  }

  destroy() {
    if (this.disposed) return
    this.disposed = true
    this.unsubscribe()
    this.view.destroy()
  }
}
