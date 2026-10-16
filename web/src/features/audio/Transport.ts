export interface MediaClock {
  currentTime: number
  duration: number
  paused: boolean
  playbackRate: number
  play(): Promise<void>
  pause(): void
  addEventListener(name: string, callback: EventListener): void
  removeEventListener(name: string, callback: EventListener): void
}

export interface FrameScheduler {
  request(callback: FrameRequestCallback): number
  cancel(id: number): void
}

export interface TransportState {
  position: number
  duration: number
  paused: boolean
  rate: number
  loop: readonly [number, number] | null
  error: string | null
}

export class Transport {
  private media: MediaClock
  private scheduler: FrameScheduler
  private listeners = new Set<(state: TransportState) => void>()
  private frame: number | null = null
  private loop: readonly [number, number] | null = null
  private disposed = false
  private generation = 0
  private wantsPlaying = false
  private error: string | null = null
  private events = ['play', 'pause', 'timeupdate', 'seeked', 'durationchange', 'ended', 'ratechange', 'error']
  private onMedia: EventListener = (event) => {
    if (event.type === 'error') this.error = '音频播放失败'
    if (this.loop && this.loop[1] > this.snapshot().duration) this.loop = null
    if (event.type === 'ended' && this.loop) {
      this.media.currentTime = this.loop[0]
      void this.play().catch(() => {})
    }
    this.sync()
  }

  constructor(media: MediaClock, scheduler: FrameScheduler = {
    request: (callback) => requestAnimationFrame(callback), cancel: (id) => cancelAnimationFrame(id),
  }) {
    this.media = media
    this.scheduler = scheduler
    for (const event of this.events) media.addEventListener(event, this.onMedia)
  }

  snapshot(): TransportState {
    const duration = Number.isFinite(this.media.duration) ? Math.max(0, this.media.duration) : 0
    return { position: Math.min(duration, Math.max(0, this.media.currentTime || 0)), duration,
      paused: this.media.paused, rate: this.media.playbackRate,
      loop: this.loop ? [...this.loop] : null, error: this.error }
  }

  get clock(): MediaClock { return this.media }

  subscribe(listener: (state: TransportState) => void): () => void {
    this.assertLive()
    this.listeners.add(listener)
    listener(this.snapshot())
    return () => this.listeners.delete(listener)
  }

  private assertLive() {
    if (this.disposed) throw new Error('Transport 已销毁')
  }

  private sync() {
    if (this.disposed) return
    if (this.frame !== null) this.scheduler.cancel(this.frame)
    this.frame = null
    if (this.loop && !this.media.paused && this.media.currentTime >= this.loop[1]) {
      this.media.currentTime = this.loop[0] + (this.media.currentTime - this.loop[1]) % (this.loop[1] - this.loop[0])
    }
    const state = this.snapshot()
    for (const listener of this.listeners) listener(state)
    if (!this.media.paused) this.frame = this.scheduler.request(() => { this.frame = null; this.sync() })
  }

  async play() {
    this.assertLive()
    const generation = ++this.generation
    this.wantsPlaying = true
    this.error = null
    try {
      await this.media.play()
      if (this.disposed || generation !== this.generation) {
        if (this.disposed || !this.wantsPlaying) this.media.pause()
        return
      }
      this.sync()
    } catch (error) {
      if (generation === this.generation && !this.disposed) {
        this.error = '无法开始播放，请检查音源或浏览器权限'
        this.sync()
      }
      throw error
    }
  }

  pause() {
    this.assertLive()
    ++this.generation
    this.wantsPlaying = false
    this.media.pause()
    this.sync()
  }

  seek(seconds: number) {
    this.assertLive()
    if (!Number.isFinite(seconds) || seconds < 0 || seconds > this.snapshot().duration) throw new Error('无效播放位置')
    this.media.currentTime = seconds
    this.sync()
  }

  setLoop(bounds: readonly [number, number] | null) {
    this.assertLive()
    if (bounds && (!bounds.every(Number.isFinite) || bounds[0] < 0 || bounds[1] <= bounds[0] || bounds[1] > this.snapshot().duration)) {
      throw new Error('无效循环区间')
    }
    this.loop = bounds ? [...bounds] : null
    this.sync()
  }

  setRate(rate: number) {
    this.assertLive()
    if (!Number.isFinite(rate) || rate < .25 || rate > 2) throw new Error('试听速度范围为 0.25–2 倍')
    this.media.playbackRate = rate
    this.sync()
  }

  destroy() {
    if (this.disposed) return
    this.disposed = true
    this.wantsPlaying = false
    ++this.generation
    this.media.pause()
    if (this.frame !== null) this.scheduler.cancel(this.frame)
    for (const event of this.events) this.media.removeEventListener(event, this.onMedia)
    this.listeners.clear()
  }
}
