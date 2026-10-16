// View-model coordinates are finite numbers. Persistent rational serialization
// and server-side edit records are supplied by the workflow integration.
export interface Anchor { q: number; t: number }
export interface Meter { start: number; numerator: number; denominator: number }
export interface Timing {
  id: string; duration: number; anchors: Anchor[]; downbeats: number[]; meters: Meter[]
  review: 'pending' | 'accepted' | 'rejected'
  evidence: { start: number; end: number; score: number; reason: string }[]
}
export type TimingEdit = { type: 'insert' | 'drag'; q: number; t: number }
  | { type: 'delete'; q: number } | { type: 'downbeat'; q: number; enabled: boolean }
  | { type: 'meter'; meters: Meter[] }
const finite = (n: number) => { if (!Number.isFinite(n)) throw new Error('坐标必须是有限数字。') }
function meterAt(timing: Timing, q: number) { return timing.meters.filter(m => m.start <= q).at(-1) ?? timing.meters[0]! }
const boundary = (q: number, meter: Meter) => Math.abs((q - meter.start) / (meter.numerator * 4 / meter.denominator) - Math.round((q - meter.start) / (meter.numerator * 4 / meter.denominator))) < 1e-9
export function validateTiming(timing: Timing) {
  finite(timing.duration)
  if (timing.duration <= 0 || timing.anchors.length < 2 || !timing.meters.length) throw new Error('至少需要两个锚点和一个拍号。')
  timing.anchors.forEach((a,i) => {
    finite(a.q); finite(a.t)
    const previous = timing.anchors[i-1]
    if (a.t < 0 || a.t > timing.duration || (previous && (a.q <= previous.q || a.t <= previous.t))) throw new Error('锚点必须按拍和秒严格递增，且位于音频内。')
  })
  timing.meters.forEach((m,i) => {
    finite(m.start)
    if (!Number.isSafeInteger(m.numerator) || m.numerator <= 0 || !Number.isSafeInteger(m.denominator) || m.denominator <= 0 || !Number.isInteger(Math.log2(m.denominator))) throw new Error('拍号无效。')
    const previous=timing.meters[i-1]
    if(previous && (m.start<=previous.start || !boundary(m.start,previous))) throw new Error('变拍号必须位于递增的小节边界。')
  })
  timing.downbeats.forEach((q,i) => {
    if (!timing.anchors.some(a=>a.q===q) || !boundary(q,meterAt(timing,q)) || (i>0 && q<=timing.downbeats[i-1]!)) throw new Error('强拍必须引用小节边界上的锚点。')
  })
  for(const e of timing.evidence) {
    if (![e.start,e.end,e.score].every(Number.isFinite) || e.start < 0 || e.end <= e.start || e.end > timing.duration || e.score < 0 || e.score > 1) throw new Error('分段证据无效。')
  }
}
export function secondsAt(timing: Timing, q: number): number {
  finite(q)
  const anchors=timing.anchors
  if(q<anchors[0]!.q || q>anchors.at(-1)!.q) throw new Error('超出已确认的时间映射。')
  let index=0
  while(index<anchors.length-2 && anchors[index+1]!.q<q) index++
  const a=anchors[index]!, b=anchors[index+1]!
  return a.t+(q-a.q)*(b.t-a.t)/(b.q-a.q)
}
export class TimingEditor {
  private value: Timing
  private pending: { base: string; value: Timing; operation: string } | null = null
  private past: Timing[] = []
  private future: Timing[] = []
  readonly taps: { seconds: number; latency: number }[] = []
  constructor(timing: Timing) { validateTiming(timing); this.value=structuredClone(timing) }
  snapshot() { return structuredClone(this.value) }
  preview() { return this.pending ? structuredClone(this.pending.value) : null }
  propose(edit: TimingEdit) {
    this.pending=null
    const next=this.snapshot(), index='q' in edit ? next.anchors.findIndex(a=>a.q===edit.q) : -1
    if(edit.type==='insert') {
      if(index>=0) throw new Error('此拍已有锚点。')
      next.anchors.push({q:edit.q,t:edit.t}); next.anchors.sort((a,b)=>a.q-b.q)
    } else if(edit.type==='drag') {
      if(index<0) throw new Error('锚点不存在。')
      next.anchors[index]!.t=edit.t
    } else if(edit.type==='delete') {
      if(index<0 || next.downbeats.includes(edit.q)) throw new Error('请先取消强拍标记；锚点必须存在。')
      next.anchors.splice(index,1)
    } else if(edit.type==='downbeat') {
      if(index<0) throw new Error('锚点不存在。')
      next.downbeats=next.downbeats.filter(q=>q!==edit.q)
      if(edit.enabled) next.downbeats.push(edit.q)
      next.downbeats.sort((a,b)=>a-b)
    } else if(edit.type==='meter') { next.meters=structuredClone(edit.meters) }
    validateTiming(next)
    this.pending={base:JSON.stringify(this.value),value:next,operation:edit.type}
  }
  cancel() { this.pending=null }
  commit(expectedId: string) {
    if(!this.pending || expectedId!==this.value.id || this.pending.base!==JSON.stringify(this.value)) throw new Error('预览已过期，请重新预览。')
    this.past.push(this.snapshot()); this.future=[]
    this.value={...this.pending.value,id:crypto.randomUUID(),review:'pending'}; this.pending=null
  }
  undo() {
    const prior=this.past.pop(); if(!prior) return
    this.future.push(this.snapshot()); this.value={...prior,id:crypto.randomUUID(),review:'pending'}; this.pending=null
  }
  redo() {
    const next=this.future.pop(); if(!next) return
    this.past.push(this.snapshot()); this.value={...next,id:crypto.randomUUID(),review:'pending'}; this.pending=null
  }
  tap(seconds: number, latency: number) {
    finite(seconds); finite(latency)
    if(seconds<0 || seconds>this.value.duration || Math.abs(latency)>1) throw new Error('Tap 或延迟超出范围。')
    this.taps.push({seconds,latency})
  }
}
