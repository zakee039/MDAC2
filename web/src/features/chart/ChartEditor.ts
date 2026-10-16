import type { Chart } from '../../services/projects.ts'
import type { Transport } from '../audio/Transport.ts'

export interface Note extends Record<string, unknown> {
  event_id: string; q: string; duration_q: string; lane_mask: ('ground'|'air')[]
  semantic: 'tap'|'hold'|'gemini'; accent: boolean; locked: boolean
  source_kind: 'manual'|'event'|'candidate'; source_id: string|null
}
export interface EditableChart extends Chart { events: Note[] }
const gcd=(a:bigint,b:bigint):bigint=> b===0n ? (a<0n?-a:a) : gcd(b,a%b)
function parts(value:string):[bigint,bigint] {
  if(typeof value!=='string' || value.length>128 || !/^-?\d+(\/\d+)?$/.test(value)) throw new Error('拍坐标须为规范整数或分数，例如 3/2。')
  const [a,b='1']=value.split('/'), n=BigInt(a!),d=BigInt(b)
  if(d<=0n || gcd(n,d)!==1n || `${n}${d===1n?'':'/'+d}`!==value) throw new Error('拍坐标不是规范分数。')
  return [n,d]
}
export function numeric(value:string) { const [n,d]=parts(value); const result=Number(n)/Number(d); if(!Number.isFinite(result))throw new Error('拍坐标超出范围。');return result }
export function endPosition(note:Note) {
  const [a,b]=parts(note.q),[c,d]=parts(note.duration_q),n=a*d+c*b,den=b*d,g=gcd(n,den)
  return `${n/g}${den/g===1n?'':'/'+(den/g)}`
}
export function validateChart(chart:EditableChart, timingId:string, time:(q:string)=>number) {
  if(chart.schema_version!==1 || chart.timing_revision!==timingId) throw new Error('谱面与时间轴版本不一致。')
  const ids=new Set<string>()
  for(const note of chart.events) {
    if(!note.event_id || ids.has(note.event_id))throw new Error('事件 ID 必须唯一。')
    ids.add(note.event_id)
    if(!note.lane_mask.length || new Set(note.lane_mask).size!==note.lane_mask.length || note.lane_mask.some(l=>!['ground','air'].includes(l)))throw new Error('轨道无效。')
    const duration=numeric(note.duration_q);numeric(note.q)
    if(!['tap','hold','gemini'].includes(note.semantic) || (note.semantic==='hold'?duration<=0:duration!==0))throw new Error('Hold 必须有正时长，其他音符时长为零。')
    if(note.semantic==='gemini' && note.lane_mask.length!==2)throw new Error('Gemini 需要地空双轨。')
    if(typeof note.locked!=='boolean' || typeof note.accent!=='boolean' || !['manual','event','candidate'].includes(note.source_kind) || (note.source_kind==='manual'?note.source_id!==null:!note.source_id))throw new Error('事件来源或标记无效。')
    for(const q of [note.q,endPosition(note)]) if(!Number.isFinite(time(q)) || time(q)<0)throw new Error('音符头尾超出时间映射。')
  }
}
export class ChartEditor {
  private value:EditableChart
  private past:EditableChart[]=[]
  private future:EditableChart[]=[]
  private timingId:string
  private time:(q:string)=>number
  constructor(chart:EditableChart,timingId:string,time:(q:string)=>number) {
    validateChart(chart,timingId,time);this.value=structuredClone(chart);this.timingId=timingId;this.time=time
  }
  snapshot(){return structuredClone(this.value)}
  private apply(next:EditableChart){
    validateChart(next,this.timingId,this.time)
    this.past.push(this.snapshot());this.future=[]
    this.value={...next,revision_id:crypto.randomUUID(),user_patch_ids:[...next.user_patch_ids,crypto.randomUUID()]}
  }
  add(note:Omit<Note,'event_id'>){const next=this.snapshot();next.events.push({...note,event_id:crypto.randomUUID()} as Note);this.apply(next)}
  update(id:string,patch:Partial<Pick<Note,'q'|'duration_q'|'lane_mask'|'semantic'|'accent'>>){
    const next=this.snapshot(),note=next.events.find(n=>n.event_id===id)
    if(!note || note.locked)throw new Error('事件不存在或已锁定，请先显式解锁。')
    Object.assign(note,patch);this.apply(next)
  }
  remove(id:string){
    const next=this.snapshot(),note=next.events.find(n=>n.event_id===id)
    if(!note || note.locked)throw new Error('事件不存在或已锁定，请先显式解锁。')
    next.events=next.events.filter(n=>n.event_id!==id);this.apply(next)
  }
  lock(id:string,locked:boolean){const next=this.snapshot(),note=next.events.find(n=>n.event_id===id);if(!note)throw new Error('事件不存在。');note.locked=locked;this.apply(next)}
  private restore(from:EditableChart[],to:EditableChart[]){
    const target=from.pop();if(!target)return
    to.push(this.snapshot());this.value={...target,revision_id:crypto.randomUUID(),user_patch_ids:[...this.value.user_patch_ids,crypto.randomUUID()]}
  }
  undo(){this.restore(this.past,this.future)}
  redo(){this.restore(this.future,this.past)}
  overlay(seconds:number,window=2){
    return this.value.events.map(note=>({note:structuredClone(note),start:this.time(note.q),end:this.time(endPosition(note))}))
      .filter(item=>item.end>=seconds-window && item.start<=seconds+window)
  }
}
export function subscribeOverlay(editor:ChartEditor,transport:Transport,receive:(items:ReturnType<ChartEditor['overlay']>,seconds:number)=>void){
  return transport.subscribe(state=>receive(editor.overlay(state.position),state.position))
}
