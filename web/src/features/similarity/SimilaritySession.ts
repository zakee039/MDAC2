import {secondsAt,validateTiming} from '../timing/TimingEditor.ts'
import type {Timing} from '../timing/TimingEditor.ts'
export interface Span {start:number;end:number}
export type Relation='unknown'|'exact_repeat'|'variation'|'rhythmic_similar'
export type ReuseMode='arrangement'|'mirror'|'full_pattern'
export interface Proposal {id:string;fingerprint:string;timingFingerprint:string;source:Span;target:Span;scores:Record<string,number>;warnings:string[]}
export interface Target {id:string;proposalFingerprint:string;span:Span;enabled:boolean;relation:Relation;modes:ReuseMode[];confirmed:boolean}
const copy=<T>(v:T):T=>JSON.parse(JSON.stringify(v))
const overlap=(a:Span,b:Span)=>a.start<b.end&&b.start<a.end
export class SimilaritySession {
  readonly timing:Timing
  readonly fingerprint:string
  readonly proposals:Proposal[]
  primary:Span
  targets:Target[]
  stale=false
  constructor(timing:Timing,fingerprint:string,proposals:Proposal[]){
    validateTiming(timing)
    if(timing.review!=='accepted'||!fingerprint||!proposals.length||new Set(proposals.map(p=>p.id)).size!==proposals.length)throw new Error('需要已确认时间轴与有效候选。')
    this.timing=copy(timing);this.fingerprint=fingerprint;this.proposals=copy(proposals);this.primary=copy(proposals[0]!.source)
    if(proposals.some(p=>p.timingFingerprint!==fingerprint||p.source.start!==this.primary.start||p.source.end!==this.primary.end||!p.fingerprint))throw new Error('候选必须使用同一主段和当前时间轴。')
    this.targets=proposals.map(p=>({id:p.id,proposalFingerprint:p.fingerprint,span:copy(p.target),enabled:false,relation:'unknown',modes:[],confirmed:false}))
    this.check(this.primary);this.targets.forEach(t=>this.check(t.span))
  }
  check(span:Span){if(!Number.isFinite(span.start)||!Number.isFinite(span.end)||span.end<=span.start)throw new Error('片段边界必须递增。');secondsAt(this.timing,span.start);secondsAt(this.timing,span.end)}
  invalidate(fingerprint:string){if(fingerprint!==this.fingerprint){this.stale=true;this.targets.forEach(t=>t.confirmed=false)}}
  private target(id:string){const t=this.targets.find(t=>t.id===id);if(!t)throw new Error('目标不存在。');return t}
  edit(id:string|'primary',span:Span){
    this.check(span)
    if(id==='primary'){this.primary=copy(span);this.targets.forEach(t=>t.confirmed=false)}
    else {const t=this.target(id);t.span=copy(span);t.confirmed=false}
  }
  fromSeconds(start:number,end:number):Span{
    const q=(seconds:number)=>{
      const a=this.timing.anchors
      if(!Number.isFinite(seconds)||seconds<a[0]!.t||seconds>a.at(-1)!.t)throw new Error('秒位置超出时间映射。')
      let i=0;while(i<a.length-2&&a[i+1]!.t<seconds)i++
      return a[i]!.q+(seconds-a[i]!.t)*(a[i+1]!.q-a[i]!.q)/(a[i+1]!.t-a[i]!.t)
    }
    const span={start:q(start),end:q(end)};this.check(span);return span
  }
  configure(id:string,enabled:boolean,relation:Relation,modes:ReuseMode[]){
    if(!['unknown','exact_repeat','variation','rhythmic_similar'].includes(relation)||new Set(modes).size!==modes.length||modes.some(m=>!['arrangement','mirror','full_pattern'].includes(m))||relation==='unknown'&&modes.length||modes.includes('full_pattern')&&relation!=='exact_repeat')throw new Error('当前关系不允许所选复用方式。')
    Object.assign(this.target(id),{enabled,relation,modes:[...modes],confirmed:false})
  }
  confirm(id:string){
    if(this.stale)throw new Error('时间轴已变化，请重新检测和确认。')
    const t=this.target(id)
    if(!t.enabled||t.relation==='unknown'||overlap(this.primary,t.span)||this.targets.some(other=>other!==t&&other.enabled&&overlap(other.span,t.span)))throw new Error('请选择关系，并消除主段和目标之间的重叠。')
    t.confirmed=true
  }
  listen(id:string|'primary'):[number,number]{
    if(this.stale)throw new Error('时间轴已变化，请重新打开候选。')
    const span=id==='primary'?this.primary:this.target(id).span
    return [secondsAt(this.timing,span.start),secondsAt(this.timing,span.end)]
  }
  confirmation(actor:string,reason:string){
    const targets=this.targets.filter(t=>t.enabled)
    if(this.stale||!actor.trim()||!reason.trim()||!targets.length||targets.some(t=>!t.confirmed))throw new Error('请逐个确认启用的目标，并填写听检结论。')
    // Recheck all relations even if caller mutated a public UI field.
    targets.forEach(t=>this.confirm(t.id))
    return copy({primary:this.primary,targets,timingFingerprint:this.fingerprint,actor,reason,review:'confirmed'})
  }
}
