import { ApiError } from './projects.ts'
import type { Timing } from '../features/timing/TimingEditor.ts'
export function fraction(value: number): string {
  if(!Number.isFinite(value))throw new Error('非有限坐标。')
  const [mantissa,exp='0']=String(value).split('e'),[whole,tail='']=mantissa!.split('.')
  let n=BigInt(whole!+tail),d=10n**BigInt(tail.length)
  const exponent=Number(exp)
  if(exponent>=0)n*=10n**BigInt(exponent);else d*=10n**BigInt(-exponent)
  let a=n<0n?-n:n,b=d;while(b){const r=a%b;a=b;b=r}
  n/=a;d/=a;return `${n}${d===1n?'':'/'+d}`
}
export const numberOf=(q:string)=>{const [n,d='1']=q.split('/');return Number(n)/Number(d)}
export function multiply(a:string,b:string,divisor=1):string {
  const [an,ad='1']=a.split('/'),[bn,bd='1']=b.split('/')
  let n=BigInt(an!)*BigInt(bn!),d=BigInt(ad)*BigInt(bd)*BigInt(divisor),x=n<0n?-n:n,y=d
  while(y){const r=x%y;x=y;y=r}n/=x;d/=x;return `${n}${d===1n?'':'/'+d}`
}
export function viewTiming(data:any):Timing {
  return {id:data.revision_id,duration:numberOf(data.mapping.audio_duration),
    anchors:data.mapping.anchors.map((a:any)=>({q:numberOf(a.q),t:numberOf(a.t)})),
    meters:data.mapping.meter_segments.map((m:any)=>({start:numberOf(m.start_q),numerator:m.numerator,denominator:m.denominator})),
    downbeats:data.downbeat_q.map(numberOf),review:data.review,evidence:[]}
}
export function storedTiming(view:Timing,source:any):any {
  return {revision_id:view.id,review:'pending',parent_fingerprint:null,downbeat_q:view.downbeats.map(q=>source.downbeat_q.find((v:string)=>numberOf(v)===q)??fraction(q)),
    mapping:{...source.mapping,anchors:view.anchors.map(a=>{
      const old=source.mapping.anchors.find((v:any)=>numberOf(v.q)===a.q)
      return {q:old?.q??fraction(a.q),t:old&&numberOf(old.t)===a.t?old.t:fraction(a.t)}
    }),meter_segments:view.meters.map(m=>({start_q:source.mapping.meter_segments.find((v:any)=>numberOf(v.start_q)===m.start)?.start_q??fraction(m.start),numerator:m.numerator,denominator:m.denominator}))}}
}
export class ManualApi {
  private token:()=>string
  constructor(token:()=>string){this.token=token}
  async request(path:string,body?:unknown,raw=false):Promise<any>{
    let response:Response
    try{response=await fetch('/api/v1/'+path,{method:body===undefined?'GET':'POST',headers:{Authorization:`Bearer ${this.token()}`,...(body===undefined?{}:{'Content-Type':raw?'application/octet-stream':'application/json'})},body:body===undefined?undefined:raw?body as Blob:JSON.stringify(body),redirect:'error',cache:'no-store'})}catch{throw new ApiError(0)}
    if(!response.ok)throw new ApiError(response.status)
    return response
  }
  async json(path:string,body?:unknown){return (await this.request(path,body)).json()}
}
