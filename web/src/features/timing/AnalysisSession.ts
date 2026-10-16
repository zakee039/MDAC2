export interface JsonApi { json(path:string,body?:unknown):Promise<any> }
export function reviewWindows(candidate:any): {label:string;start:number;end:number}[] {
  const audio=candidate.observation.payload.audio,duration=audio.sample_count/audio.sample_rate
  const beats:number[]=candidate.observation.payload.beat_times
  const residual:number[]=candidate.fit.residual_seconds
  const worst=residual.reduce((best,v,i)=>Math.abs(v)>Math.abs(residual[best]??0)?i:best,0)
  const centers=[{label:'开头与弱起',time:beats[0]??0},{label:'最大拟合偏差',time:beats[worst]??duration/2},{label:'尾部与静音',time:beats.at(-1)??duration}]
  return centers.map(({label,time})=>({label,start:Math.max(0,time-2),end:Math.min(duration,time+3)})).filter(w=>w.end>w.start)
}
export class AnalysisSession {
  readonly api:JsonApi
  readonly project:string
  readonly base:string|null
  job:any=null
  candidate:any=null
  migration:any=null
  private generation=0
  private disposed=false
  constructor(api:JsonApi,project:string,base:string|null){this.api=api;this.project=project;this.base=base}
  get path(){return `projects/${encodeURIComponent(this.project)}/timing-analysis`}
  get current(){return !!this.candidate && this.candidate.project_id===this.project && this.candidate.base_revision===this.base}
  get adoptable(){return this.current && !!this.candidate.fit.accepted && !!this.candidate.timing}
  dispose(){this.disposed=true;this.generation++}
  async history(){
    const generation=++this.generation,result=await this.api.json(this.path)
    if(this.disposed||generation!==this.generation)return
    this.job=result.jobs[0]??null
    if(this.job)await this.refresh()
  }
  async start(options:any){
    const generation=++this.generation
    this.candidate=null;this.migration=null
    const job=await this.api.json(this.path,{...options,expected_revision:this.base})
    if(!this.disposed&&generation===this.generation)this.job=job
  }
  async refresh(){
    if(!this.job)return
    const generation=++this.generation,id=this.job.job_id
    const job=await this.api.json('jobs/'+id)
    if(this.disposed||generation!==this.generation)return
    this.job={...job,job_id:id}
    if(job.state==='succeeded'){
      const candidate=await this.api.json('jobs/'+id+'/timing-analysis')
      if(!this.disposed&&generation===this.generation){this.candidate=candidate;this.migration=null}
    }
  }
  async cancel(){if(this.job){await this.api.json('jobs/'+this.job.job_id+'/cancel',{run_id:this.job.run_id});await this.refresh()}}
  async preview(policy:string){
    if(!this.adoptable)throw new Error('候选已过期或未通过拟合，请重新分析。')
    const value=await this.api.json(this.path+'/'+this.candidate.candidate_id+'/preview',{position_policy:policy})
    if(!this.disposed)this.migration={...value,policy}
  }
  async adopt(reason:string){
    if(!this.adoptable||!this.migration||this.migration.conflicts.length||!reason.trim())throw new Error('请先听检、填写理由并确认无冲突的迁移预览。')
    return this.api.json(this.path+'/'+this.candidate.candidate_id+'/adopt',{
      position_policy:this.migration.policy,preview_fingerprint:this.migration.fingerprint,
      actor:'local-user',reason,command_id:crypto.randomUUID()})
  }
}
