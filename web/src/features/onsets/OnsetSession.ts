import type { JsonApi } from '../timing/AnalysisSession.ts'

export class OnsetSession {
  readonly api:JsonApi
  readonly project:string
  readonly base:string
  job:any=null
  candidate:any=null
  selection:string[]=[]
  diff:any=null
  blocked=false
  submitting=false
  private epoch=0
  private disposed=false
  private committed=false
  private previewPending=false
  constructor(api:JsonApi,project:string,base:string){this.api=api;this.project=project;this.base=base}
  get path(){return `projects/${encodeURIComponent(this.project)}/onset-analysis`}
  get current(){return !this.disposed&&!this.committed&&!!this.candidate&&this.candidate.project_id===this.project&&this.candidate.base_revision===this.base}
  get editable(){return this.current&&!this.blocked&&!this.submitting}
  dispose(){this.disposed=true;this.epoch++;this.diff=null}
  setBlocked(value:boolean){this.blocked=value;if(value){if(this.previewPending)this.epoch++;this.diff=null}}
  async history(){
    const epoch=++this.epoch,result=await this.api.json(this.path)
    if(this.disposed||epoch!==this.epoch)return
    this.job=result.jobs[0]??null
    if(this.job)await this.refresh()
  }
  async start(difficulty:string){
    if(this.disposed||this.blocked||this.submitting||['queued','running'].includes(this.job?.state))throw new Error('请先保存草稿并等待当前作业结束。')
    const epoch=++this.epoch;this.candidate=null;this.diff=null;this.selection=[];this.committed=false
    const job=await this.api.json(this.path,{expected_revision:this.base,difficulty})
    if(!this.disposed&&epoch===this.epoch)this.job=job
  }
  async refresh(){
    if(!this.job||this.disposed)return
    const epoch=++this.epoch,id=this.job.job_id
    const job=await this.api.json('jobs/'+id)
    if(this.disposed||epoch!==this.epoch)return
    this.job={...job,job_id:id}
    if(job.state==='succeeded'){
      const candidate=await this.api.json('jobs/'+id+'/onset-analysis')
      if(this.disposed||epoch!==this.epoch)return
      if(candidate.candidate_id!==this.candidate?.candidate_id){
        this.candidate=candidate;this.selection=[...candidate.selected_ids];this.diff=null
      }
    }
  }
  async cancel(){if(this.job){await this.api.json('jobs/'+this.job.job_id+'/cancel',{run_id:this.job.run_id});await this.refresh()}}
  select(ids:string[]){
    if(!this.editable)throw new Error('旧结果或未保存草稿不能修改选留。')
    const candidates=new Map<string,any>(this.candidate.onsets.candidates.map((c:any)=>[c.candidate_id,c]))
    if(ids.length>2048||new Set(ids).size!==ids.length||ids.some(id=>!candidates.has(id)||candidates.get(id).quantized_q===null))throw new Error('请选择可量化的候选，最多 2048 个。')
    this.selection=[...ids];this.epoch++;this.diff=null
  }
  toggle(id:string,checked:boolean){this.select(checked?[...this.selection,id]:this.selection.filter(i=>i!==id))}
  async preview(){
    if(!this.editable||this.selection.length>2048)throw new Error('请先处理草稿、过期结果或超限选留。')
    const epoch=++this.epoch,id=this.candidate.candidate_id,selected=[...this.selection]
    this.diff=null
    this.previewPending=true
    try{
      const value=await this.api.json(this.path+'/'+id+'/preview',{selected_ids:selected})
      if(!this.disposed&&!this.blocked&&epoch===this.epoch&&id===this.candidate?.candidate_id)this.diff=value
    }finally{this.previewPending=false}
  }
  async adopt(reason:string){
    if(!this.editable||!this.diff||!reason.trim())throw new Error('请先查看当前差异并填写采纳理由。')
    const diff=this.diff,id=this.candidate.candidate_id
    this.submitting=true
    try{
      const result=await this.api.json(this.path+'/'+id+'/adopt',{selected_ids:[...diff.selected_ids],
        preview_fingerprint:diff.fingerprint,actor:'local-user',reason,command_id:crypto.randomUUID()})
      this.committed=true;this.diff=null
      return this.disposed?null:result
    }finally{this.submitting=false}
  }
}
