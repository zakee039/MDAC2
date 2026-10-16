import type {JsonApi} from '../timing/AnalysisSession.ts'
import {fraction} from '../../services/manual.ts'

export class StructureSession {
  readonly api:JsonApi;readonly project:string;readonly base:string
  candidate:any=null;job:any=null;review:any=null;diff:any=null;operation:any=null
  available=false;blocked=false;submitting=false
  private disposed=false;private epoch=0;private pendingReview=false
  constructor(api:JsonApi,project:string,base:string){this.api=api;this.project=project;this.base=base}
  get root(){return `projects/${encodeURIComponent(this.project)}`}
  get path(){return this.root+'/structure-analysis'}
  get current(){return !this.disposed&&this.candidate?.project_id===this.project&&this.candidate?.base_revision===this.base}
  get editable(){return this.current&&!this.blocked&&!this.submitting}
  dispose(){this.disposed=true;this.epoch++;this.review=null;this.diff=null}
  setBlocked(value:boolean){this.blocked=value;if(value){if(this.pendingReview)this.epoch++;this.diff=null}}
  invalidateReview(){this.epoch++;this.review=null;this.diff=null}
  invalidateDiff(){this.epoch++;this.diff=null}
  async history(){
    const epoch=++this.epoch
    const [capability,history,operation]=await Promise.all([this.api.json('structure/capabilities'),this.api.json(this.path),this.api.json(this.root+'/structure-operation')])
    if(this.disposed||epoch!==this.epoch)return
    this.available=capability.available;this.operation=operation.revision===this.base?operation:null;this.job=history.jobs[0]??null
    if(this.job)await this.refresh()
  }
  async start(){
    if(!this.available||this.blocked||this.submitting||this.disposed||['queued','running'].includes(this.job?.state))throw Error('请先保存草稿，并确认分析环境可用、当前作业已结束。')
    const epoch=++this.epoch;this.candidate=null;this.review=null;this.diff=null
    const job=await this.api.json(this.path,{expected_revision:this.base})
    if(!this.disposed&&epoch===this.epoch)this.job=job
  }
  async refresh(){
    if(!this.job||this.disposed)return
    const epoch=++this.epoch,id=this.job.job_id,job=await this.api.json('jobs/'+this.job.job_id)
    if(this.disposed||epoch!==this.epoch)return
    this.job={...job,job_id:id}
    if(job.state==='succeeded'){
      const candidate=await this.api.json('jobs/'+id+'/structure-analysis')
      if(this.disposed||epoch!==this.epoch)return
      if(candidate.candidate_id!==this.candidate?.candidate_id){this.candidate=candidate;this.review=null;this.diff=null}
    }
  }
  async cancel(){if(this.job){await this.api.json('jobs/'+this.job.job_id+'/cancel',{run_id:this.job.run_id});await this.refresh()}}
  async confirm(value:any){
    if(!this.editable)throw Error('请使用当前谱面的候选，并先保存草稿。')
    const bounds=(s:any)=>({start_q:fraction(s.start),end_q:fraction(s.end)})
    const epoch=++this.epoch;this.review=null;this.diff=null;this.pendingReview=true
    try{
      const result=await this.api.json(this.path+'/'+this.candidate.candidate_id+'/confirm',{
        primary:bounds(value.primary),targets:value.targets.map((t:any)=>({proposal_id:t.id,span:bounds(t.span),relation:t.relation,allowed_modes:[...t.modes],confirmed:t.confirmed})),actor:value.actor,reason:value.reason})
      if(!this.disposed&&!this.blocked&&epoch===this.epoch)this.review=result
    }finally{this.pendingReview=false}
  }
  async preview(options:any){
    if(!this.editable||!this.review)throw Error('请先确认当前关系和复用许可。')
    const epoch=++this.epoch,review=this.review.review_id,copy=JSON.parse(JSON.stringify(options))
    this.diff=null;this.pendingReview=true
    try{
      const result=await this.api.json(this.root+'/structure-reviews/'+review+'/preview',copy)
      if(!this.disposed&&!this.blocked&&epoch===this.epoch&&this.review?.review_id===review)this.diff={...result,options:copy}
    }finally{this.pendingReview=false}
  }
  async apply(confirmFull=false){
    if(!this.editable||!this.review||!this.diff||this.diff.review_id!==this.review.review_id||this.diff.conflicts.length)throw Error('请先检查当前无冲突的差异预览。')
    if(this.diff.options.mode==='full_pattern'&&!confirmFull)throw Error('完整替换需要明确确认目标音符的增删。')
    this.submitting=true
    try{
      const result=await this.api.json(this.root+'/structure-reviews/'+this.review.review_id+'/apply',{
        options:JSON.parse(JSON.stringify(this.diff.options)),preview_fingerprint:this.diff.fingerprint,command_id:crypto.randomUUID()})
      this.diff=null;this.review=null;return this.disposed?null:result
    }finally{this.submitting=false}
  }
  async undo(){
    if(this.disposed||this.blocked||this.submitting||!this.operation?.operation_id||this.operation.revision!==this.base)throw Error('仅能撤销未再编辑的本次复用。')
    this.submitting=true
    try{
      const result=await this.api.json(this.root+'/structure-operations/'+this.operation.operation_id+'/undo',{expected_revision:this.base,command_id:crypto.randomUUID()})
      this.operation=null;return this.disposed?null:result
    }finally{this.submitting=false}
  }
}
