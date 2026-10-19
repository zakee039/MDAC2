export type ProjectStamp={project_id:string;version:number}
export function acceptsLLMResult(current:ProjectStamp|null,started:ProjectStamp,view:number,startedView:number,disposed:boolean,requireVersion=true){
 return !disposed&&view===startedView&&current?.project_id===started.project_id&&(!requireVersion||current.version===started.version)
}
export function mergeSuggestedFields<T extends {title:string;author:string;scene:string}>(current:T,started:T,suggestion:T){
 const next={...current}
 for(const key of ['title','author','scene'] as const)if(current[key]===started[key])next[key]=suggestion[key]
 return next
}
