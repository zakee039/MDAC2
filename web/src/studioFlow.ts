export type FlowProject={draft?:unknown;confirmed?:boolean;generation?:unknown;export?:unknown;active_attempt?:string;difficulty?:{generation:string;chart_sha256:string;algorithm:string;value:string};difficulty_prediction?:{generation:string;chart_sha256:string;algorithm:string};attempts?:{attempt_id:string;operation:string;state:string}[]}
export function ratingConfirmed(p:FlowProject|null){
 const r=p?.difficulty,q=p?.difficulty_prediction
 return !!r&&!!q&&r.generation===p?.generation&&r.generation===q.generation&&r.chart_sha256===q.chart_sha256&&r.algorithm===q.algorithm
}
export function projectStage(p:FlowProject){
 if(p.draft)return 2
 const attempt=p.attempts?.find(a=>a.attempt_id===p.active_attempt)
 if(attempt)return attempt.operation==='export'?4:3
 if(p.export)return 4
 // A failed/cancelled package remains on Export with a retry action.
 if(ratingConfirmed(p)&&p.attempts?.at(-1)?.operation==='export')return 4
 return p.confirmed?3:2
}
export function exportAction(p:FlowProject|null,dirty:boolean){
 if(!p||dirty)return 'blocked'
 if(p.active_attempt)return p.attempts?.find(a=>a.attempt_id===p.active_attempt)?.operation==='export'?'show':'blocked'
 return p.export?'show':p.generation&&ratingConfirmed(p)?'build':'blocked'
}
