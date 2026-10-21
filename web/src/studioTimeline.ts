import type {BeatSettings} from './studioBpm'
export interface Trim {start_ms:number;end_ms:number}
export interface TempoPoint {start_ms:number;bpm:number;offset_ms:number;tap_latency_ms:number}
// Stored markers are rounded to 1 ms; decoded media duration is more precise.
// Include that rounding margin at the viewport edges and draw on the boundary.
export function markerPosition(time:number,start:number,end:number):number|null{
 const tolerance=.000500001
 if(![time,start,end].every(Number.isFinite)||time<start-tolerance||time>end+tolerance)return null
 return Math.max(0,Math.min(100,100*(time-start)/Math.max(.001,end-start)))
}
export function rebaseTiming(timing:BeatSettings,oldStart:number,newStart:number):BeatSettings{
 const delta=newStart-oldStart
 return {...timing,offset_ms:timing.offset_ms-delta,...(timing.segments?{segments:timing.segments.map(s=>({...s,start_ms:s.start_ms-delta,offset_ms:s.offset_ms-delta}))}:{})}
}
export function tempoAt(timing:BeatSettings,timeMs:number){return [...(timing.segments||[])].reverse().find(s=>s.start_ms<=timeMs)||timing}
export function visibleSegments(timing:BeatSettings,durationMs:number){
 const points=timing.segments||[],prior=points.reduce((last,s,i)=>s.start_ms<=0?i:last,-1)
 return [{index:prior,start_ms:0,end_ms:points.find(s=>s.start_ms>0)?.start_ms??durationMs,value:prior<0?timing:points[prior]!},
 ...points.flatMap((s,index)=>s.start_ms>0&&s.start_ms<durationMs?[{index,start_ms:s.start_ms,end_ms:Math.min(durationMs,points[index+1]?.start_ms??durationMs),value:s}]:[])]
}
export function fitSegmentBeats(times:number[],startMs:number,endMs:number){
 const selected=times.filter(t=>t*1000>=startMs&&t*1000<endMs)
 if(selected.length<4)throw new Error('此段有效拍点不足4个，请手打或手动填写')
 const gaps=selected.slice(1).map((t,i)=>t-selected[i]!).sort((a,b)=>a-b),period=gaps[Math.floor(gaps.length/2)]!
 const bpm=60/period
 if(bpm<30||bpm>400)throw new Error('BPM超出30–400，请手动填写')
 return {bpm:Math.round(bpm*1000000)/1000000,offset_ms:Math.round(selected[0]!*1000)}
}
