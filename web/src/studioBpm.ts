export interface BeatSettings {enabled:boolean;bpm:number;offset_ms:number;tap_latency_ms:number;subdivisions:number}
export const defaultBeatSettings=():BeatSettings=>({enabled:false,bpm:120,offset_ms:0,tap_latency_ms:0,subdivisions:12})
export function upgradeBeatSettings(value?:Partial<BeatSettings>){
 const timing={...defaultBeatSettings(),...value,subdivisions:12}
 return {timing,needsConfirmation:!!value?.enabled&&value.subdivisions!==12}
}
export function estimateTaps(raw:number[],latencyMs:number){
 if(raw.length<4||!Number.isFinite(latencyMs)||Math.abs(latencyMs)>1000)throw new Error('Need four valid taps')
 const times=raw.map(t=>t-latencyMs/1000)
 if(times.some(t=>!Number.isFinite(t))||times.some((t,i)=>i>0&&t<=times[i-1]!))throw new Error('Taps must increase')
 const gaps=times.slice(1).map((t,i)=>t-times[i]!).sort((a,b)=>a-b)
 const period=gaps[Math.floor(gaps.length/2)]!
 const bpm=Math.round(60/period*1000)/1000
 if(bpm<30||bpm>400)throw new Error('Tap BPM out of range')
 const origins=times.map((t,i)=>t-i*period).sort((a,b)=>a-b)
 const offset=((origins[Math.floor(origins.length/2)]!%period)+period)%period
 return {bpm,offset_ms:Math.round(offset*1000),spread_ms:Math.round((gaps.at(-1)!-gaps[0]!)*1000)}
}
