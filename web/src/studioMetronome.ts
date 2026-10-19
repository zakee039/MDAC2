type ClockSettings={bpm:number;offset_ms:number;volume:number;segments?:{start_ms:number;bpm:number;offset_ms:number}[];end_time?:number}
/** Beat positions use media time, so seeking never produces a burst of missed beats. */
export function beatWindow(time:number,bpm:number,offsetMs:number,last:number,horizon=.12){
 if(![time,bpm,offsetMs].every(Number.isFinite)||bpm<30||bpm>400)return []
 const period=60/bpm,origin=offsetMs/1000
 const first=Math.max(Math.ceil((Math.max(0,time)-origin)/period-1e-8),last+1)
 const end=Math.floor((time+horizon-origin)/period+1e-8)
 return Array.from({length:Math.max(0,end-first+1)},(_,i)=>({index:first+i,time:origin+(first+i)*period}))
}

/** Original synthesized kick: a low body, audible mid harmonics and a short attack. */
export function thickBeat(sampleRate:number){
 const data=new Float32Array(Math.ceil(sampleRate*.14))
 let phase=0,seed=84931,peak=0,previous=0
 for(let i=0;i<data.length;i++){
  const t=i/sampleRate
  seed=(1664525*seed+1013904223)>>>0
  const noise=seed/2147483648-1,high=(noise-previous)*.5;previous=noise
  phase+=2*Math.PI*(65+105*Math.exp(-t/.014))/sampleRate
  const body=(Math.sin(phase)+.36*Math.sin(phase*2)+.18*Math.sin(phase*3))*Math.exp(-t/.065)
  const attack=(.38*high+.3*Math.sin(2*Math.PI*2400*t))*Math.exp(-t/.008)
  const fade=Math.min(1,t/.001,Math.max(0,(.14-t)/.025))
  data[i]=Math.tanh(2.1*(body+attack))*fade
  peak=Math.max(peak,Math.abs(data[i]))
 }
 // Normalize the actual sample, not just its volume slider. Leave peak headroom.
 for(let i=0;i<data.length;i++)data[i]*=.97/peak
 return data
}

export class DrumClock {
 private nodes=new Set<AudioBufferSourceNode>()
 private timer:ReturnType<typeof setInterval>|undefined
 private last=-Infinity
 private buffer:AudioBuffer
 private context:AudioContext
 private audio:HTMLAudioElement
 private config:()=>ClockSettings
 constructor(context:AudioContext,audio:HTMLAudioElement,config:()=>ClockSettings){
  this.context=context;this.audio=audio;this.config=config
  const rate=context.sampleRate
  const samples=thickBeat(rate)
  this.buffer=context.createBuffer(1,samples.length,rate)
  this.buffer.getChannelData(0).set(samples)
 }
 unlock(){return this.context.resume()}
 async start(){
  this.stop()
  // Install the timer before awaiting resume, so pause can cancel a pending start.
  this.timer=setInterval(()=>this.tick(),25)
  try{await this.context.resume();if(this.timer)this.tick()}catch{this.stop();throw new Error('audio unavailable')}
 }
 stop(){if(this.timer)clearInterval(this.timer);this.timer=undefined;this.last=-Infinity;for(const n of this.nodes){try{n.stop()}catch{}n.disconnect()}this.nodes.clear()}
 private tick(){
  if(this.audio.paused||this.audio.ended||this.audio.seeking||this.context.state!=='running')return
  const c=this.config(),rate=this.audio.playbackRate||1
  const stamp=this.context.getOutputTimestamp?.()
  const lag=stamp?.contextTime?Math.max(0,Math.min(.2,this.context.currentTime-stamp.contextTime)):(this.context.baseLatency||0)
  const sections=[{start_ms:-Infinity,bpm:c.bpm,offset_ms:c.offset_ms},...(c.segments||[])]
  const beats=sections.flatMap((section,i)=>{
   const end=Math.min(c.end_time??Infinity,(sections[i+1]?.start_ms??Infinity)/1000)
   const now=Math.max(this.audio.currentTime,section.start_ms/1000)
   if(now>this.audio.currentTime+(.12+lag)*rate)return []
   return beatWindow(now,section.bpm,section.offset_ms,-Infinity,(.12+lag)*rate).filter(b=>b.time<end&&b.time>this.last+1e-8)
  }).sort((a,b)=>a.time-b.time)
  for(const beat of beats){
   this.last=beat.time
   const delay=(beat.time-this.audio.currentTime)/rate-lag
   if(delay<-.025)continue
   const node=this.context.createBufferSource(),gain=this.context.createGain()
   node.buffer=this.buffer;gain.gain.value=Math.max(0,Math.min(1,c.volume))
   node.connect(gain).connect(this.context.destination);this.nodes.add(node)
   node.onended=()=>{this.nodes.delete(node);node.disconnect();gain.disconnect()}
   node.start(this.context.currentTime+Math.max(0,delay))
  }
 }
 dispose(){this.stop();void this.context.close()}
}
