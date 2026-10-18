import test from 'node:test'
import assert from 'node:assert/strict'
import {beatWindow,DrumClock,thickBeat} from '../src/studioMetronome.ts'
import {projectStage,exportAction} from '../src/studioFlow.ts'
test('thick beat has sustained body, peak headroom and fits fastest supported tempo',()=>{
 for(const rate of [44100,48000]){
  const data=thickBeat(rate)
  assert.ok(data.length/rate<60/400)
  assert.ok(data.every(Number.isFinite))
  assert.ok(Math.abs(Math.max(...data.map(Math.abs))-.97)<1e-6)
  const body=data.slice(Math.round(.02*rate),Math.round(.09*rate))
  const rms=Math.sqrt(body.reduce((sum,x)=>sum+x*x,0)/body.length)
  assert.ok(rms>.35)
  assert.ok(data[0]===0);assert.ok(Math.abs(data.at(-1))<.001)
 }
})
test('next builds directly, resumes package progress, and blocks stale charts',()=>{
 const generated={confirmed:true,generation:'chart'}
 assert.equal(projectStage(generated),3)
 assert.equal(exportAction(generated,false),'build')
 assert.equal(exportAction(generated,true),'blocked')
 assert.equal(exportAction({confirmed:true},false),'blocked')
 const running={...generated,active_attempt:'a',attempts:[{attempt_id:'a',operation:'export',state:'running'}]}
 assert.equal(projectStage(running),4);assert.equal(exportAction(running,false),'show')
 const failed={...running,active_attempt:undefined,attempts:[{attempt_id:'a',operation:'export',state:'failed'}]}
 assert.equal(projectStage(failed),4);assert.equal(exportAction(failed,false),'build')
 assert.equal(exportAction({...generated,export:'package'},false),'show')
 assert.equal(projectStage({...generated,draft:{}}),2)
 assert.equal(exportAction({...running,attempts:[{attempt_id:'a',operation:'generate',state:'running'}]},false),'blocked')
})
test('beat clock respects tempo, positive/negative offsets, seeks and no catch-up',()=>{
 assert.deepEqual(beatWindow(1,120,100,-Infinity),[{index:2,time:1.1}])
 assert.deepEqual(beatWindow(1.05,120,100,2),[])
 assert.deepEqual(beatWindow(0,120,-100,-Infinity),[])
 assert.deepEqual(beatWindow(.35,120,-100,-Infinity),[{index:1,time:.4}])
 const seek=beatWindow(120,120,0,-Infinity)
 assert.equal(seek.length,1);assert.equal(seek[0].time,120)
 assert.equal(beatWindow(120,NaN,0,-Infinity).length,0)
 assert.equal(beatWindow(2,60,0,-Infinity)[0].time,2)
})
test('scheduled drum nodes stop on pause/dispose and restart without duplicate beats',async()=>{
 const nodes=[]
 const ctx={sampleRate:44100,currentTime:0,state:'running',baseLatency:0,
  createBuffer:(_c,n)=>({getChannelData:()=>new Float32Array(n)}),
  createBufferSource:()=>{const node={connect:()=>({connect(){}}),disconnect(){},start(t){this.time=t},stop(){this.stopped=true;this.onended?.()}};nodes.push(node);return node},
  createGain:()=>({gain:{value:0},disconnect(){}}),resume:async()=>{},close:async()=>{ctx.closed=true}}
 const audio={paused:false,ended:false,seeking:false,currentTime:1,playbackRate:1}
 const clock=new DrumClock(ctx,audio,()=>({bpm:120,offset_ms:100,volume:.9}))
 try{await clock.start();assert.equal(nodes.length,1);assert.ok(Math.abs(nodes[0].time-.1)<1e-8)
  clock.stop();assert.ok(nodes[0].stopped)
  audio.currentTime=12;await clock.start();assert.equal(nodes.length,2)
  audio.paused=true;clock.stop();await clock.start();assert.equal(nodes.length,2)
 }finally{clock.dispose()}
 assert.ok(ctx.closed);assert.ok(nodes.every(n=>n.stopped))
})
