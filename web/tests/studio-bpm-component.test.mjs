import test from 'node:test'
import assert from 'node:assert/strict'
import {readFile} from 'node:fs/promises'
import {parse,compileScript} from '@vue/compiler-sfc'
import ts from 'typescript'
import {createRenderer,nextTick} from 'vue'

// Execute the actual SFC controller with media/Web Audio stand-ins. This
// covers interaction/lifecycle races that the isolated beat-clock tests miss.
globalThis.document={documentElement:{lang:''}}
globalThis.requestAnimationFrame=()=>1
globalThis.cancelAnimationFrame=()=>{}
const source=await readFile(new URL('../src/components/StudioBpm.vue',import.meta.url),'utf8')
let script=compileScript(parse(source).descriptor,{id:'bpm-regression'}).content
script=script.replace(/import StudioTimeline from '[^']+'/,"const StudioTimeline={}")
script=script.replace(/from '(\.\.\/[^']+)'/g,(_,name)=>`from '${new URL(name+'.ts',new URL('../src/components/StudioBpm.vue',import.meta.url)).href}'`)
script=script.replace(/from ["']vue["']/g,`from '${import.meta.resolve('vue')}'`)
const code=ts.transpileModule(script,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText
const Component=(await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'))).default
Component.render=()=>null
const renderer=createRenderer({
 createComment:()=>({}),createElement:()=>({}),createText:()=>({}),
 insert(){},remove(){},setText(){},setElementText(){},patchProp(){},
 parentNode:()=>null,nextSibling:()=>null,
})
function deferred(){let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});return {promise,resolve,reject}}
function mount({resume,enabled=true}={}){
 const contexts=[]
 globalThis.AudioContext=class {
  sampleRate=44100;currentTime=0;state='running';baseLatency=0;nodes=[]
  constructor(){contexts.push(this)}
  createBuffer(_channels,n){return {getChannelData:()=>new Float32Array(n)}}
  createBufferSource(){const n={connect:()=>({connect(){}}),disconnect(){},start(){n.started=true},stop(){n.stopped=true;n.onended?.()}};this.nodes.push(n);return n}
  createGain(){return {gain:{value:0},disconnect(){}}}
  resume(){return resume?.promise??this.pending?.promise??Promise.resolve()}
  close(){this.closed=true;return Promise.resolve()}
 }
 const app=renderer.createApp(Component,{modelValue:{enabled,bpm:120,offset_ms:0,subdivisions:12,tap_latency_ms:0},
  trim:{start_ms:0,end_ms:5000},duration:5,audioUrl:'/audio',active:true,blocked:false,status:'',analysis:null})
 app.mount({})
 const state=app._instance.setupState
 const media={paused:true,ended:false,seeking:false,readyState:4,currentTime:1,playbackRate:1,volume:1,
  play(){this.paused=false;state.isPlaying=true;void state.playDrum();return this.pending?.promise??Promise.resolve()},
  pause(){this.paused=true;state.isPlaying=false;state.stopDrum()}}
 state.audio=media
 return {app,state,media,contexts}
}

test('BPM player starts drums and restores them after tapping disables drums',async()=>{
 const {app,state,media,contexts}=mount()
 try{
  await state.toggle();await nextTick()
  assert.equal(media.paused,false);assert.ok(contexts[0].nodes.some(n=>n.started))
  state.tap(-1);await nextTick()
  assert.equal(state.drumEnabled,false);assert.equal(media.paused,false)
  const count=contexts[0].nodes.length
  state.drumEnabled=true;await nextTick();await Promise.resolve()
  assert.ok(contexts[0].nodes.length>count)
  state.pause();assert.equal(media.paused,true)
  assert.ok(contexts[0].nodes.every(n=>n.stopped))
 }finally{app.unmount()}
})

test('switching away while play is pending cannot restart media',async()=>{
 const {app,state,media}=mount()
 try{
  media.pending=deferred()
  const playing=state.toggle()
  app._instance.props.active=false;await nextTick()
  // A browser can complete play after the component was paused.
  media.paused=false;media.pending.resolve();await playing
  assert.equal(media.paused,true);assert.equal(state.pendingPlay,false)
 }finally{app.unmount()}
})

test('a second click cancels pending play and unmount closes Web Audio',async()=>{
 const {app,state,media,contexts}=mount()
 media.pending=deferred()
 const playing=state.toggle()
 await state.toggle()
 media.paused=false;media.pending.resolve();await playing
 assert.equal(media.paused,true)
 app.unmount()
 assert.ok(contexts[0].closed)
})

test('suspended drum initialization never blocks music with chart alignment disabled',async()=>{
 const resume=deferred()
 const {app,state,media}=mount({resume,enabled:false})
 try{
  await state.toggle()
  assert.equal(media.paused,false);assert.equal(state.pendingPlay,false)
  state.pause();resume.resolve();await nextTick()
  assert.equal(media.paused,true)
 }finally{app.unmount()}
})

test('music errors are reported separately and the next click can retry',async()=>{
 const {app,state,media}=mount()
 try{
  media.pending=deferred()
  const playing=state.toggle();media.pending.reject(new Error('missing audio'));await playing
  assert.equal(state.playbackError,true);assert.equal(state.pendingPlay,false)
  assert.equal(media.paused,true);assert.equal(state.drumError,false)
  delete media.pending;await state.toggle()
  assert.equal(state.playbackError,false);assert.equal(media.paused,false)
 }finally{app.unmount()}
})

test('volume endpoints and replay after clip end reach the actual media',async()=>{
 const {app,state,media}=mount()
 try{
  state.musicVolume=0;await nextTick();assert.equal(media.volume,0)
  state.musicVolume=1;await nextTick();assert.equal(media.volume,1)
  media.currentTime=5;await state.toggle()
  assert.equal(media.currentTime,0);assert.equal(media.paused,false)
 }finally{app.unmount()}
})
