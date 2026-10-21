import test from 'node:test'
import assert from 'node:assert/strict'
import {readFile} from 'node:fs/promises'
import {parse,compileScript} from '@vue/compiler-sfc'
import ts from 'typescript'
import {createRenderer,nextTick} from 'vue'

globalThis.document={documentElement:{lang:''}}
globalThis.requestAnimationFrame=()=>1
globalThis.cancelAnimationFrame=()=>{}
const url=new URL('../src/components/StudioTrim.vue',import.meta.url)
let script=compileScript(parse(await readFile(url,'utf8')).descriptor,{id:'trim-regression'}).content
script=script.replace(/import StudioTimeline from '[^']+'/,"const StudioTimeline={}")
script=script.replace(/from '(\.\.\/[^']+)'/g,(_,name)=>`from '${new URL(name+'.ts',url).href}'`)
script=script.replace(/from ["']vue["']/g,`from '${import.meta.resolve('vue')}'`)
const code=ts.transpileModule(script,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText
const Component=(await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'))).default
Component.render=()=>null
const renderer=createRenderer({
 createComment:()=>({}),createElement:()=>({}),createText:()=>({}),
 insert(){},remove(){},setText(){},setElementText(){},patchProp(){},
 parentNode:()=>null,nextSibling:()=>null,
})
function mount(){
 const app=renderer.createApp(Component,{modelValue:{start_ms:2000,end_ms:9000},duration:12,audioUrl:'/audio',active:true,blocked:false})
 app.mount({})
 const state=app._instance.setupState
 const media={paused:true,currentTime:0,
  play(){this.paused=false;state.play();return this.pending??Promise.resolve()},
  pause(){this.paused=true;state.isPlaying=false;state.stopFrame()}}
 state.audio=media
 return {app,state,media}
}

test('trim waveform player respects in/out, seeks and replays from in-point',async()=>{
 const {app,state,media}=mount()
 try{
  await state.start();assert.equal(media.paused,false);assert.equal(media.currentTime,2)
  state.seek(6);assert.equal(media.currentTime,6)
  media.currentTime=9;state.tick();assert.equal(media.paused,true)
  state.toggle();await nextTick();assert.equal(media.currentTime,2);assert.equal(media.paused,false)
  await state.tail();assert.equal(media.currentTime,4)
  state.toggle();assert.equal(media.paused,true)
 }finally{app.unmount()}
})

test('trim playback failure is visible and pending play is cancelled on page leave',async()=>{
 const {app,state,media}=mount()
 try{
  media.pending=Promise.reject(new Error('audio unavailable'))
  await state.start();assert.equal(state.playbackError,true);assert.equal(state.pendingPlay,false)
  let resolve;media.pending=new Promise(r=>{resolve=r})
  const pending=state.start()
  app._instance.props.active=false;await nextTick()
  media.paused=false;resolve();await pending
  assert.equal(media.paused,true)
 }finally{app.unmount()}
})
