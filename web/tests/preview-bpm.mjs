// Isolated manual regression page: actual components/CSS, synthetic audio only.
// node tests/preview-bpm.mjs; open http://127.0.0.1:4319/__bpm_review
import {createServer} from 'vite'
import {fileURLToPath} from 'node:url'

const root=fileURLToPath(new URL('../',import.meta.url))
const rate=22050,seconds=30,pcm=Buffer.alloc(rate*seconds*2)
for(let i=0;i<rate*seconds;i++)pcm.writeInt16LE(Math.round(1200*Math.sin(2*Math.PI*220*i/rate)),i*2)
const header=Buffer.alloc(44)
header.write('RIFF');header.writeUInt32LE(36+pcm.length,4);header.write('WAVEfmt ',8)
header.writeUInt32LE(16,16);header.writeUInt16LE(1,20);header.writeUInt16LE(1,22)
header.writeUInt32LE(rate,24);header.writeUInt32LE(rate*2,28);header.writeUInt16LE(2,32)
header.writeUInt16LE(16,34);header.write('data',36);header.writeUInt32LE(pcm.length,40)
const wav=Buffer.concat([header,pcm])
const html=`<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BPM 控件回归检查</title></head><body><div id="app"></div><script type="module">
import {createApp,ref} from 'vue/dist/vue.esm-bundler.js'
import StudioBpm from '/src/components/StudioBpm.vue'
import StudioTrim from '/src/components/StudioTrim.vue'
import '/src/studio.css'
createApp({
 components:{StudioBpm,StudioTrim},
 setup(){return {timing:ref({enabled:false,bpm:127.823,offset_ms:194,tap_latency_ms:0,subdivisions:12}),trim:ref({start_ms:2000,end_ms:28000}),bpm:ref(),clip:ref()}},
 template:\`<main class="editor-surface" style="margin:20px auto;padding:24px;max-width:1240px"><h2>BPM 与节拍对齐 · 控件检查</h2><StudioBpm ref="bpm" v-model="timing" :trim="trim" :duration="30" audio-url="/fixture/audio" :active="true" :blocked="false" status="" :analysis="null" @playing="clip?.pause()"/><details><summary>裁剪播放器（检查两处播放互斥）</summary><StudioTrim ref="clip" v-model="trim" :duration="30" audio-url="/fixture/audio" :active="true" :blocked="false" @playing="bpm?.pause()"/></details></main>\`
}).mount('#app')
</script></body></html>`
const server=await createServer({root,appType:'custom',server:{host:'127.0.0.1',port:4319,strictPort:true}})
server.middlewares.use(async(req,res,next)=>{
 const path=req.url?.split('?')[0]
 if(path==='/__bpm_review'){res.setHeader('Content-Type','text/html');res.end(await server.transformIndexHtml(path,html))}
 else if(path==='/fixture/waveform'){res.setHeader('Content-Type','application/json');res.end(JSON.stringify({peaks:Array.from({length:600},(_,i)=>.25+.65*Math.abs(Math.sin(i*.13)))}))}
 else if(path==='/fixture/audio'){
  const match=/bytes=(\d+)-(\d*)/.exec(req.headers.range||'')
  const start=match?Number(match[1]):0,end=match&&match[2]?Math.min(Number(match[2]),wav.length-1):wav.length-1
  res.setHeader('Content-Type','audio/wav');res.setHeader('Accept-Ranges','bytes')
  if(match){res.statusCode=206;res.setHeader('Content-Range',`bytes ${start}-${end}/${wav.length}`)}
  res.setHeader('Content-Length',end-start+1);res.end(wav.subarray(start,end+1))
 }else next()
})
await server.listen()
console.log('Synthetic-audio regression page: http://127.0.0.1:4319/__bpm_review')
