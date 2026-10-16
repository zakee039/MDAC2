import test from 'node:test'
import assert from 'node:assert/strict'
import {AnalysisSession,reviewWindows} from '../src/features/timing/AnalysisSession.ts'
const candidate=(base='c')=>({candidate_id:'hash',project_id:'p',base_revision:base,timing:{},fit:{accepted:true,residual_seconds:[.01,.03,-.02]},observation:{payload:{audio:{sample_count:1000,sample_rate:100},beat_times:[.2,5,9.8]}}})
test('old candidates and failed fits cannot be adopted',async()=>{
 const s=new AnalysisSession({json:async()=>{throw new Error('must not request')}},'p','c')
 s.candidate=candidate('old');assert.equal(s.current,false);await assert.rejects(s.preview('keep_audio_time'))
 s.candidate=candidate();s.candidate.fit.accepted=false;assert.equal(s.adoptable,false)
})
test('late analysis responses are discarded after disposal',async()=>{
 let resolve;const api={json:()=>new Promise(r=>resolve=r)},s=new AnalysisSession(api,'p','c')
 const pending=s.start({model_id:'final0'});s.dispose();resolve({job_id:'late'});await pending;assert.equal(s.job,null)
})
test('adoption binds reviewed policy, candidate and preview and needs a reason',async()=>{
 const calls=[],api={json:async(path,body)=>{calls.push({path,body});return {fingerprint:'preview',conflicts:[],diffs:[]}}}
 const s=new AnalysisSession(api,'p','c');s.candidate=candidate()
 await assert.rejects(s.adopt('reason'));await s.preview('keep_audio_time');await assert.rejects(s.adopt(' '));await s.adopt('listened')
 assert.equal(calls[1].body.preview_fingerprint,'preview');assert.equal(calls[1].body.position_policy,'keep_audio_time');assert.match(calls[1].path,/hash\/adopt$/)
 s.migration.conflicts=[{}];await assert.rejects(s.adopt('reason'))
})
test('review loops bound edges and include the largest residual region',()=>{
 assert.deepEqual(reviewWindows(candidate()),[{label:'开头与弱起',start:0,end:3.2},{label:'最大拟合偏差',start:3,end:8},{label:'尾部与静音',start:7.800000000000001,end:10}])
})
