import test from 'node:test'
import assert from 'node:assert/strict'
import {OnsetSession} from '../src/features/onsets/OnsetSession.ts'
const candidate=(base='c')=>({candidate_id:'hash',project_id:'p',base_revision:base,selected_ids:['one'],onsets:{candidates:[{candidate_id:'one',quantized_q:'1'},{candidate_id:'two',quantized_q:'2'},{candidate_id:'raw',quantized_q:null}]}})
function session(api={json:async()=>{throw Error('unexpected')}}){const s=new OnsetSession(api,'p','c');s.candidate=candidate();s.selection=['one'];return s}

test('stale, draft and unquantized candidates cannot be submitted',async()=>{
 const s=session();s.candidate=candidate('old');await assert.rejects(s.preview())
 s.candidate=candidate();s.setBlocked(true);assert.throws(()=>s.select(['two']));await assert.rejects(s.preview())
 s.setBlocked(false);assert.throws(()=>s.select(['raw']));assert.throws(()=>s.select(['one','one']));assert.throws(()=>s.select(['unknown']))
})
test('selection change invalidates pending preview and accepted preview',async()=>{
 let resolve;const s=session({json:()=>new Promise(r=>resolve=r)})
 const pending=s.preview();s.select(['two']);resolve({fingerprint:'late',selected_ids:['one']});await pending
 assert.equal(s.diff,null);s.diff={fingerprint:'old'};s.toggle('two',false);assert.equal(s.diff,null)
})
test('adoption sends exactly reviewed IDs and reason and prevents double submit',async()=>{
 const calls=[];let finish
 const s=session({json:async(path,body)=>{calls.push({path,body});if(path.endsWith('/preview'))return {fingerprint:'diff',selected_ids:['one']};return new Promise(r=>finish=r)}})
 await assert.rejects(s.adopt('reason'));await s.preview();await assert.rejects(s.adopt(' '))
 const pending=s.adopt('checked');await assert.rejects(s.adopt('again'));assert.throws(()=>s.select(['two']))
 assert.deepEqual(calls[1].body.selected_ids,['one']);assert.equal(calls[1].body.preview_fingerprint,'diff')
 finish({chart:{}});assert.deepEqual(await pending,{chart:{}});await assert.rejects(s.adopt('again'))
})
test('disposal and temporary blocking invalidate in-flight preview',async()=>{
 for(const action of [s=>s.dispose(),s=>{s.setBlocked(true);s.setBlocked(false)}]){
  let resolve;const s=session({json:()=>new Promise(r=>resolve=r)});const pending=s.preview();action(s)
  resolve({fingerprint:'late'});await pending;assert.equal(s.diff,null)
 }
})
test('history restores successful candidate while another panel loads and refresh keeps selection',async()=>{
 const api={json:async path=>path==='projects/p/onset-analysis'?{jobs:[{job_id:'j',state:'succeeded'}]}:path==='jobs/j'?{state:'succeeded'}:candidate()}
 const s=new OnsetSession(api,'p','c');const pending=s.history();s.setBlocked(true);await pending
 assert.equal(s.candidate.candidate_id,'hash');s.setBlocked(false);s.select(['two']);await s.refresh();assert.deepEqual(s.selection,['two'])
})
