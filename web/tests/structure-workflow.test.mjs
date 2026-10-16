import test from 'node:test'
import assert from 'node:assert/strict'
import {reactive} from 'vue'
import {StructureSession} from '../src/features/similarity/StructureSession.ts'
const candidate={candidate_id:'candidate',project_id:'p',base_revision:'c',proposals:[]}
function session(api){const s=reactive(new StructureSession(api,'p','c'));s.candidate=candidate;s.available=true;return s}
const options={target_ids:['target'],mode:'arrangement',source_hold:'reject',target_hold:'reject',locked:'reject'}
const confirmation={primary:{start:0,end:4},targets:[{id:'target',span:{start:8,end:12},relation:'exact_repeat',modes:['arrangement'],confirmed:true}],actor:'user',reason:'checked'}

test('relation edits and draft changes invalidate delayed review/preview',async()=>{
 let resolve;const s=session({json:()=>new Promise(r=>resolve=r)})
 const pending=s.confirm(confirmation);s.invalidateReview();resolve({review_id:'late'});await pending;assert.equal(s.review,null)
 s.review={review_id:'review'};const p=s.preview(options);s.setBlocked(true);s.setBlocked(false);resolve({fingerprint:'old'});await p;assert.equal(s.diff,null)
})
test('Vue reactive options remain serializable and apply binds the reviewed diff',async()=>{
 const calls=[],s=session({json:async(path,body)=>{calls.push({path,body});return path.endsWith('/preview')?{review_id:'review',fingerprint:'fp',conflicts:[]}:{chart:{revision_id:'new'}}}})
 s.review={review_id:'review'};await s.preview(options);const result=await s.apply()
 assert.equal(result.chart.revision_id,'new');assert.deepEqual(calls[1].body.options,options);assert.equal(calls[1].body.preview_fingerprint,'fp')
 await assert.rejects(s.apply())
})
test('full replacement requires extra explicit confirmation and conflicts block it',async()=>{
 const s=session({json:async()=>({chart:{}})});s.review={review_id:'review'}
 s.diff={review_id:'review',fingerprint:'fp',conflicts:[],options:{...options,mode:'full_pattern'}}
 await assert.rejects(s.apply());s.diff.conflicts=[['LOCKED_TARGET_EVENT','id']];await assert.rejects(s.apply(true))
 s.diff.conflicts=[];await s.apply(true)
})
test('old candidate and post-disposal response cannot overwrite state',async()=>{
 let resolve;const s=session({json:()=>new Promise(r=>resolve=r)})
 s.candidate={...candidate,base_revision:'old'};await assert.rejects(s.confirm(confirmation))
 s.candidate=candidate;const p=s.confirm(confirmation);s.dispose();resolve({review_id:'late'});await p;assert.equal(s.review,null)
})
test('undo is restored only for exact current applied revision and is fenced while submitting',async()=>{
 const calls=[];let finish
 const s=session({json:async(path,body)=>{calls.push({path,body});return new Promise(r=>finish=r)}})
 s.operation={operation_id:'op',revision:'other'};await assert.rejects(s.undo())
 s.operation={operation_id:'op',revision:'c'};const pending=s.undo();await assert.rejects(s.undo())
 assert.deepEqual(calls[0].body.expected_revision,'c');finish({chart:{revision_id:'undone'}});await pending;assert.equal(s.operation,null)
})
