import test from 'node:test'
import assert from 'node:assert/strict'
import {SimilaritySession} from '../src/features/similarity/SimilaritySession.ts'
const timing={id:'t',duration:12,anchors:[{q:0,t:0},{q:8,t:4},{q:16,t:12}],downbeats:[],meters:[{start:0,numerator:4,denominator:4}],review:'accepted',evidence:[]}
const proposal={id:'p',fingerprint:'p-hash',timingFingerprint:'t-hash',source:{start:0,end:4},target:{start:8,end:12},scores:{mix:1},warnings:['baseline']}
const session=()=>new SimilaritySession(timing,'t-hash',[proposal])
test('A/B loops and second editing use local tempo',()=>{
 const s=session();assert.deepEqual(s.listen('primary'),[0,2]);assert.deepEqual(s.listen('p'),[4,8]);assert.deepEqual(s.fromSeconds(5,7),{start:9,end:11})
 assert.throws(()=>s.fromSeconds(-1,2));assert.throws(()=>s.edit('primary',{start:4,end:1}))
})
test('each enabled target needs explicit relation and confirmation; edits revoke it',()=>{
 const s=session();assert.throws(()=>s.confirmation('u','reason'));s.configure('p',true,'variation',['arrangement']);s.confirm('p')
 assert.equal(s.confirmation('u','listened').targets[0].confirmed,true)
 s.edit('primary',{start:1,end:4});assert.throws(()=>s.confirmation('u','reason'))
 assert.throws(()=>s.configure('p',true,'variation',['full_pattern']))
 s.configure('p',true,'exact_repeat',['full_pattern']);s.confirm('p');assert.throws(()=>s.confirmation('u',' '))
 s.invalidate('new');assert.throws(()=>s.confirmation('u','reason'));assert.throws(()=>s.listen('p'))
})
test('overlapping target or primary cannot be confirmed and emitted data is detached',()=>{
 const s=session();s.configure('p',true,'variation',[]);s.edit('p',{start:2,end:5});assert.throws(()=>s.confirm('p'))
 s.edit('p',{start:8,end:12});s.confirm('p');const value=s.confirmation('u','reason');value.targets[0].span.start=0;assert.equal(s.targets[0].span.start,8)
 assert.throws(()=>new SimilaritySession(timing,'new',[proposal]))
})
