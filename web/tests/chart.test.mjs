import test from 'node:test'
import assert from 'node:assert/strict'
import {ChartEditor,numeric,endPosition,subscribeOverlay} from '../src/features/chart/ChartEditor.ts'
const base=()=>({revision_id:'chart',timing_revision:'timing',events:[],onset_revision:null,arrangement_policy_hash:null,semantics_policy_hash:null,user_patch_ids:[],schema_version:1})
const note=()=>({q:'1/3',duration_q:'0',lane_mask:['ground'],semantic:'tap',accent:false,locked:false,source_kind:'manual',source_id:null})
const time=q=>{const n=numeric(q);if(n<0||n>16)throw new Error('bounds');return n/2}
test('simultaneous events stay distinct; undo/redo preserves IDs and immutable source',()=>{
 const original=base(),e=new ChartEditor(original,'timing',time);e.add(note());e.add({...note(),lane_mask:['air']})
 const both=e.snapshot();assert.equal(both.events.length,2);assert.notEqual(both.events[0].event_id,both.events[1].event_id)
 e.undo();assert.equal(e.snapshot().events.length,1);e.redo();assert.deepEqual(e.snapshot().events,both.events);assert.equal(original.events.length,0)
})
test('lock blocks every edit/delete until explicit unlock and undo restores locks',()=>{
 const e=new ChartEditor(base(),'timing',time);e.add(note());const id=e.snapshot().events[0].event_id
 e.lock(id,true);assert.throws(()=>e.update(id,{q:'2'}));assert.throws(()=>e.remove(id))
 e.lock(id,false);e.update(id,{q:'2',lane_mask:['air']});assert.equal(e.snapshot().events[0].event_id,id)
 e.undo();e.undo();assert.equal(e.snapshot().events[0].locked,true)
})
test('Hold tails exact rational addition; bad duration, lanes and timing rejected atomically',()=>{
 const e=new ChartEditor(base(),'timing',time);const hold={...note(),semantic:'hold',duration_q:'2/3'}
 assert.equal(endPosition(hold),'1');e.add(hold)
 for(const n of [{...hold,q:'16'},{...hold,duration_q:'0'},{...note(),duration_q:'1'},{...note(),semantic:'gemini'},{...note(),lane_mask:['air','air']},{...note(),q:'2/6'}])assert.throws(()=>e.add(n))
 assert.equal(e.snapshot().events.length,1);assert.throws(()=>new ChartEditor(base(),'other',time))
})
test('overlay follows shared transport seek/pause snapshots and cleans subscription',()=>{
 const e=new ChartEditor(base(),'timing',time);e.add({...note(),q:'8',semantic:'hold',duration_q:'4'})
 let receive,removed=false,last
 const stop=subscribeOverlay(e,{subscribe(fn){receive=fn;fn({position:0});return()=>{removed=true}}},(items,seconds)=>{last={items,seconds}})
 assert.equal(last.items.length,0);receive({position:5});assert.equal(last.items[0].end,6);assert.equal(last.seconds,5)
 receive({position:0});assert.equal(last.items.length,0);stop();assert.equal(removed,true)
})
