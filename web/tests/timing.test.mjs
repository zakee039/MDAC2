import test from 'node:test'
import assert from 'node:assert/strict'
import {TimingEditor,secondsAt,validateTiming} from '../src/features/timing/TimingEditor.ts'
const fixture=()=>({id:'original',duration:8,anchors:[{q:0,t:0},{q:4,t:2},{q:8,t:4},{q:16,t:8}],downbeats:[0],meters:[{start:0,numerator:4,denominator:4}],review:'accepted',evidence:[{start:2,end:4,score:.3,reason:'漂移'}]})
test('preview does not mutate source; commit invalidates review and preserves evidence',()=>{
 const base=fixture(),editor=new TimingEditor(base)
 editor.propose({type:'drag',q:4,t:2.2});assert.equal(editor.snapshot().anchors[1].t,2)
 const copy=editor.preview();copy.anchors[1].t=7;assert.equal(editor.preview().anchors[1].t,2.2)
 assert.throws(()=>editor.commit('obsolete'));editor.commit('original')
 assert.equal(base.review,'accepted');assert.equal(editor.snapshot().review,'pending');assert.deepEqual(editor.snapshot().evidence,base.evidence)
 assert.equal(secondsAt(editor.snapshot(),4),2.2)
})
test('multiple undo/redo creates revisions without resurrecting approval',()=>{
 const editor=new TimingEditor(fixture())
 for(const t of [2.1,2.3]){editor.propose({type:'drag',q:4,t});editor.commit(editor.snapshot().id)}
 editor.undo();assert.equal(editor.snapshot().anchors[1].t,2.1)
 editor.undo();assert.equal(editor.snapshot().anchors[1].t,2);assert.equal(editor.snapshot().review,'pending')
 editor.redo();assert.equal(editor.snapshot().anchors[1].t,2.1)
 editor.propose({type:'insert',q:2,t:1});editor.commit(editor.snapshot().id);editor.redo();assert.equal(editor.snapshot().anchors.length,5)
})
test('invalid edits clear preview, monotonic/downbeat/audio constraints',()=>{
 const editor=new TimingEditor(fixture())
 for(const edit of [{type:'drag',q:4,t:4},{type:'insert',q:2,t:NaN},{type:'delete',q:0},{type:'downbeat',q:2,enabled:true},{type:'insert',q:18,t:9}])assert.throws(()=>editor.propose(edit))
 assert.equal(editor.preview(),null)
 editor.propose({type:'downbeat',q:0,enabled:false});editor.commit(editor.snapshot().id)
 editor.propose({type:'delete',q:0});editor.commit(editor.snapshot().id);assert.equal(editor.snapshot().anchors.length,3)
 assert.throws(()=>secondsAt(editor.snapshot(),0))
})
test('6/8 and pickup retain quarter units; tap preserves raw time and latency',()=>{
 const base=fixture();base.anchors=[{q:-1,t:0},{q:0,t:1},{q:3,t:4},{q:6,t:7}];base.meters=[{start:0,numerator:6,denominator:8}]
 const editor=new TimingEditor(base);editor.propose({type:'downbeat',q:3,enabled:true});editor.commit(base.id)
 assert.equal(secondsAt(editor.snapshot(),-0.5),.5)
 editor.tap(2,.1);assert.deepEqual(editor.taps,[{seconds:2,latency:.1}]);assert.throws(()=>editor.tap(9,0))
 assert.throws(()=>validateTiming({...base,meters:[...base.meters,{start:4,numerator:4,denominator:4}]}))
})
