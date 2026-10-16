import test from 'node:test'
import assert from 'node:assert/strict'
import {fraction,viewTiming,storedTiming} from '../src/services/manual.ts'
test('decimal/scientific inputs serialize canonical rationals without floating denominator guesses',()=>{
 assert.equal(fraction(0),'0');assert.equal(fraction(.125),'1/8');assert.equal(fraction(-.25),'-1/4');assert.equal(fraction(1e-7),'1/10000000');assert.throws(()=>fraction(NaN))
})
test('unchanged rational timing anchors retain exact stored strings',()=>{
 const source={revision_id:'t',review:'pending',parent_fingerprint:null,downbeat_q:['0'],mapping:{anchors:[{q:'0',t:'0'},{q:'1/3',t:'1/7'}],audio_duration:'1/7',meter_segments:[{start_q:'0',numerator:4,denominator:4}]}}
 const view=viewTiming(source),stored=storedTiming(view,source)
 assert.deepEqual(stored.mapping,source.mapping);view.anchors[1].t=.125;assert.equal(storedTiming(view,source).mapping.anchors[1].t,'1/8')
})
