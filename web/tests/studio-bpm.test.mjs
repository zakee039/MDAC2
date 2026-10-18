import test from 'node:test'
import assert from 'node:assert/strict'
import {estimateTaps,defaultBeatSettings,upgradeBeatSettings} from '../src/studioBpm.ts'
test('new timing uses twelve subdivisions and legacy enabled charts require confirmation',()=>{
 assert.equal(defaultBeatSettings().subdivisions,12)
 const old={enabled:true,bpm:137,offset_ms:12,tap_latency_ms:80}
 const updated=upgradeBeatSettings(old)
 assert.equal(updated.needsConfirmation,true)
 assert.deepEqual(updated.timing,{...old,subdivisions:12})
 assert.equal(old.subdivisions,undefined)
 assert.equal(upgradeBeatSettings({...old,subdivisions:12}).needsConfirmation,false)
 assert.equal(upgradeBeatSettings().needsConfirmation,false)
})
test('tap BPM uses media times and compensates input latency only once',()=>{
 const taps=[1.13,1.63,2.13,2.63,3.13,3.63]
 const fit=estimateTaps(taps,100)
 assert.equal(fit.bpm,120);assert.equal(fit.offset_ms,30)
 assert.equal(estimateTaps(taps,0).offset_ms,130)
 assert.equal(defaultBeatSettings().enabled,false)
})
test('insufficient, paused or nonmonotonic taps are rejected',()=>{
 assert.throws(()=>estimateTaps([1,2,3],0))
 assert.throws(()=>estimateTaps([1,2,2,3],0))
 assert.throws(()=>estimateTaps([1,2,3,NaN],0))
 assert.throws(()=>estimateTaps([1,2,3,4],2000))
})
