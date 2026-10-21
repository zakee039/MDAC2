import test from 'node:test'
import assert from 'node:assert/strict'
import {rebaseTiming,visibleSegments,tempoAt,fitSegmentBeats,markerPosition} from '../src/studioTimeline.ts'

test('out-point is visible on initial load and restore despite sub-ms duration rounding',()=>{
 for(const duration of [180.4796,180.4804,180.4805,123.456789]){
  const stored=Math.round(duration*1000)/1000
  const position=markerPosition(stored,0,duration)
  assert.notEqual(position,null)
  assert.ok(position>99.99&&position<=100)
 }
 assert.equal(markerPosition(0,0,180.4796),0)
 assert.equal(markerPosition(180.480,0,180.4796),100)
 assert.equal(markerPosition(180.480,100,120),null) // genuinely outside zoom
 assert.equal(markerPosition(180.482,0,180.4796),null)
 assert.equal(markerPosition(99.9996,100,120),0)
})

test('in-point changes preserve original beat phase, including hidden tempo changes and restore',()=>{
 const original={enabled:true,bpm:120,offset_ms:120,tap_latency_ms:80,subdivisions:12,segments:[{start_ms:5000,bpm:90,offset_ms:5100,tap_latency_ms:0},{start_ms:20000,bpm:180,offset_ms:20010,tap_latency_ms:0}]}
 const shifted=rebaseTiming(original,0,10000)
 assert.equal(shifted.offset_ms,-9880)
 assert.equal(shifted.segments[0].start_ms,-5000)
 assert.equal(shifted.segments[0].offset_ms,-4900)
 assert.equal(tempoAt(shifted,0).bpm,90)
 assert.deepEqual(visibleSegments(shifted,20000).map(s=>[s.start_ms,s.end_ms,s.value.bpm]),[[0,10000,90],[10000,20000,180]])
 assert.deepEqual(rebaseTiming(shifted,10000,0),original)
})

test('per-segment detection fits only selected source interval, not whole-song average',()=>{
 const times=[...Array.from({length:20},(_,i)=>i*.5),...Array.from({length:10},(_,i)=>10+i)]
 assert.equal(fitSegmentBeats(times,0,10000).bpm,120)
 assert.deepEqual(fitSegmentBeats(times,10000,20000),{bpm:60,offset_ms:10000})
 assert.throws(()=>fitSegmentBeats(times,19000,20000))
})
