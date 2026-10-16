import test from 'node:test'
import assert from 'node:assert/strict'
import { Transport } from '../src/features/audio/Transport.ts'

function setup() {
  const callbacks = new Map()
  let next = 0
  const media = new EventTarget()
  Object.assign(media, { currentTime: 0, duration: 10, paused: true, playbackRate: 1,
    async play() { this.paused = false; this.dispatchEvent(new Event('play')) },
    pause() { this.paused = true; this.dispatchEvent(new Event('pause')) } })
  const scheduler = { request(fn) { callbacks.set(++next, fn); return next }, cancel(id) { callbacks.delete(id) } }
  return { media, transport: new Transport(media, scheduler), callbacks,
    tick() { for (const [id, fn] of [...callbacks]) { callbacks.delete(id); fn(0) } } }
}

test('all views read the media clock, including background jumps', async () => {
  const { media, transport, tick, callbacks } = setup()
  const one = [], two = []
  transport.subscribe((value) => one.push(value.position))
  transport.subscribe((value) => two.push(value.position))
  await transport.play()
  assert.equal(callbacks.size, 1)
  media.currentTime = 7.25
  tick()
  assert.equal(one.at(-1), 7.25)
  assert.deepEqual(one, two)
  transport.pause()
  assert.equal(callbacks.size, 0)
  transport.seek(2)
  assert.equal(two.at(-1), 2)
  transport.destroy()
})

test('loop uses overshoot and rate does not accumulate another clock', async () => {
  const { media, transport, tick } = setup()
  transport.setLoop([2, 4])
  transport.setRate(2)
  await transport.play()
  media.currentTime = 4.3
  tick()
  assert.ok(Math.abs(media.currentTime - 2.3) < .00001)
  assert.equal(transport.snapshot().rate, 2)
  transport.destroy()
})

test('invalid coordinates and destroy', () => {
  const { transport, callbacks } = setup()
  for (const position of [-1, 11, NaN]) assert.throws(() => transport.seek(position))
  for (const loop of [[3, 2], [-1, 2], [0, 11]]) assert.throws(() => transport.setLoop(loop))
  assert.throws(() => transport.setRate(Infinity))
  transport.destroy()
  transport.destroy()
  assert.equal(callbacks.size, 0)
  assert.throws(() => transport.seek(1))
})

test('play rejection is visible and does not synthesize progress', async () => {
  const { media, transport } = setup()
  media.play = async () => { throw new Error('denied') }
  await assert.rejects(transport.play())
  assert.ok(transport.snapshot().error)
  assert.equal(transport.snapshot().position, 0)
  transport.destroy()
})

test('late play promise cannot resume after pause', async () => {
  const { media, transport } = setup()
  let release
  media.play = () => new Promise((resolve) => { release = () => { media.paused = false; resolve() } })
  const pending = transport.play()
  transport.pause()
  release()
  await pending
  assert.equal(media.paused, true)
  transport.destroy()
})
