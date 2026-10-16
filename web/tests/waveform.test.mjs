import test from 'node:test'
import assert from 'node:assert/strict'
import { Transport } from '../src/features/audio/Transport.ts'
import { WaveformController } from '../src/features/audio/Waveform.ts'

function setup() {
  const media = new EventTarget()
  Object.assign(media, { currentTime: 0, duration: 10, paused: true, playbackRate: 1, pause() {}, async play() {} })
  const transport = new Transport(media, { request() { return 1 }, cancel() {} })
  const views = []
  const changes = [], errors = []
  const factory = (options) => {
    const view = { options, cursor(position) { this.position = position }, regions(items) { this.items = items }, destroy() { this.destroyed = true } }
    views.push(view)
    return view
  }
  const create = () => new WaveformController(transport, [[0, .5, -.5]], 10, factory, (r) => changes.push(r), (e) => errors.push(e))
  return { media, transport, create, views, changes, errors }
}

test('two waveforms use exactly the shared media, seek via transport', () => {
  const { media, transport, create, views } = setup()
  const a = create(), b = create()
  assert.equal(views[0].options.media, media)
  assert.equal(views[1].options.media, media)
  views[0].options.onSeek(3)
  assert.equal(views[0].position, 3)
  assert.equal(views[1].position, 3)
  a.destroy()
  transport.seek(5)
  assert.equal(views[0].position, 3)
  assert.equal(views[1].position, 5)
  b.destroy(); transport.destroy()
})

test('regions emit seconds without silently committing project state', () => {
  const { create, views, changes, errors, transport } = setup()
  const controller = create()
  const regions = [{ id: 'a', start: 1, end: 2 }]
  controller.setRegions(regions)
  views[0].options.onRegion({ id: 'a', start: 2, end: 4 })
  assert.deepEqual(regions, [{ id: 'a', start: 1, end: 2 }])
  assert.equal(changes[0].start, 2)
  views[0].options.onRegion({ id: 'a', start: 3, end: 2 })
  assert.equal(errors.length, 1)
  controller.destroy()
  views[0].options.onRegion({ id: 'a', start: 2, end: 5 })
  assert.equal(changes.length, 1)
  transport.destroy()
})

test('remount unsubscribes and invalid peaks are rejected', () => {
  const { transport, create, views } = setup()
  create().destroy()
  const next = create()
  transport.seek(4)
  assert.equal(views[0].position, 0)
  assert.equal(views[1].position, 4)
  assert.throws(() => new WaveformController(transport, [[NaN]], 10, () => {}, () => {}, () => {}))
  next.destroy(); transport.destroy()
})
