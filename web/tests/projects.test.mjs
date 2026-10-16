import test from 'node:test'
import assert from 'node:assert/strict'
import { ProjectsApi, ApiError } from '../src/services/projects.ts'
import { ProjectSession } from '../src/pages/projects/ProjectSession.ts'
const chart = {revision_id:'draft', timing_revision:'timing', events:[], schema_version:1, onset_revision:null, arrangement_policy_hash:null, semantics_policy_hash:null, user_patch_ids:[]}
const project = {project_id:'project', title:'Song', head_revision:null, archived:false, chart:null}
test('API uses exact project command contract, auth and same-origin IDs', async () => {
  const requests=[]
  const api = new ProjectsApi(() => 'secret', async (url, init) => { requests.push({url,init}); return Response.json(project) })
  await api.create('Song'); await api.get('project'); await api.save(project,chart)
  const {url,init}=requests.at(-1)
  assert.equal(url,'/api/v1/projects/project/commands')
  assert.equal(init.headers.Authorization,'Bearer secret')
  assert.deepEqual(Object.keys(JSON.parse(init.body)).sort(),['command_id','expected_revision','payload','type'])
  assert.equal(JSON.parse(init.body).expected_revision,null)
  await assert.rejects(api.get('../secrets')); assert.equal(requests.length,3)
})
test('conflict retains draft, explicit merge rebases once and still uses CAS',async () => {
  let fail=true, saved
  const remote={...project,head_revision:'other',chart:{...chart,revision_id:'other'}}
  const session=new ProjectSession({create:async()=>project,get:async()=>remote, save:async(p,c)=>{saved=p; if(fail)throw new ApiError(409); return {revision_id:c.revision_id}}})
  await session.create('Song'); session.edit(chart); await session.save()
  assert.equal(session.conflict,true); assert.deepEqual(session.draft,chart)
  await session.open('another'); assert.equal(session.project.project_id,'project')
  await session.compare(); assert.throws(()=>session.merge(null)); assert.equal(session.conflict,true)
  session.merge(chart); fail=false; await session.save()
  assert.equal(saved.head_revision,'other'); assert.equal(session.dirty,false)
  assert.notEqual(session.project.head_revision,'draft')
})
test('network failure retains dirty data; create/open failures preserve project',async () => {
  const session=new ProjectSession({create:async()=>project,get:async()=>{throw new ApiError(404)},save:async()=>{throw new ApiError(0)}})
  await session.create('Song'); await session.open('missing'); assert.equal(session.project.title,'Song')
  session.edit(chart); await session.save(); assert.equal(session.dirty,true); assert.deepEqual(session.draft,chart)
  session.discard(); assert.equal(session.dirty,false); assert.equal(session.draft,null)
})
test('error bodies never surface server paths; malformed success is visible',async () => {
  for (const status of [401,403,404,409,422,500]) {
    const api=new ProjectsApi(()=>'',async()=>new Response('SECRET PATH',{status}))
    await assert.rejects(api.get('p'),e=>e.status===status&&!e.message.includes('SECRET'))
  }
  await assert.rejects(new ProjectsApi(()=>'',async()=>new Response('bad json')).get('p'),e=>e.status===502)
})

test('native fetch receives the browser global receiver', async () => {
  let receiver
  const api = new ProjectsApi(() => 'token', async function () {
    receiver = this
    return new Response(JSON.stringify({project_id:'p', title:'p', head_revision:null, archived:false, chart:null}))
  })
  await api.get('p')
  assert.equal(receiver, globalThis)
})
