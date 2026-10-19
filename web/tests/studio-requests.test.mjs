import test from 'node:test'
import assert from 'node:assert/strict'
import {acceptsLLMResult,mergeSuggestedFields} from '../src/studioRequests.ts'

test('late LLM responses cannot overwrite another project, newer revision or reopened view',()=>{
 const started={project_id:'a',version:3}
 assert.equal(acceptsLLMResult(started,started,1,1,false),true)
 for(const [current,view,disposed] of [[null,1,false],[{project_id:'b',version:3},1,false],[{project_id:'a',version:4},1,false],[started,2,false],[started,1,true]])
  assert.equal(acceptsLLMResult(current,started,view,1,disposed),false)
})
test('LLM suggestions fill untouched fields and preserve manual edits',()=>{
 const before={title:'old',author:'',scene:'scene_01',creator:'mikee',model:'v3'}
 const edited={...before,title:'manual',creator:'new creator'}
 const result=mergeSuggestedFields(edited,before,{title:'suggested',author:'artist',scene:'scene_12'})
 assert.deepEqual(result,{...edited,author:'artist',scene:'scene_12'})
 assert.equal(edited.author,'')
})
