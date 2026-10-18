import test from 'node:test'
import assert from 'node:assert/strict'
import {applyPreset,presets,sceneNames} from '../src/studioOptions.ts'

test('vendor switch clears previous credentials and model without changing saved name',()=>{
 const previous={name:'My config',id:'saved',base_url:'https://private.example/v1',protocol:'openai_compatible',model:'private-model',api_key:'secret',has_key:true,clear_key:false,preset:'custom'}
 const next=applyPreset(previous,'deepseek')
 assert.equal(next.base_url,'https://api.deepseek.com')
 assert.equal(next.api_key,'');assert.equal(next.has_key,false);assert.equal(next.clear_key,true)
 assert.equal(next.model,'');assert.equal(next.name,'My config');assert.equal(next.id,'saved')
 assert.equal(previous.api_key,'secret')
 assert.equal(applyPreset(next,'custom').base_url,next.base_url)
})
test('presets select actual protocols and scene labels distinguish Oriental from Touhou',()=>{
 const blank={base_url:'',protocol:'',model:'',api_key:'',has_key:false,clear_key:false,preset:'custom'}
 assert.equal(applyPreset(blank,'anthropic').protocol,'anthropic_compatible')
 assert.equal(applyPreset(blank,'openai_images').protocol,'openai_images')
 assert.equal(new Set(presets.map(p=>p.id)).size,presets.length)
 assert.equal(sceneNames.scene_01[0],'霓虹街道')
 assert.notEqual(sceneNames.scene_06[0],sceneNames.scene_08[0])
 assert.equal(Object.keys(sceneNames).length,10)
})
