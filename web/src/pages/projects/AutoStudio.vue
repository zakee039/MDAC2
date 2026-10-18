<script setup lang="ts">
import {ref,computed,onMounted,onBeforeUnmount,watch,nextTick} from 'vue'
import {presets,applyPreset,sceneNames} from '../../studioOptions'
import {projectStage,exportAction} from '../../studioFlow'
import StudioBpm from '../../components/StudioBpm.vue'
import {defaultBeatSettings,upgradeBeatSettings,type BeatSettings} from '../../studioBpm'
import StudioIcon from '../../components/StudioIcon.vue'
import StudioDialog from '../../components/StudioDialog.vue'
import {language,t,type Language} from '../../studioI18n'
const props=defineProps<{settingsOpen:boolean;managerOpen:boolean}>()
const emit=defineEmits<{closeSettings:[];closeManager:[]}>()
const connected=ref(false),busy=ref(false),error=ref(''),notice=ref('')
const projects=ref<any[]>([]),models=ref<any[]>([]),scenes=ref<Record<string,string>>({}),project=ref<any>(null)
const stage=ref(1),fileInput=ref<HTMLInputElement|null>(null),dirty=ref(false),search=ref('')
const form=ref({title:'',author:'',creator:'mikee',scene:'scene_01',model:'v3'})
const blankLLM=()=>({id:'',name:'',protocol:'',base_url:'',model:'',api_key:'',proxy_url:'',has_key:false,preset:'custom',clear_key:false})
const llm=ref(blankLLM()),profiles=ref<ReturnType<typeof blankLLM>[]>([]),llmConfigured=ref(false)
const bindings=ref({scene:'',cover:''}),coverPrompt=ref(''),promptDirty=ref(false),deliveryFolder=ref('')
function choosePreset(event:Event){llm.value=applyPreset(llm.value,(event.target as HTMLSelectElement).value)}
function applyLLM(d:any,preserveEditor=false){if(!Array.isArray(d.profiles)||!d.bindings){connected.value=false;throw new Error(t('restartServer'))}profiles.value=d.profiles;bindings.value=d.bindings;llmConfigured.value=d.scene_configured;if(!preserveEditor)llm.value={id:d.id,name:d.name,protocol:d.protocol,base_url:d.base_url,model:d.model,proxy_url:d.proxy_url,has_key:d.has_key,api_key:'',preset:d.preset||'custom',clear_key:false}}
async function assignModel(role:'scene'|'cover',event:Event){const field=event.target as HTMLSelectElement;await act(async()=>applyLLM(await api('/llm/bind',{role,id:field.value}),true));field.value=bindings.value[role]}
async function chooseLLM(event:Event){const id=(event.target as HTMLSelectElement).value;if(!id){llm.value=blankLLM();error.value='';notice.value='';return}await act(async()=>applyLLM(await api('/llm/select',{id})))}
async function deleteLLM(){await act(async()=>{applyLLM(await api('/llm/delete',{id:llm.value.id}));notice.value=t('configDeleted')})}
const timing=ref<BeatSettings>(defaultBeatSettings())
const requests=ref<Record<string,{state:string;started:number;elapsed:number}>>({})
const now=ref(Date.now())
const statusClock=setInterval(()=>now.value=Date.now(),500)
function requestText(kind:string){const r=requests.value[kind];if(!r)return '';return t(r.state)+(r.state==='requestRunning'?' · '+Math.max(0,Math.floor((now.value-r.started)/1000))+' s':' · '+r.elapsed+' s')}
async function tracked(kind:string,fn:()=>Promise<unknown>){const started=Date.now();requests.value[kind]={state:'requestRunning',started,elapsed:0};try{const fallback=await fn();requests.value[kind]={state:fallback===true?'requestFallback':'requestDone',started,elapsed:Math.round((Date.now()-started)/1000)}}catch(e){requests.value[kind]={state:'requestFailed',started,elapsed:Math.round((Date.now()-started)/1000)};throw e}}
function updateTiming(value:BeatSettings){timing.value=value;dirty.value=true}
async function detectTiming(){await act(()=>tracked('timing',async()=>{await preserveCurrent();const p=await api('/projects/'+project.value.project_id+'/timing/detect',{version:project.value.version});project.value=p;timing.value={...timing.value,bpm:p.timing_analysis.bpm,offset_ms:p.timing_analysis.offset_ms};dirty.value=true}))}
const steps=['import','details','generate','export']
const image=ref<HTMLImageElement|null>(null),canvas=ref<HTMLCanvasElement|null>(null),cropOpen=ref(false),cropDirty=ref(false)
const crop=ref({x:.5,y:.5,zoom:1})
let timer:ReturnType<typeof setTimeout>|undefined,epoch=0,disposed=false
const visibleProjects=computed(()=>projects.value.filter(p=>(p.draft?.title||p.metadata.title).toLowerCase().includes(search.value.toLowerCase())))
const coverUrl=computed(()=>project.value?.cover?'/api/v1/studio/projects/'+project.value.project_id+'/cover?original=true&v='+project.value.cover.original_sha256:'')
function sceneName(key:string){return sceneNames[key]?.[language.value==='zh'?0:language.value==='en'?1:2]||scenes.value[key]}
async function api(path:string,body?:unknown,raw?:File){
 const r=await fetch('/api/v1/studio'+path,{method:body!==undefined||raw?'POST':'GET',headers:raw?{'Content-Type':'application/octet-stream','X-Filename':encodeURIComponent(raw.name)}:body!==undefined?{'Content-Type':'application/json'}:{},body:raw||(body===undefined?undefined:JSON.stringify(body)),redirect:'error',cache:'no-store'})
 if(!r.ok){let message=r.status===409?t('conflict'):t('error');if(language.value==='zh'){try{const data=await r.json();if(data.code?.startsWith('LLM')||data.code==='MEDIA_IMPORT'||data.code?.startsWith('DELIVERY')||data.code?.startsWith('TIMING'))message=data.message}catch{}}throw new Error(message)}
 return r.json()
}
async function act(fn:()=>Promise<void>){if(busy.value)return;busy.value=true;error.value='';notice.value='';try{await fn()}catch(e){error.value=(e as Error).message}finally{busy.value=false}}
async function refreshList(){const d=await api('');projects.value=d.projects;return d}
async function connect(){await act(async()=>{const d=await refreshList();if(d.workflow_version!==6)throw new Error(t('restartServer'));connected.value=true;models.value=d.models;scenes.value=d.scenes;applyLLM(d.llm);language.value=d.preferences.language;deliveryFolder.value=d.preferences.delivery_folder||''})}
function setProject(p:any){if(project.value?.project_id!==p.project_id)requests.value={};project.value=p;const upgraded=upgradeBeatSettings(p.timing_draft||p.timing);timing.value=upgraded.timing;form.value={...(p.draft||p.metadata)};dirty.value=!!p.draft||upgraded.needsConfirmation;if(upgraded.needsConfirmation)notice.value=t('gridUpgrade');crop.value=p.cover?.crop||{x:.5,y:.5,zoom:1};cropDirty.value=false;cropOpen.value=false;coverPrompt.value=p.cover_prompt??p.cover?.user_prompt??'';promptDirty.value=false}
async function open(p:any){await act(async()=>{await preserveCurrent();clearTimeout(timer);epoch++;setProject(await api('/projects/'+p.project_id));stage.value=dirty.value?2:projectStage(project.value);emit('closeManager');poll()})}
async function preserveCurrent(){
 if(!project.value)return
 clearTimeout(timer);epoch++
 try{
 if(cropDirty.value){project.value=await api('/projects/'+project.value.project_id+'/cover/crop',{version:project.value.version,crop:crop.value});cropDirty.value=false}
 if(dirty.value||promptDirty.value){project.value=await api('/projects/'+project.value.project_id+'/draft',{version:project.value.version,metadata:form.value,cover_prompt:coverPrompt.value,timing:timing.value});promptDirty.value=false;dirty.value=!!project.value.draft}
 }finally{poll()}
}
async function newProject(){await act(async()=>{await preserveCurrent();clearTimeout(timer);epoch++;project.value=null;dirty.value=false;cropDirty.value=false;stage.value=1;emit('closeManager')})}
async function remove(p:any){await act(async()=>{await api('/projects/'+p.project_id+'/delete',{version:p.version});if(project.value?.project_id===p.project_id){clearTimeout(timer);epoch++;project.value=null;stage.value=1;dirty.value=false;cropDirty.value=false}await refreshList()})}
function poll(){clearTimeout(timer);if(!project.value?.active_attempt||disposed)return;const id=project.value.project_id,stamp=epoch
 timer=setTimeout(async()=>{try{const p=await api('/projects/'+id);if(disposed||stamp!==epoch)return;project.value=p;if(!p.active_attempt){await refreshList();stage.value=projectStage(p)}poll()}catch(e){if(stamp!==epoch||disposed)return;error.value=(e as Error).message;poll()}},1500)}
async function suggestInternal(){const p=await api('/projects/'+project.value.project_id+'/suggest',{version:project.value.version});project.value=p;form.value={...form.value,title:p.suggestion.title,author:p.suggestion.author,scene:p.suggestion.scene};dirty.value=true;notice.value=t('suggested')}
async function upload(event:Event){const f=(event.target as HTMLInputElement).files?.[0];if(!f)return;await act(async()=>{await preserveCurrent();const p=await api('/import',undefined,f);epoch++;clearTimeout(timer);setProject(p);stage.value=2;await refreshList()});if(fileInput.value)fileInput.value.value=''}
async function generateCover(mode:'local'|'llm'){await preserveCurrent();project.value=await api('/projects/'+project.value.project_id+'/cover/generate',{version:project.value.version,metadata:{title:form.value.title,author:form.value.author},force:true,mode,user_prompt:coverPrompt.value});crop.value=project.value.cover.crop;cropDirty.value=false;notice.value=t(mode==='local'?'localCoverReady':project.value.cover.origin==='local_text'?'coverFallback':'saved');return mode==='llm'&&project.value.cover.origin==='local_text'}
async function confirm(){await act(async()=>{await preserveCurrent();setProject(await api('/projects/'+project.value.project_id+'/confirm',{version:project.value.version,metadata:form.value,timing:timing.value}));stage.value=3;await refreshList()})}
async function chooseChartModel(event:Event){const field=event.target as HTMLSelectElement;const model=field.value;await act(async()=>{setProject(await api('/projects/'+project.value.project_id+'/confirm',{version:project.value.version,metadata:{...form.value,model},timing:timing.value}));await refreshList()});field.value=form.value.model}

async function run(operation:string){await act(async()=>{await api('/projects/'+project.value.project_id+'/run',{version:project.value.version,operation});project.value=await api('/projects/'+project.value.project_id);stage.value=operation==='generate'?3:4;poll()})}
async function cancel(){await act(async()=>{await api('/projects/'+project.value.project_id+'/cancel',{});notice.value=t('cancelling')})}
async function saveLLM(probe=false){await act(async()=>{if(probe){const result=await api('/llm/probe',llm.value);notice.value=t(result.image_probe?'imageTestPassed':'testPassed')}else{applyLLM(await api('/llm',llm.value));notice.value=t('saved')}})}
async function chooseLanguage(event:Event){const old=language.value;language.value=(event.target as HTMLSelectElement).value as Language;await act(async()=>{try{await api('/preferences',{language:language.value})}catch(e){language.value=old;throw e}})}
async function selectFolder(){const r=await api('/delivery-folder/select',{});deliveryFolder.value=r.preferences.delivery_folder;return !r.cancelled}
async function deliver(){await act(async()=>{if(!deliveryFolder.value&&!await selectFolder())return;const r=await api('/projects/'+project.value.project_id+'/deliver',{});if(r.folder_required){deliveryFolder.value='';return}notice.value=t('delivered')})}

async function download(){await act(async()=>{const r=await fetch('/api/v1/studio/projects/'+project.value.project_id+'/package',{redirect:'error'});if(!r.ok)throw new Error(t('error'));const url=URL.createObjectURL(await r.blob());const a=document.createElement('a');a.href=url;const disposition=r.headers.get('content-disposition')||'';a.download=disposition.includes('filename*=')?decodeURIComponent(disposition.match(/filename\*=UTF-8''([^;]+)/i)?.[1]||'chart.mdm'):(disposition.match(/filename="([^"]+)"/i)?.[1]||'chart.mdm');a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)})}
const pendingChanges=computed(()=>dirty.value||cropDirty.value||promptDirty.value)
function canStep(i:number){return i===1||!!project.value&&(i===2||!pendingChanges.value&&(i===3&&project.value.confirmed||i===4&&exportAction(project.value,false)!=='blocked'))}
async function goStep(i:number){if(i!==4){stage.value=i;return}const action=exportAction(project.value,pendingChanges.value);if(action==='build')await run('export');else if(action==='show')stage.value=4}

function drawCover(){const im=image.value,cv=canvas.value;if(!im?.naturalWidth||!cv)return;const ctx=cv.getContext('2d');if(!ctx)return;const size=Math.min(im.naturalWidth,im.naturalHeight)/crop.value.zoom;ctx.clearRect(0,0,512,512);ctx.save();if(project.value?.cover?.shape==='circle'){ctx.beginPath();ctx.arc(256,256,256,0,Math.PI*2);ctx.clip()}ctx.drawImage(im,(im.naturalWidth-size)*crop.value.x,(im.naturalHeight-size)*crop.value.y,size,size,0,0,512,512);ctx.restore()}
async function uploadCover(e:Event){const field=e.target as HTMLInputElement,f=field.files?.[0];if(!f)return;await act(async()=>{if(f.size>8*1024*1024)throw new Error(t('coverLimit'));project.value=await api('/projects/'+project.value.project_id+'/cover?version='+project.value.version,undefined,f);crop.value=project.value.cover.crop;cropOpen.value=true;cropDirty.value=false;await nextTick();drawCover()});field.value=''}
async function saveCrop(){await act(async()=>{project.value=await api('/projects/'+project.value.project_id+'/cover/crop',{version:project.value.version,crop:crop.value});cropDirty.value=false;notice.value=t('saved')})}
watch(crop,()=>{drawCover()},{deep:true})
watch(coverUrl,async()=>{await nextTick();drawCover()})
watch(()=>props.managerOpen,async open=>{if(open&&connected.value)try{await refreshList()}catch(e){error.value=(e as Error).message}})
watch(()=>props.settingsOpen,async open=>{if(open&&connected.value)try{applyLLM(await api('/llm'))}catch(e){error.value=(e as Error).message}})
onMounted(connect)
onBeforeUnmount(()=>{clearInterval(statusClock);disposed=true;clearTimeout(timer);epoch++})
</script>

<template>
<section class="auto-studio">
 <nav id="workflow" class="steps" :aria-label="t('generate')"><button v-for="(key,i) in steps" :key="key" :class="{selected:stage===i+1}" :aria-current="stage===i+1?'step':undefined" :disabled="!connected||!canStep(i+1)||busy" @click="goStep(i+1)"><span>0{{i+1}}</span>{{t(key)}}</button></nav>
 <p v-if="error" role="alert">{{error}}</p><p v-if="notice" role="status" class="notice">{{notice}}</p><button v-if="!connected&&!busy" @click="connect">{{t('retryConnection')}}</button>
 <section v-show="stage===1" class="panel import-panel"><div class="import-icon"><StudioIcon name="upload" :size="32"/></div><h1>{{t('importTitle')}}</h1><p>{{t('importHint')}}</p><button class="primary" :disabled="!connected||busy" @click="fileInput?.click()">{{t('choose')}}</button><input ref="fileInput" hidden type="file" accept=".ncm,.mp3,.flac,.wav,.ogg,.m4a,.aac,.opus,.wma,.aif,.aiff,.alac,.ape,.mp4,.m4v,.mov,.mkv,.webm,.avi,.wmv,.flv,.mpeg,.mpg,.ts,.m2ts,.3gp" :aria-label="t('choose')" @change="upload"/><small>{{t('formats')}}</small><p v-if="busy" role="status">{{t('busy')}}</p></section>
 <template v-if="project">
  <div v-show="stage!==1" class="project-caption"><strong>{{project.metadata.title}}</strong><span>{{project.metadata.model.toUpperCase()}} · {{project.metadata.creator}}</span></div>
  <section v-show="stage===2" class="panel content details-page">
   <h2>{{t('details')}}</h2>
   <section class="detail-module" aria-labelledby="basic-heading">
    <div class="module-heading"><span>01</span><h3 id="basic-heading">{{t('basicInfo')}}</h3></div>
    <div class="request-row"><button :disabled="busy||!!project.active_attempt||!llmConfigured" @click="act(()=>tracked('metadata',suggestInternal))">{{t('llmFill')}}</button><span role="status" aria-live="polite">{{requestText('metadata')}}</span></div>
    <p class="module-hint">{{t(llmConfigured?'llmHint':'llmMissing')}}</p>
    <fieldset :disabled="busy||!!project.active_attempt"><div class="fields basic-fields">
     <label>{{t('title')}}<input v-model="form.title" required maxlength="256" @input="dirty=true"/></label>
     <label>{{t('artist')}}<input v-model="form.author" maxlength="256" :placeholder="t('unknown')" @input="dirty=true"/></label>
     <label>{{t('creator')}}<input v-model="form.creator" required maxlength="256" @input="dirty=true"/></label>
     <label>{{t('scene')}}<select v-model="form.scene" @change="dirty=true"><option v-for="(_,key) in scenes" :key="key" :value="key">{{sceneName(String(key))}}</option></select></label>
     <label>{{t('model')}}<select v-model="form.model" @change="dirty=true"><option v-for="m in models" :key="m.id" :value="m.id" :disabled="!m.available">{{m.label}} · {{m.difficulty}} · {{m.available?(m.default?t('default'):''):t('unavailable')}}</option></select></label>
    </div></fieldset>
   </section>
   <section class="detail-module" aria-labelledby="cover-heading">
    <div class="module-heading"><span>02</span><h3 id="cover-heading">{{t('cover')}}</h3></div>
    <div class="cover-layout">
     <div class="cover-art"><img v-if="coverUrl" ref="image" :src="coverUrl" :alt="t('cover')" hidden @load="drawCover"/><canvas v-if="coverUrl" ref="canvas" width="512" height="512" :aria-label="t('cover')" class="cover-preview"/><div v-else class="cover-placeholder"><StudioIcon name="wave" :size="50"/><span>{{t('noCover')}}</span></div></div>
     <div class="cover-controls">
      <p class="module-hint">{{t('coverHint')}}</p>
      <button :disabled="busy||!!project.active_attempt||!form.title.trim()||cropDirty" @click="act(()=>generateCover('local').then(()=>{delete requests.cover}))">{{t('localCover')}}</button>
      <div class="request-row"><button class="primary" :disabled="busy||!!project.active_attempt||!form.title.trim()||cropDirty" @click="act(()=>tracked('cover',()=>generateCover('llm')))">{{t('llmCover')}}</button><span role="status" aria-live="polite">{{requestText('cover')}}</span></div>
      <label>{{t('coverPrompt')}}<textarea v-model="coverPrompt" rows="3" maxlength="2000" :placeholder="t('coverPromptHint')" :disabled="busy||!!project.active_attempt" @input="promptDirty=true"/></label><small>{{t('coverResearchHint')}}</small>
      <small v-if="project.cover?.fallback_reason" class="fallback-note">{{t('coverFallback')}}</small>
      <label class="cover-upload">{{t('uploadCover')}}<input type="file" accept="image/png,image/jpeg,image/webp" :disabled="busy||!!project.active_attempt" @change="uploadCover"/></label><small>{{t('coverLimit')}}</small>
      <template v-if="coverUrl"><div><button :disabled="busy||!!project.active_attempt" @click="cropOpen=!cropOpen">{{t('crop')}}</button></div><fieldset v-if="cropOpen" :disabled="busy||!!project.active_attempt"><label>{{t('horizontal')}}<input v-model.number="crop.x" type="range" min="0" max="1" step="0.01" @input="cropDirty=true"/></label><label>{{t('vertical')}}<input v-model.number="crop.y" type="range" min="0" max="1" step="0.01" @input="cropDirty=true"/></label><label>{{t('zoom')}}<input v-model.number="crop.zoom" type="range" min="1" max="3" step="0.01" @input="cropDirty=true"/></label><button :disabled="!cropDirty" @click="saveCrop">{{t('cropSave')}}</button></fieldset></template>
     </div>
    </div>
   </section>
   <section class="detail-module" aria-labelledby="bpm-heading">
    <div class="module-heading"><span>03</span><h3 id="bpm-heading">{{t('bpmModule')}}</h3></div>
    <StudioBpm :active="stage===2&&!settingsOpen&&!managerOpen" :model-value="timing" :audio-url="'/api/v1/studio/projects/'+project.project_id+'/audio'" :blocked="busy||!!project.active_attempt" :status="requestText('timing')" :analysis="project.timing_analysis" @update:model-value="updateTiming" @detect="detectTiming"/>
   </section>
   <div class="details-footer"><button class="primary" :disabled="busy||!!project.active_attempt||cropDirty||!form.title.trim()||!form.creator.trim()" @click="confirm">{{t('confirm')}}</button></div>
  </section>
  <section v-show="stage===3" class="panel content"><h2>{{t('generate')}}</h2><p>{{t('generateHint')}}</p><label class="generation-model">{{t('model')}}<select :value="form.model" :disabled="busy||dirty||!!project.active_attempt" @change="chooseChartModel"><option v-for="m in models" :key="m.id" :value="m.id" :disabled="!m.available">{{m.label}} · {{m.difficulty}}</option></select></label><p v-if="dirty">{{t('dirty')}}</p><button class="primary" :disabled="busy||dirty||!!project.active_attempt||!project.confirmed" @click="run('generate')">{{t(project.generation?'retry':'start')}}</button><button class="primary" :disabled="busy||!!project.active_attempt||!project.generation||pendingChanges" @click="goStep(4)">{{t('nextPackage')}}</button><button v-if="project.active_attempt" :disabled="busy" @click="cancel">{{t('cancel')}}</button></section>
  <section v-show="stage===4" class="panel content"><h2>{{t('export')}}</h2>
   <p v-if="project.active_attempt" role="status">{{t('packageBuilding')}}</p>
   <button v-if="project.active_attempt" :disabled="busy" @click="cancel">{{t('cancel')}}</button>
   <template v-else-if="project.export"><p>{{t('packageReady')}}</p><button :disabled="busy" @click="deliver">{{t('deliver')}}</button><button class="primary" :disabled="busy" @click="download">{{t('download')}}</button></template>
   <template v-else><p>{{t('packageRetryHint')}}</p><button class="primary" :disabled="busy||!project.generation||pendingChanges" @click="run('export')">{{t('packageRetry')}}</button></template>
  </section>
  <section v-if="stage!==1&&project.attempts?.length" class="panel content history"><h3>{{t('history')}}</h3><ol><li v-for="a in [...project.attempts].reverse()" :key="a.attempt_id"><strong>{{t(a.operation==='generate'?'generate':'export')}}</strong><span>{{t(a.state)}}</span><small v-if="a.state==='running'">{{t(a.progress?.stage||'busy')}}</small></li></ol></section>
 </template>
 <StudioDialog :open="managerOpen" :title="t('projects')" @close="emit('closeManager')">
  <div class="section-title"><h2>{{t('projects')}}</h2><div class="manager-header-actions"><button class="primary" :disabled="busy" @click="newProject">{{t('newConfig')}}</button><button @click="emit('closeManager')">{{t('close')}}</button></div></div>
  <input v-model="search" class="search" :aria-label="t('search')" :placeholder="t('search')"/>
  <p v-if="!visibleProjects.length">{{t('empty')}}</p>
  <div class="project-list"><div v-for="p in visibleProjects" :key="p.project_id" class="project-row"><div><strong>{{p.draft?.title||p.metadata.title}}</strong><small>{{p.metadata.model.toUpperCase()}} · {{t(p.active_attempt?'running':p.draft?'saved':p.export?'export':p.generation?'succeeded':'saved')}}</small></div><button :disabled="busy" @click="open(p)">{{t('load')}}</button><button :disabled="busy||!!p.active_attempt" @click="remove(p)">{{t('delete')}}</button></div></div>
  <p v-if="error" role="alert">{{error}}</p>
 </StudioDialog>
 <StudioDialog :open="settingsOpen" :title="t('settings')" @close="emit('closeSettings')">
  <div class="section-title"><h2>{{t('settings')}}</h2><button @click="emit('closeSettings')">{{t('close')}}</button></div>
  <label class="language-select">{{t('language')}}<select :value="language" :disabled="busy||!connected" @change="chooseLanguage"><option value="zh">简体中文</option><option value="en">English</option><option value="ja">日本語</option></select></label>
  <div class="fields task-models">
   <label>{{t('sceneTask')}}<select :value="bindings.scene" :disabled="busy||!connected" @change="assignModel('scene',$event)"><option value="">{{t('chooseModel')}}</option><option v-for="p in profiles.filter(p=>p.protocol!=='openai_images')" :key="p.id" :value="p.id">{{p.name}}</option></select></label>
   <label>{{t('coverTask')}}<select :value="bindings.cover" :disabled="busy||!connected" @change="assignModel('cover',$event)"><option value="">{{t('chooseModel')}}</option><option v-for="p in profiles.filter(p=>p.protocol==='openai_images')" :key="p.id" :value="p.id">{{p.name}}</option></select></label>
  </div>
  <label class="delivery-setting">{{t('deliveryFolder')}}<span>{{deliveryFolder||t('folderFirstUse')}}</span><button :disabled="busy" @click="act(async()=>{await selectFolder()})">{{t('chooseFolder')}}</button></label><hr/><h3>{{t('llm')}}</h3>
  <fieldset :disabled="busy||!connected">
   <label>{{t('profile')}}<select :value="llm.id" @change="chooseLLM"><option value="">{{t('newConfig')}}</option><option v-for="p in profiles" :key="p.id" :value="p.id">{{p.name}}</option></select></label>
   <div class="fields">
    <label>{{t('protocol')}}<select v-model="llm.protocol" @change="llm.preset='custom'"><option disabled value="">{{t('chooseProtocol')}}</option><option value="openai_compatible">OpenAI compatible</option><option value="anthropic_compatible">Anthropic compatible</option><option value="openai_images">OpenAI Images compatible</option></select></label>
    <label>{{t('preset')}}<select :value="llm.preset" @change="choosePreset"><option v-for="p in presets" :key="p.id" :value="p.id">{{p.id==='custom'?t('customPreset'):p.name}}</option></select></label>
    <label>{{t('root')}}<input v-model="llm.base_url" @input="llm.preset='custom'" placeholder="https://example.com/v1" autocomplete="off"/></label>
    <label>{{t('key')}}<input v-model="llm.api_key" type="password" autocomplete="new-password" :placeholder="t(llm.has_key?'keySaved':'keyOptional')"/></label>
    <label>{{t('apiModel')}}<input v-model="llm.model" maxlength="256" autocomplete="off"/></label><label>{{t('proxy')}}<input v-model="llm.proxy_url" placeholder="http://127.0.0.1:7890" autocomplete="off"/><small>{{t('proxyHint')}}</small></label>
   </div>
   <div class="llm-actions"><label>{{t('saveName')}}<input v-model="llm.name" maxlength="128" autocomplete="off"/></label><div class="llm-buttons"><button :disabled="!llm.id" @click="deleteLLM">{{t('delete')}}</button><button @click="saveLLM(true)">{{t('test')}}</button><button class="primary" @click="saveLLM()">{{t('save')}}</button></div></div>
  </fieldset>
  <p v-if="busy" role="status">{{t('busy')}}</p><p v-if="error" role="alert">{{error}}</p><p v-if="notice" role="status">{{notice}}</p>
 </StudioDialog>
</section>
</template>
<style scoped>
.detail-module{border-top:1px solid #e1e8df;padding:26px 0}.module-heading{display:flex;align-items:center;gap:12px;margin-bottom:18px}.module-heading>span{font-size:12px;color:#779381}.module-heading h3{font-size:18px;margin:0}.request-row{display:flex;align-items:center;gap:14px;flex-wrap:wrap}.request-row>span{font-size:12px;color:#61756a}.module-hint{margin:0 0 12px}.basic-fields{grid-template-columns:repeat(2,minmax(0,1fr))}.cover-layout{display:grid;grid-template-columns:minmax(220px,2fr) minmax(0,3fr);gap:30px}.cover-art{align-self:start}.cover-preview,.cover-placeholder{max-width:100%!important}.cover-controls textarea{resize:vertical;width:100%;padding:12px;border:1px solid #dce5db;border-radius:7px;font:inherit;min-height:90px}.cover-controls>small{display:block;margin:8px 0}.fallback-note{color:#916b31}.details-footer{display:flex;justify-content:flex-end;border-top:1px solid #e1e8df;padding-top:14px}@media(max-width:760px){.cover-layout{grid-template-columns:1fr}.cover-art{max-width:360px;width:100%;justify-self:center}}@media(max-width:450px){.basic-fields{grid-template-columns:1fr}}

.details-layout form .fields{grid-template-columns:1fr}.llm-cover{margin:12px 0}.llm-cover textarea{resize:vertical;width:100%;padding:12px;border:1px solid #dce5db;border-radius:7px;font:inherit;min-height:80px}.generation-model{max-width:360px;margin:20px 0}.delivery-setting{margin-top:18px;overflow-wrap:anywhere}.delivery-setting button{align-self:flex-start}.cover-panel>button{display:block}
.task-models{padding:18px;background:#f4f8f1;border:1px solid #e1e8df;border-radius:10px}.manager-header-actions{display:flex;gap:10px}.section-title+.search{margin-top:24px}

.llm-actions{display:flex;align-items:flex-end;gap:18px}.llm-actions>label{flex:1;min-width:0}.llm-buttons{display:flex;gap:8px;flex-shrink:0}.llm-buttons button{min-height:42px;white-space:nowrap}.fields>label{min-width:0}@media(max-width:560px){.llm-actions{flex-direction:column;align-items:stretch}.llm-buttons{justify-content:flex-end}}

.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:0 0 24px}.steps button{padding:18px 10px;font-size:13px;gap:10px;background:#fff;white-space:nowrap}.steps button.selected{border-color:#87a979;background:#eff5eb;color:#276e59}.steps span{font-size:11px;opacity:.65}.content{padding:28px;margin:18px 0}.content h2{font-size:22px;margin:0 0 12px}.content p,.import-panel p{color:#61756a;line-height:1.8;font-size:13px}.content button{margin:8px 10px 8px 0}.import-panel{padding:64px 24px;text-align:center;min-height:360px;display:flex;align-items:center;flex-direction:column}.import-panel h1{font-size:26px;margin:22px 0 8px}.import-panel button{margin:20px 0 18px}.import-icon{width:70px;height:70px;background:#eef5e9;border-radius:20px;display:grid;place-items:center;color:#3c8a6b}.import-panel small,small{font-size:11px;color:#758479;line-height:1.8}.project-caption{display:flex;align-items:center;gap:20px;margin:24px 2px 12px;overflow-wrap:anywhere}.project-caption span{font-size:12px;color:#758479}.fields{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:22px 0}.fields .wide{grid-column:1/-1}label{display:flex;flex-direction:column;gap:8px;font-size:13px}input:not([type=checkbox]),select{padding:11px;border:1px solid #dce5db;border-radius:7px;background:white;min-width:0;width:100%}.details-layout{display:grid;grid-template-columns:minmax(0,2fr) minmax(260px,1fr);gap:36px}.cover-panel{padding-top:18px}.cover-panel h3{font-size:14px;margin:4px 0 12px}.cover-preview,.cover-placeholder{width:100%;aspect-ratio:1;max-width:400px;border-radius:10px;border:1px solid #e1e8df;background:#f2f6ee}.cover-placeholder{display:flex;align-items:center;justify-content:center;flex-direction:column;gap:14px;color:#779381;font-size:12px}.cover-panel p{font-size:12px;line-height:1.7}.cover-panel input[type=file]{font-size:11px;padding:8px}.cover-panel input[type=range]{padding:0;margin:5px 0}.cover-upload{margin:12px 0}.notice{padding:12px;border-radius:8px;background:#edf5e6;overflow-wrap:anywhere}.history h3{font-size:14px}.history ol{list-style:none;padding:0}.history li{display:flex;flex-wrap:wrap;gap:20px;padding:12px 0;border-top:1px solid #edf0eb;font-size:12px}.history small{flex-basis:100%}.manager-actions{display:flex;flex-wrap:wrap;gap:8px;margin:24px 0 16px}.chosen{background:#edf5e6}.search{margin-bottom:12px}.project-row{display:flex;align-items:center;gap:10px;padding:17px 0;border-bottom:1px solid #edf0eb}.project-row>div{flex:1;min-width:0}.project-row strong{display:block;overflow-wrap:anywhere;font-size:14px}.project-row small{display:block;margin-top:7px}.language-select{margin:26px 0}.check{flex-direction:row;align-items:center}fieldset{border:0;padding:0;min-width:0}hr{border:0;border-top:1px solid #e5ebe1;margin:26px 0}blockquote{margin:20px 0;padding:12px 20px;line-height:2;border-left:3px solid #98b77e;background:#f5f8f1;font-size:13px}.section-title h2{font-size:20px}.section-title+ p,dialog p{font-size:13px}.section-title~fieldset>button{margin:10px 12px 0 0}@media(max-width:760px){.steps{gap:5px}.steps button{padding:12px 5px;font-size:11px;flex-direction:column;gap:4px}.details-layout{grid-template-columns:1fr;gap:16px}.cover-panel{max-width:none}.content{padding:22px}.fields{gap:12px}.project-caption{flex-wrap:wrap;gap:8px}.import-panel{padding:42px 20px}.import-panel h1{font-size:22px}}@media(max-width:450px){.fields{grid-template-columns:1fr}.project-row{flex-wrap:wrap}.project-row>div{flex-basis:100%}}
</style>
