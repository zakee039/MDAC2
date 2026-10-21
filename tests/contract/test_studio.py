import json
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from mdac2.api.projects import create_app
from mdac2.api.studio import attach_studio
from mdac2.application.studio_workflow import StudioWorkflow
from mdac2.application.studio_metadata import MetadataLLM,opening_talk,NOTICE
from mdac2.infrastructure.job_backend import JobBackend
from mdac2.infrastructure.artifact_store import ArtifactStore

TOKEN='s'*40


def test_trim_tempo_map_persist_cache_identity_and_invalidate(context):
    import io
    import numpy as np
    import soundfile as sf
    from mdac2.application.studio_timing import settings
    c,s=context;data=io.BytesIO();sf.write(data,np.zeros((44100*4,2)),44100,format='WAV')
    p=c.post('/api/v1/studio/import',content=data.getvalue(),headers={'x-filename':'trim.wav'}).json()
    pid=p['project_id'];base='/api/v1/studio/projects/'+pid
    trim=dict(start_ms=500,end_ms=3500)
    timing=dict(settings(),enabled=True,offset_ms=-380,segments=[dict(start_ms=1500,bpm=90,offset_ms=1500,tap_latency_ms=20)])
    p=c.post(base+'/draft',json=dict(version=p['version'],metadata=p['metadata'],trim=trim,timing=timing)).json()
    again=StudioWorkflow(s.root,s.runtime,s.repository).get(pid)
    assert again['trim_draft']==trim and again['timing_draft']==timing
    p=c.post(base+'/confirm',json=dict(version=p['version'],metadata=p['metadata'],trim=trim,timing=timing)).json()
    assert p['trim']==trim and p['trim_draft'] is None
    a=s.enqueue(pid,p['version'],'generate');folder=s.directory/pid/a['attempt_id']
    request=json.loads((folder/'request.json').read_text())
    assert request['trim']==trim and request['timing']==timing and '-trim-' in request['cache']
    s.finish(a,'succeeded','fixture');p=s.get(pid)
    changed=c.post(base+'/confirm',json=dict(version=p['version'],metadata=p['metadata'],trim=dict(start_ms=500,end_ms=3400),timing=timing)).json()
    assert changed['generation'] is None and (folder/'request.json').exists()
    response=c.get(base+'/waveform?start=1&end=1.5')
    assert response.status_code==200 and response.json()['start']==1
    assert c.post(base+'/confirm',json=dict(version=changed['version'],metadata=p['metadata'],trim=dict(start_ms=0,end_ms=5000))).status_code==422
    assert c.post(base+'/draft',json=dict(version=1,metadata=p['metadata'],trim=trim)).status_code==409


def test_trim_detector_is_bounded_and_does_not_commit_suggestion(context,monkeypatch):
    import subprocess
    c,s=context;p=imported(c);base='/api/v1/studio/projects/'+p['project_id']
    def detect(command,**kwargs):
        assert kwargs['timeout']==120
        assert c.post(base+'/delete',json=dict(version=p['version'])).status_code==409
        Path(command[-1]).write_text(json.dumps(dict(candidate_ms=None,needs_review=True)))
    monkeypatch.setattr(subprocess,'run',detect)
    assert c.post(base+'/trim/detect').json()['candidate_ms'] is None
    assert s.get(p['project_id'])['version']==p['version']
    def timeout(*a,**k):raise subprocess.TimeoutExpired('analysis',120)
    monkeypatch.setattr(subprocess,'run',timeout)
    assert c.post(base+'/trim/detect').status_code==422
    assert s.timing_project is None and s.timing_lock.acquire(blocking=False)
    s.timing_lock.release()


def rated_fixture(s,p):
    """Complete the immutable output files omitted by stubbed worker fixtures."""
    from mdac2.data_preparation.audio_features import sha
    folder=s.directory/p['project_id']/p['generation'];folder.mkdir(exist_ok=True)
    (folder/'chart.json').write_text(json.dumps(dict(duration=3,events=[])))
    (folder/'result.json').write_text(json.dumps(dict(status='succeeded',chart_sha256=sha(folder/'chart.json'))))
    return s.confirm_difficulty(p['project_id'],p['version'],8.3)


def test_legacy_four_part_generation_is_preserved_until_explicit_confirmation(context):
    c,s=context;p=imported(c);pid=p['project_id']
    old=dict(enabled=True,bpm=120.,offset_ms=0.,tap_latency_ms=0.)
    # Simulate the persisted schema from the previous application version.
    with s.repository._connection() as db:
        p.update(timing=old,confirmed=True);s._write(db,p);db.commit()
    with pytest.raises(ValueError,match='12'):s.enqueue(pid,p['version'],'generate')
    assert s.get(pid)['timing']==old
    # Existing generated charts may still be exported with their four-part identity.
    with s.repository._connection() as db:
        p.update(generation='previous-chart');s._write(db,p);db.commit()
    p=rated_fixture(s,p)
    export=s.enqueue(pid,p['version'],'export')
    request=json.loads((s.directory/pid/export['attempt_id']/'request.json').read_text())
    assert request['timing']['subdivisions']==4
    s.finish(export,'cancelled','fixture');p=s.get(pid)
    p=s.confirm(pid,p['version'],p['metadata'],old)
    assert p['timing']['subdivisions']==12 and p['generation'] is None
    a=s.enqueue(pid,p['version'],'generate')
    assert json.loads((s.directory/pid/a['attempt_id']/'request.json').read_text())['timing']['subdivisions']==12


def test_timing_draft_persistence_request_and_invalidation(context):
    from mdac2.application.studio_timing import settings
    c,s=context;p=imported(c);pid=p['project_id'];base='/api/v1/studio/projects/'+pid
    timing=dict(settings(),enabled=True,bpm=137,offset_ms=-63,tap_latency_ms=80)
    p=c.post(base+'/draft',json=dict(version=p['version'],metadata=p['metadata'],timing=timing)).json()
    again=StudioWorkflow(s.root,s.runtime,s.repository)
    assert again.get(pid)['timing_draft']==timing
    p=c.post(base+'/confirm',json=dict(version=p['version'],metadata=p['metadata'],timing=timing)).json()
    assert p['timing']==timing and p['timing_draft'] is None
    a=s.enqueue(pid,p['version'],'generate')
    request=json.loads((s.directory/pid/a['attempt_id']/'request.json').read_text())
    assert request['timing']==timing
    s.finish(a,'succeeded','fixture');p=s.get(pid)
    p=rated_fixture(s,p)
    pack=s.enqueue(pid,p['version'],'export')
    assert c.post(base+'/run',json=dict(version=p['version'],operation='export')).status_code==409
    s.finish(pack,'succeeded','fixture');p=s.get(pid)
    changed=c.post(base+'/confirm',json=dict(version=p['version'],metadata=p['metadata'],timing=timing|{'offset_ms':10})).json()
    assert changed['generation'] is None and changed['export'] is None
    assert (s.directory/pid/a['attempt_id']/'request.json').exists()
    assert c.post(base+'/confirm',json=dict(version=changed['version'],metadata=p['metadata'],timing=timing|{'bpm':0})).status_code==422


def test_timing_audio_ranges_detection_stale_and_timeout(context,monkeypatch):
    import subprocess
    from mdac2.infrastructure.timing_models import TimingModels
    c,s=context;p=imported(c);pid=p['project_id'];base='/api/v1/studio/projects/'+pid
    response=c.get(base+'/audio',headers={'Range':'bytes=0-127'})
    assert response.status_code==206 and len(response.content)==128 and response.content[:4]==b'RIFF'
    assert response.headers['content-range'].startswith('bytes 0-127/')
    monkeypatch.setattr(TimingModels,'configuration',lambda *args:('python','fixture',{}))
    result=dict(bpm=120,offset_ms=12,beat_count=20,p95_ms=8,needs_review=True,variable_tempo_warning=False)
    def detect(command,**kwargs):
        assert kwargs['timeout']==180 and kwargs['cwd']==s.root
        assert c.post(base+'/delete',json=dict(version=s.get(pid)['version'])).status_code==409
        Path(command[-1]).write_text(json.dumps(result))
    monkeypatch.setattr(subprocess,'run',detect)
    response=c.post(base+'/timing/detect',json=dict(version=p['version']))
    assert response.status_code==200
    p=response.json();assert p['timing_analysis']==result and not p.get('timing',{}).get('enabled')
    def stale(command,**kwargs):
        s.save_draft(pid,p['version'],p['metadata']|{'title':'new draft'})
        Path(command[-1]).write_text(json.dumps(result|{'bpm':140}))
    monkeypatch.setattr(subprocess,'run',stale)
    assert c.post(base+'/timing/detect',json=dict(version=p['version'])).status_code==409
    assert s.get(pid)['timing_analysis']['bpm']==120 and s.timing_project is None
    def timeout(command,**kwargs):raise subprocess.TimeoutExpired(command,180)
    monkeypatch.setattr(subprocess,'run',timeout)
    response=c.post(base+'/timing/detect',json=dict(version=s.get(pid)['version']))
    assert response.status_code==422 and response.json()['code']=='TIMING_ANALYSIS'
    assert s.timing_lock.acquire(blocking=False);s.timing_lock.release()


def test_task_model_bindings_are_independent_and_persistent(context,monkeypatch):
    c,s=context
    base=dict(protocol='openai_compatible',base_url='http://127.0.0.1:45678/v1',api_key='')
    a=s.llm.save(base|dict(id='',name='Song profile',model='song-model'))
    b=s.llm.save(base|dict(id='',name='Cover profile',model='image-model',protocol='openai_images'))
    assert s.llm.public()['bindings']==dict(scene=a['id'],cover='')
    r=c.post('/api/v1/studio/llm/bind',json=dict(role='cover',id=b['id']))
    assert r.status_code==200 and r.json()['cover_configured']
    assert c.post('/api/v1/studio/llm/bind',json=dict(role='scene',id=b['id'])).status_code==422
    seen=[]
    monkeypatch.setattr(s.llm,'_complete',lambda config,*args:seen.append(config['model']) or json.dumps(dict(title='song',author='artist',scene='scene_01',franchise='other')))
    s.llm.select(b['id']);assert s.llm.suggest('song')['model']=='song-model'
    assert seen==['song-model']
    s.llm.save(base|dict(id=a['id'],name='Renamed song',model='song-model-2'))
    assert s.llm.load('cover')['model']=='image-model'
    assert MetadataLLM(s.llm.path).load('scene')['name']=='Renamed song'
    s.llm.delete(a['id']);assert not s.llm.public()['scene_configured']
    assert s.llm.load('cover')['id']==b['id']
    s.llm.save(base|dict(id=b['id'],name='Now text',model='text-model'))
    assert s.llm.public()['bindings']==dict(scene='',cover='')


def test_existing_profile_store_migrates_scene_binding_only(tmp_path):
    path=tmp_path/'settings.json'
    profile=MetadataLLM.blank()|dict(id='old',name='Old',model='text',base_url='https://example.com/v1',protocol='openai_compatible',key='opaque')
    path.write_text(json.dumps(dict(version=2,active_id='old',profiles=[profile])))
    s=MetadataLLM(path)
    assert s.public()['bindings']==dict(scene='old',cover='')
    assert s.load('scene')['key']=='opaque' and not s.load('cover')['key']


def test_assigned_image_transport_prompt_proxy_and_no_paid_probe(tmp_path,monkeypatch):
    import base64,httpx
    from mdac2.application import studio_imagegen as images
    root=Path(__file__).resolve().parents[2];seen=[];clients=[]
    raw=images.fallback('曲名','作者');original=httpx.Client
    def handler(request):
        seen.append(request)
        if request.method=='GET':return httpx.Response(200,json={'data':[{'id':'image-fixture'}]})
        return httpx.Response(200,json={'data':[{'b64_json':base64.b64encode(raw).decode()}]})
    def client(**kwargs):
        clients.append(kwargs.copy());kwargs.pop('proxy',None)
        return original(transport=httpx.MockTransport(handler),**kwargs)
    monkeypatch.setattr(httpx,'Client',client)
    s=MetadataLLM(tmp_path/'settings.json');p=s.save(dict(name='Image',protocol='openai_images',model='image-fixture',base_url='https://example.com/custom/v1',proxy_url='http://127.0.0.1:7890',api_key='fixture-key'))
    s.assign('cover',p['id'])
    assert s.probe()['image_probe'] and len(seen)==1 and seen[0].method=='GET'
    data,details=images.generate(root,s.load('cover'),'雪の歌','作者')
    assert data==raw and details['model']=='image-fixture'
    request=seen[-1];body=json.loads(request.content)
    assert str(request.url)=='https://example.com/custom/v1/images/generations'
    assert request.headers['authorization']=='Bearer fixture-key'
    assert body['model']=='image-fixture' and body['n']==1
    assert '雪の歌' in body['prompt'] and '作者' in body['prompt'] and 'DEFAULT STYLE' in body['prompt']
    assert 'COLOR PALETTE' not in body['prompt'] and 'response_format' not in body
    assert all(k['proxy']=='http://127.0.0.1:7890' and not k['trust_env'] and not k['follow_redirects'] for k in clients)


def test_generated_cover_circle_fallback_recrop_and_stale_result(context,monkeypatch):
    import io,shutil
    from PIL import Image
    from mdac2.application import studio_cover,studio_imagegen
    c,s=context;root=Path(__file__).resolve().parents[2]
    original_binary=studio_cover.binary
    monkeypatch.setattr(studio_cover,'binary',lambda _,name:original_binary(root,name))
    (s.root/'recipes').mkdir(exist_ok=True);shutil.copy(root/'recipes/cover-art-h-v1.md',s.root/'recipes/cover-art-h-v1.md')
    p=imported(c);pid=p['project_id'];base='/api/v1/studio/projects/'+pid
    r=c.post(base+'/cover/generate',json=dict(version=p['version'],metadata=p['metadata'],mode='llm'))
    assert r.status_code==200;r=r.json();cover=r['cover']
    assert cover['origin']=='local_text' and cover['fallback_reason']=='unconfigured'
    im=Image.open(s.cover_path(pid));assert im.size==(512,512) and im.getpixel((0,0))[3]==0 and im.getpixel((256,256))[3]==255
    assert Image.open(s.cover_path(pid,True)).mode=='RGB'
    # Repeated confirmation reuses the cover; cropping keeps its source and circle.
    assert s.generate_cover(pid,r['version'],r['metadata'])['version']==r['version']
    r=s.save_cover(pid,r['version'],crop=dict(x=.3,y=.6,zoom=1.2));assert r['cover']['shape']=='circle'
    configured=s.llm.save(dict(id='',name='Cover',model='image-choice',protocol='openai_images',base_url='https://example.com/v1'))
    s.llm.assign('cover',configured['id'])
    s.llm.save(dict(id='',name='Song',model='text-choice',protocol='openai_compatible',base_url='https://example.com/v1'))
    old_path=s.cover_path(pid);monkeypatch.setattr(studio_imagegen,'generate',lambda *args:(b'not an image',{}))
    r=s.generate_cover(pid,r['version'],r['metadata'],True)
    assert r['cover']['origin']=='local_text' and r['cover']['fallback_reason']=='generation_failed' and old_path.exists()
    valid=studio_imagegen.fallback('成功','作者')
    def selected(root,profile,*args):
        assert profile['id']==configured['id'] and profile['model']=='image-choice'
        return valid,dict(model='fixture',profile_id=profile['id'],style='H-v1')
    monkeypatch.setattr(studio_imagegen,'generate',selected)
    r=s.generate_cover(pid,r['version'],r['metadata'],True)
    assert r['cover']['origin']=='imagegen' and r['cover']['model']=='fixture'
    version=r['version']
    def racing(*args):
        s.save_draft(pid,version,r['metadata']|dict(title='Changed while generating'))
        return valid,dict(model='fixture')
    monkeypatch.setattr(studio_imagegen,'generate',racing)
    response=c.post(base+'/cover/generate',json=dict(version=version,metadata=r['metadata'],force=True,mode='llm'))
    assert response.status_code==200 and s.get(pid)['draft']['title']=='Changed while generating'
    assert response.json()['cover']['origin']=='imagegen'


def test_uploaded_cover_round_png_recrop_and_history(context,monkeypatch):
    import io
    from PIL import Image
    from mdac2.application import studio_cover
    c,s=context;root=Path(__file__).resolve().parents[2];binary=studio_cover.binary
    monkeypatch.setattr(studio_cover,'binary',lambda _,name:binary(root,name))
    p=imported(c);pid=p['project_id'];base='/api/v1/studio/projects/'+pid
    data=io.BytesIO();Image.new('RGB',(800,600),'red').save(data,format='JPEG')
    response=c.post(base+'/cover?version='+str(p['version']),content=data.getvalue())
    assert response.status_code==200
    p=response.json();original=s.cover_path(pid,True).read_bytes();first=s.cover_path(pid)
    identity=p['cover']['id']
    for crop in [None,dict(x=.2,y=.7,zoom=1.4)]:
        if crop:p=s.save_cover(pid,p['version'],crop=crop)
        assert p['cover']['shape']=='circle' and p['cover']['id']==identity
        im=Image.open(s.cover_path(pid))
        assert im.format=='PNG' and im.mode=='RGBA' and im.size==(512,512)
        assert all(im.getpixel(xy)[3]==0 for xy in [(0,0),(511,0),(0,511),(511,511)])
        assert im.getpixel((256,256))[3]==255
        assert s.cover_path(pid,True).read_bytes()==original
    again=StudioWorkflow(s.root,s.runtime,s.repository)
    selected=again.select_cover(pid,p['version'],identity)
    assert selected['cover']['shape']=='circle' and first.exists()
    assert Image.open(io.BytesIO(original)).size==(800,600)


def test_generated_asset_url_does_not_forward_credentials(monkeypatch):
    from mdac2.application import studio_imagegen as images
    root=Path(__file__).resolve().parents[2];calls=[]
    profile=MetadataLLM.blank()|dict(protocol='openai_images',base_url='https://example.com/v1',model='image')
    def request(p,method,url,**kwargs):
        calls.append((url,kwargs));return json.dumps({'data':[{'url':'https://cdn.example.com/generated.png'}]}).encode() if method=='POST' else b'image'
    monkeypatch.setattr(images,'request',request)
    monkeypatch.setattr(images.socket,'getaddrinfo',lambda *args,**kwargs:[(2,1,6,'',('93.184.216.34',443))])
    assert images.generate(root,profile,'Title','Artist')[0]==b'image'
    assert calls[-1][1]['authenticated'] is False
    monkeypatch.setattr(images.socket,'getaddrinfo',lambda *args,**kwargs:[(2,1,6,'',('127.0.0.1',443))])
    with pytest.raises(ValueError,match='不可用'):images.generate(root,profile,'Title','Artist')


def test_generated_asset_url_accepts_proxy_synthetic_dns(monkeypatch):
    from mdac2.application import studio_imagegen as images
    root=Path(__file__).resolve().parents[2];calls=[]
    profile=MetadataLLM.blank()|dict(protocol='openai_images',base_url='https://example.com/v1',model='image',proxy_url='http://127.0.0.1:7897')
    def request(p,method,url,**kwargs):
        calls.append((url,kwargs));return json.dumps({'data':[{'url':'https://cdn.example.com/generated.png'}]}).encode() if method=='POST' else b'image'
    monkeypatch.setattr(images,'request',request)
    monkeypatch.setattr(images.socket,'getaddrinfo',lambda *args,**kwargs:[(2,1,6,'',('198.18.1.14',443))])
    assert images.generate(root,profile,'Title','Artist')[0]==b'image'
    assert calls[-1][1]['authenticated'] is False

def audio_bytes():
    import io,wave,struct
    f=io.BytesIO()
    with wave.open(f,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(8000)
        w.writeframes(struct.pack('<h',1000)*800)
    return f.getvalue()

@pytest.fixture(autouse=True)
def media_binaries(monkeypatch):
    from mdac2.application.studio_cover import binary
    root=Path(__file__).resolve().parents[2]
    monkeypatch.setattr('mdac2.application.studio_media.binary',lambda _,name:binary(root,name))


@pytest.fixture
def context(tmp_path):
    repo=JobBackend(tmp_path/'state.sqlite');store=ArtifactStore(tmp_path/'store')
    for m in ('v2','v3','v4'):
        p=tmp_path/'models'/m;p.mkdir(parents=True);(p/'manifest.json').write_text('{}')
    for sid in ('r2','v2','v3','v4'):
        path=tmp_path/'recipes'/'strategies'/(sid+'.json');path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    studio=StudioWorkflow(tmp_path,tmp_path/'runtime',repo)
    app=create_app(repo,store,token=TOKEN,hosts=('127.0.0.1:8000',))
    attach_studio(app,studio)
    client=TestClient(app,base_url='http://127.0.0.1:8000',client=('127.0.0.1',111))
    client.headers['Authorization']='Bearer '+TOKEN
    return client,studio

def imported(client):
    response=client.post('/api/v1/studio/import',content=audio_bytes(),headers={'X-Filename':'song.mp3','Content-Type':'application/octet-stream'})
    assert response.status_code==201,response.text
    return response.json()

def test_auth_and_persistent_project(context):
    c,s=context
    assert c.get('/api/v1/studio',headers={'Authorization':'bad'}).status_code==401
    p=imported(c);assert p['metadata']['creator']=='mikee' and p['metadata']['strategy_id']=='v4'
    again=StudioWorkflow(s.root,s.runtime,s.repository)
    assert again.get(p['project_id'])['source_sha256']==p['source_sha256']
    assert Path(p['source']).read_bytes()==audio_bytes()
    assert again.list()[0]['project_id']==p['project_id']


def test_rating_confirmation_export_binding_and_regeneration(context):
    c,s=context;p=imported(c);pid=p['project_id'];base='/api/v1/studio/projects/'+pid
    p=s.confirm(pid,p['version'],p['metadata'])
    assert c.post(base+'/difficulty',json=dict(version=p['version'],value=8)).status_code==422
    a=s.enqueue(pid,p['version'],'generate');s.finish(a,'succeeded','fixture');p=s.get(pid)
    p=rated_fixture(s,p)
    assert p['difficulty_prediction']['value']==1 and p['difficulty']['value']=='8.3'
    reloaded=StudioWorkflow(s.root,s.runtime,s.repository).get(pid)
    assert reloaded['difficulty']==p['difficulty']
    assert c.post(base+'/difficulty',json=dict(version=p['version']-1,value=8)).status_code==409
    for invalid in (None,True,'NaN',21,9.555):
        assert c.post(base+'/difficulty',json=dict(version=p['version'],value=invalid)).status_code==422
    pack=s.enqueue(pid,p['version'],'export')
    request=json.loads((s.directory/pid/pack['attempt_id']/'request.json').read_text())
    assert request['difficulty']['value']=='8.3' and request['difficulty']['generation']==a['attempt_id']
    assert c.post(base+'/difficulty',json=dict(version=s.get(pid)['version'],value=9)).status_code==409
    s.finish(pack,'succeeded','fixture');p=s.get(pid)
    p=c.post(base+'/difficulty',json=dict(version=p['version'],value=9.2)).json()
    assert p['export'] is None and p['generation']==a['attempt_id']
    second=s.enqueue(pid,p['version'],'generate');s.finish(second,'succeeded','fixture');p=s.get(pid)
    assert p['difficulty'] is None
    # Build a new output, deliberately without confirming its level.
    rated_fixture(s,p)
    with s.repository._connection() as db:
        d=s._read(db,pid);d['difficulty']=None;s._write(db,d);db.commit()
    with pytest.raises(ValueError,match='确认'):s.enqueue(pid,d['version'],'export')
    p=s.confirm_difficulty(pid,d['version'],8.8)
    (s.directory/pid/second['attempt_id']/'chart.json').write_text('{}')
    assert s.get(pid)['difficulty_prediction'] is None
    with pytest.raises(ValueError,match='校验'):s.enqueue(pid,p['version'],'export')

def test_confirm_enqueue_duplicate_cancel_retry(context):
    c,s=context;p=imported(c);pid=p['project_id'];base='/api/v1/studio/projects/'+pid
    assert c.post(base+'/run',json=dict(version=1,operation='generate')).status_code==422
    p=c.post(base+'/confirm',json=dict(version=1,metadata=p['metadata'])).json()
    a=c.post(base+'/run',json=dict(version=p['version'],operation='generate')).json()
    assert a['state']=='queued'
    assert c.post(base+'/run',json=dict(version=p['version'],operation='generate')).status_code==409
    assert c.post(base+'/cancel').status_code==200
    assert (s.directory/pid/a['attempt_id']/'CANCEL').exists()
    s.finish(a,'cancelled','cancelled');p=s.get(pid)
    second=s.enqueue(pid,p['version'],'generate')
    assert second['attempt_id']!=a['attempt_id']
    assert len(s.get(pid)['attempts'])==2


def test_legacy_project_strategy_is_preserved_without_rewriting_on_read(context):
    c,s=context;p=imported(c);pid=p['project_id']
    p['metadata']['model']='v2';p['metadata'].pop('strategy_id')
    p['draft']=dict(p['metadata'],model='r2')
    with s.repository._connection() as db:
        s._write(db,p);db.commit()
        original=db.execute('SELECT data FROM studio_projects WHERE project_id=?',(pid,)).fetchone()[0]
    loaded=s.get(pid)
    assert loaded['metadata']['strategy_id']=='v2' and loaded['draft']['strategy_id']=='r2'
    assert s.list()[0]['metadata']['strategy_id']=='v2'
    with s.repository._connection() as db:
        assert db.execute('SELECT data FROM studio_projects WHERE project_id=?',(pid,)).fetchone()[0]==original
    confirmed=s.confirm(pid,p['version'],loaded['draft'])
    assert confirmed['metadata']['strategy_id']=='r2'
    task=s.enqueue(pid,confirmed['version'],'generate')
    request=json.loads((s.directory/pid/task['attempt_id']/'request.json').read_text())
    assert request['strategy_id']=='r2'

def test_metadata_change_invalidates_output_not_old_attempt(context):
    c,s=context;p=imported(c);pid=p['project_id'];p=s.confirm(pid,1,p['metadata'])
    a=s.enqueue(pid,p['version'],'generate');s.finish(a,'succeeded','done');p=s.get(pid)
    assert p['generation']==a['attempt_id']
    q=s.confirm(pid,p['version'],dict(p['metadata'],title='new'))
    assert q['generation'] is None and len(s.get(pid)['attempts'])==1

def test_llm_suggestion_does_not_overwrite_confirmed_info(context,monkeypatch):
    c,s=context;p=imported(c);pid=p['project_id'];p=s.confirm(pid,1,dict(p['metadata'],title='manual'))
    monkeypatch.setattr(s.llm,'suggest',lambda title:dict(title='suggested',author='A',scene='scene_10'))
    result=s.suggest(pid,p['version'])
    assert result['metadata']['title']=='manual' and result['suggestion']['title']=='suggested'
    assert result['confirmed']

def test_llm_stale_result_rejected(context,monkeypatch):
    c,s=context;p=imported(c);pid=p['project_id']
    def response(title):
        s.confirm(pid,1,dict(p['metadata'],title='changed'))
        return dict(title='stale',author='',scene='scene_01')
    monkeypatch.setattr(s.llm,'suggest',response)
    assert c.post('/api/v1/studio/projects/'+pid+'/suggest',json={'version':1}).status_code==409
    assert s.get(pid)['metadata']['title']=='changed'

@pytest.mark.parametrize('franchise,expected',[('miku','scene_10'),('touhou','scene_08')])
def test_llm_scene_priority(tmp_path,monkeypatch,franchise,expected):
    service=MetadataLLM(tmp_path/'settings.json')
    monkeypatch.setattr(service,'complete',lambda *args:json.dumps(dict(title='song',author='artist',scene='scene_01',franchise=franchise,reason='test')))
    assert service.suggest('song')['scene']==expected

@pytest.mark.parametrize('value',['[]','not json','{"title":"a\\n#BPM 900"}'])
def test_llm_rejects_invalid_output(tmp_path,monkeypatch,value):
    service=MetadataLLM(tmp_path/'settings.json');monkeypatch.setattr(service,'complete',lambda *args:value)
    with pytest.raises(ValueError):service.suggest('song')

@pytest.mark.skipif(os.name!='nt',reason='Windows credential encryption')
def test_key_encrypted_and_never_returned(context):
    c,s=context;secret='test-secret-not-a-real-key'
    data=dict(protocol='openai_compatible',base_url='https://example.invalid/v1',model='example',api_key=secret)
    response=c.post('/api/v1/studio/llm',json=data)
    assert response.status_code==200,response.text
    assert secret not in response.text and response.json()['has_key']
    assert secret not in s.llm.path.read_text()
    assert secret not in c.get('/api/v1/studio').text
    assert c.post('/api/v1/studio/llm',json=dict(data,api_key='')).json()['has_key']
    assert not c.post('/api/v1/studio/llm',json=dict(data,api_key='',clear_key=True)).json()['has_key']

def test_opening_dialog_schema_and_close():
    talk=opening_talk();assert talk['version']==2
    for lang in ('English','ChineseS','ChineseT','Japanese','Korean'):
        a,b=talk[lang];assert a['text']==NOTICE.replace('\n','<br>') and a['dialogState']==1 and b['dialogState']==2
        assert a['time']==0 and b['time']==4 and a['dialogIndex']==b['dialogIndex']
        assert a['dialogType']==b['dialogType']==2 and b['text']==''

def test_unsafe_upload_names_and_endpoint_config(context):
    c,s=context
    assert c.post('/api/v1/studio/import',content=b'x',headers={'X-Filename':'..%2Fescape.mp3'}).status_code==422
    assert c.post('/api/v1/studio/llm',json=dict(protocol='openai_compatible',base_url='https://host/v1/chat/completions',model='a')).status_code==422


@pytest.mark.parametrize('protocol,endpoint',[('openai_compatible','/custom/v1/chat/completions'),('anthropic_compatible','/custom/v1/messages')])
def test_real_http_metadata_transport(tmp_path,protocol,endpoint):
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    import threading
    seen=[]
    answer=json.dumps(dict(title='Snow Song Show',author='sasakure.UK, DECO*27',scene='scene_01',franchise='miku',reason='Miku'))
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append((self.path,body))
            response={'choices':[{'message':{'content':answer}}]} if protocol=='openai_compatible' else {'content':[{'type':'text','text':answer}]}
            raw=json.dumps(response).encode();self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        llm=MetadataLLM(tmp_path/'llm.json');llm.save(dict(protocol=protocol,base_url=f'http://127.0.0.1:{server.server_port}/custom/v1',model='fixture'))
        assert llm.suggest('Snow Song Show')['scene']=='scene_10'
        path,body=seen[0];assert path==endpoint
        assert json.loads(body['messages'][-1]['content'])=={'song_name':'Snow Song Show'}
        assert body['model']=='fixture'
        from mdac2.application.studio_scenes import AVAILABLE_SCENES
        system=body['messages'][0]['content'] if protocol=='openai_compatible' else body['system']
        assert json.dumps(AVAILABLE_SCENES,ensure_ascii=False) in system
        assert '尝试检索歌曲背景' in system and '不得声称已检索或编造来源' in system
    finally:server.shutdown();server.server_close();thread.join()


def test_studio_backup_restore_paths_and_package(tmp_path):
    from mdac2.application.backup import backup_workspace,restore_workspace
    from mdac2.data_preparation.audio_features import sha
    runtime=tmp_path/'runtime';runtime.mkdir();repo=JobBackend(runtime/'state.sqlite');store=ArtifactStore(runtime/'store')
    for model in ('v2','v3'):
        dest=tmp_path/'models'/model;dest.mkdir(parents=True);(dest/'manifest.json').write_text('{}')
    strategy=tmp_path/'recipes/strategies/v4.json';strategy.parent.mkdir(parents=True);strategy.write_text('{}')
    studio=StudioWorkflow(tmp_path,runtime,repo);source=tmp_path/'song.mp3';source.write_bytes(audio_bytes())
    p=studio.import_audio(source,'song.mp3');pid=p['project_id'];p=studio.confirm(pid,1,p['metadata'])
    a=studio.enqueue(pid,p['version'],'generate')
    with pytest.raises(ValueError,match='等待'):backup_workspace(repo,store,tmp_path/'active-backup')
    studio.finish(a,'succeeded','fixture');p=studio.get(pid)
    p=rated_fixture(studio,p)
    a=studio.enqueue(pid,p['version'],'export');folder=studio.directory/pid/a['attempt_id']
    (folder/'chart.mdm').write_bytes(b'package');(folder/'result.json').write_text(json.dumps({'package_sha256':sha(folder/'chart.mdm')}))
    studio.finish(a,'succeeded','fixture');studio.llm.path.write_text('do not backup credentials')
    backup_workspace(repo,store,tmp_path/'backup');restored=restore_workspace(tmp_path/'backup',tmp_path/'restored')
    new=StudioWorkflow(tmp_path,restored,JobBackend(restored/'state.sqlite'))
    assert Path(new.get(pid)['source']).is_relative_to(restored)
    assert Path(new.get(pid)['source']).read_bytes()==audio_bytes()
    assert new.package(pid)[0].read_bytes()==b'package'
    assert not new.llm.path.exists()


@pytest.mark.skipif(os.name!='nt',reason='Windows runtime lease')
def test_second_runtime_worker_cannot_start(context):
    _,studio=context
    second=StudioWorkflow(studio.root,studio.runtime,studio.repository)
    studio.start()
    try:
        with pytest.raises(RuntimeError,match='已有'):second.start()
    finally:studio.close()
    second.start();second.close()


def test_no_password_local_boundary(tmp_path):
    from mdac2.server import build_app
    app=build_app(Path(__file__).resolve().parents[2],runtime_root=tmp_path/'chart')
    c=TestClient(app,base_url='http://127.0.0.1:8000',client=('127.0.0.1',1000))
    assert c.get('/api/v1/studio').status_code==200
    assert c.get('/api/v1/studio',headers={'Origin':'https://untrusted.invalid'}).status_code==403
    assert c.post('/api/v1/studio/import',content=b'x',headers={'Sec-Fetch-Site':'cross-site','X-Filename':'x.mp3'}).status_code==403
    assert c.get('/api/v1/studio',headers={'Host':'attacker.invalid'}).status_code==403
    assert c.post('/api/v1/studio/preferences',json={'language':'ja'}).json()=={'language':'ja','delivery_folder':''}
    assert c.get('/api/v1/studio').json()['preferences']['language']=='ja'
    assert c.post('/api/v1/studio/preferences',json={'language':'xx'}).status_code==422


def test_permanent_delete_and_active_rejection(context):
    c,s=context;p=imported(c);base='/api/v1/studio/projects/'+p['project_id']
    p=s.confirm(p['project_id'],p['version'],p['metadata']);a=s.enqueue(p['project_id'],p['version'],'generate')
    assert c.post(base+'/delete',json={'version':s.get(p['project_id'])['version']}).status_code==409
    s.finish(a,'cancelled','fixture');p=s.get(p['project_id'])
    unrelated=s.directory/'cache';unrelated.mkdir();(unrelated/'keep').write_text('cache')
    assert c.post(base+'/delete',json={'version':p['version']-1}).status_code==409
    assert c.post(base+'/delete',json={'version':p['version']}).status_code==200
    assert not s.list() and not (s.directory/p['project_id']).exists()
    assert c.get(base).status_code==404 and (unrelated/'keep').exists()
    assert 'trash' not in c.get('/api/v1/studio').json()
    with s.repository._connection() as db:
        for table in ('projects','studio_projects','studio_attempts'):
            assert not db.execute(f'SELECT 1 FROM {table} WHERE project_id=?',(p['project_id'],)).fetchone()


def test_draft_survives_new_import_without_changing_old_generation(context):
    c,s=context;p=imported(c);pid=p['project_id'];p=s.confirm(pid,p['version'],p['metadata'])
    a=s.enqueue(pid,p['version'],'generate');s.finish(a,'succeeded','fixture');p=s.get(pid)
    draft=dict(p['metadata'],title='',author='unfinished edit')
    saved=c.post('/api/v1/studio/projects/'+pid+'/draft',json=dict(version=p['version'],metadata=draft)).json()
    assert c.post('/api/v1/studio/projects/'+pid+'/run',json=dict(version=saved['version'],operation='generate')).status_code==422
    new=imported(c)
    assert new['project_id']!=pid and len(s.list())==2
    old=s.get(pid);assert old['draft']==draft and old['generation']==a['attempt_id']
    assert old['metadata']==p['metadata'] and Path(old['source']).exists()
    again=StudioWorkflow(s.root,s.runtime,s.repository)
    assert again.get(pid)['draft']==draft
    confirmed=again.confirm(pid,saved['version'],dict(draft,title='new title'))
    assert confirmed['draft'] is None and confirmed['generation'] is None


def test_import_rejects_invalid_media_without_orphan_project(context):
    c,s=context
    for filename in ('bad.mp4','bad.txt'):
        r=c.post('/api/v1/studio/import',content=b'invalid',headers={'X-Filename':filename})
        assert r.status_code==422
    assert not s.list() and not list(s.directory.iterdir())


def test_delete_rejects_project_path_redirection(context,monkeypatch):
    c,s=context;p=imported(c)
    original=Path.resolve
    target=s.directory/p['project_id']
    def redirected(path,*args,**kwargs):
        return s.root.parent if path==target else original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'resolve',redirected)
    with pytest.raises(ValueError,match='escapes'):s.delete(p['project_id'],p['version'])
    assert Path(p['source']).exists() and s.get(p['project_id'])


def test_cover_normalization_crop_and_export_invalidation(tmp_path):
    import struct,zlib
    from mdac2.application.studio_cover import normalize
    from mdac2.application.studio_imagegen import circle
    root=Path(__file__).resolve().parents[2]
    # A 2:1 PNG exercises horizontal crop without depending on Pillow.
    def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    raw=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',100,50,8,2,0,0,0))+chunk(b'IDAT',zlib.compress((b'\0'+b'\xff\0\0'*50+b'\0\0\xff'*50)*50))+chunk(b'IEND',b'')
    left=normalize(root,raw,dict(x=0,y=.5,zoom=1));right=normalize(root,raw,dict(x=1,y=.5,zoom=1))
    assert struct.unpack('>II',left[16:24])==(512,512) and left!=right
    with pytest.raises(ValueError):normalize(root,raw,dict(x=2,y=0,zoom=1))
    runtime=tmp_path/'chart';repo=JobBackend(runtime/'state.sqlite');s=StudioWorkflow(root,runtime,repo)
    source=tmp_path/'song.mp3';source.write_bytes(audio_bytes());p=s.import_audio(source,'song.mp3');pid=p['project_id']
    p=s.confirm(pid,p['version'],p['metadata']);a=s.enqueue(pid,p['version'],'generate');s.finish(a,'succeeded','fixture');p=s.get(pid)
    p=s.save_cover(pid,p['version'],raw);assert p['generation']==a['attempt_id'] and p['export'] is None
    first=p['cover']['path'];p=s.save_cover(pid,p['version'],crop=dict(x=0,y=.5,zoom=1))
    assert Path(first).exists() and s.cover_path(pid).read_bytes()==circle(left)


def test_llm_named_profiles_migrate_edit_select_delete(context):
    from mdac2.application.studio_metadata import protect
    c,s=context
    secret='fixture-secret-only'
    legacy=dict(protocol='openai_compatible',base_url='https://example.invalid/v1',model='old',key=protect(secret))
    s.llm.path.parent.mkdir(parents=True,exist_ok=True)
    s.llm.path.write_text(json.dumps(legacy),encoding='utf-8')
    public=c.get('/api/v1/studio/llm').json()
    assert public['id']=='legacy' and public['has_key'] and len(public['profiles'])==1
    assert s.llm.load()['key']==legacy['key']
    assert secret not in json.dumps(public) and legacy['key'] not in json.dumps(public)
    fresh=dict(id='',name='Second',protocol='anthropic_compatible',base_url='https://example.invalid/v1',model='second',api_key='',proxy_url='')
    second=c.post('/api/v1/studio/llm',json=fresh).json()
    assert not second['has_key'] and second['id']!='legacy' and len(second['profiles'])==2
    changed=c.post('/api/v1/studio/llm',json=dict(fresh,id='legacy',name='Renamed',model='edited')).json()
    assert changed['has_key'] and changed['name']=='Renamed' and len(changed['profiles'])==2
    assert protect(s.llm.load()['key'],True)==secret
    assert c.post('/api/v1/studio/llm/select',json={'id':second['id']}).json()['model']=='second'
    assert MetadataLLM(s.llm.path).load()['id']==second['id']
    assert c.post('/api/v1/studio/llm',json=fresh).status_code==422
    assert c.post('/api/v1/studio/llm',json=dict(fresh,id='missing')).status_code==422
    assert c.post('/api/v1/studio/llm/delete',json={'id':'missing'}).status_code==422
    after=c.post('/api/v1/studio/llm/delete',json={'id':second['id']}).json()
    assert after['id']=='legacy' and after['has_key'] and len(after['profiles'])==1
    empty=c.post('/api/v1/studio/llm/delete',json={'id':'legacy'}).json()
    assert empty['profiles']==[] and not empty['configured']


def test_llm_draft_probe_never_saves_or_inherits_new_key(context,monkeypatch):
    c,s=context
    data=dict(id='',name='Saved',protocol='openai_compatible',base_url='https://example.invalid/v1',model='saved',api_key='fixture-key')
    saved=s.llm.save(data);before=s.llm.path.read_bytes();seen=[]
    def reply(config,*args):seen.append(config);return 'OK'
    monkeypatch.setattr(s.llm,'_complete',reply)
    draft=dict(data,id=saved['id'],model='unsaved',api_key='')
    assert c.post('/api/v1/studio/llm/probe',json=draft).status_code==200
    assert seen[-1]['model']=='unsaved' and seen[-1]['key']
    assert s.llm.path.read_bytes()==before
    assert c.post('/api/v1/studio/llm/probe',json=dict(draft,id='',name='New')).status_code==200
    assert not seen[-1]['key'] and s.llm.path.read_bytes()==before


@pytest.mark.parametrize('proxy',['socks5://127.0.0.1:7890','http://host:bad','http://user:pass@host:7890','http://host/path','not-a-url'])
def test_llm_proxy_validation(tmp_path,proxy):
    s=MetadataLLM(tmp_path/'llm.json')
    with pytest.raises(ValueError):s.save(dict(id='',name='P',protocol='openai_compatible',base_url='https://example.invalid/v1',model='test',proxy_url=proxy))
    assert not s.path.exists()


def test_llm_proxy_is_explicit_and_environment_is_ignored(tmp_path,monkeypatch):
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    import threading
    seen=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append((self.server.server_port,self.path,body['model']))
            raw=b'{"choices":[{"message":{"content":"OK"}}]}'
            self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    servers=[ThreadingHTTPServer(('127.0.0.1',0),Handler) for _ in range(2)]
    threads=[threading.Thread(target=s.serve_forever,daemon=True) for s in servers]
    for t in threads:t.start()
    try:
        origin,proxy=servers
        monkeypatch.setenv('HTTP_PROXY',f'http://127.0.0.1:{proxy.server_port}')
        monkeypatch.setenv('HTTPS_PROXY',f'http://127.0.0.1:{proxy.server_port}')
        monkeypatch.setenv('ALL_PROXY',f'http://127.0.0.1:{proxy.server_port}')
        monkeypatch.setenv('NO_PROXY','*')
        s=MetadataLLM(tmp_path/'llm.json')
        data=dict(id='',name='Direct',protocol='openai_compatible',base_url=f'http://127.0.0.1:{origin.server_port}/v1',model='direct')
        saved=s.save(data);assert s.probe()['ok']
        assert seen[-1][0]==origin.server_port and seen[-1][1]=='/v1/chat/completions'
        s.probe(dict(data,id=saved['id'],proxy_url=f'http://127.0.0.1:{proxy.server_port}',model='proxy'))
        assert seen[-1]==(proxy.server_port,data['base_url']+'/chat/completions','proxy')
        assert s.load()['proxy_url']=='' and s.load()['model']=='direct'
        proxy.shutdown();proxy.server_close();count=len(seen)
        with pytest.raises(ValueError,match='代理'):
            s.probe(dict(data,id=saved['id'],proxy_url=f'http://127.0.0.1:{proxy.server_port}'))
        assert len(seen)==count  # No fallback to the origin on proxy failure.
    finally:
        for server in servers:server.shutdown();server.server_close()
        for thread in threads:thread.join()


@pytest.mark.parametrize('suffix,codec',[('.mp4','aac'),('.mkv','flac'),('.mov','aac'),('.webm','libopus'),('.wma','wmav2')])
def test_real_video_and_audio_import_extracts_pcm(context,tmp_path,suffix,codec):
    import subprocess,soundfile as sf,numpy as np
    from mdac2.application.studio_cover import binary
    c,s=context;root=Path(__file__).resolve().parents[2];source=tmp_path/('original'+suffix)
    args=[str(binary(root,'ffmpeg.exe')),'-v','error','-nostdin','-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=0.4']
    if suffix in ('.mp4','.mkv','.mov','.webm'):
        args+=['-f','lavfi','-i','color=c=blue:s=32x32:r=10:d=0.4','-c:v','libvpx-vp9' if suffix=='.webm' else 'mpeg4']
    subprocess.run(args+['-c:a',codec,str(source)],check=True,capture_output=True)
    original=source.read_bytes();p=s.import_audio(source,source.name)
    assert source.read_bytes()==original and Path(p['source']).read_bytes()==original
    prepared=Path(p['prepared_audio']['path']);signal,rate=sf.read(prepared/'mix.wav',always_2d=True)
    assert rate==44100 and signal.shape[1]==2 and .35<len(signal)/rate<.6 and np.abs(signal).max()>.01
    assert sf.info(prepared/'mix.wav').subtype=='FLOAT'
    record=json.loads((prepared/'audio.json').read_text())
    assert record['source_sha256']==p['source_sha256']
    p=s.confirm(p['project_id'],p['version'],p['metadata']);a=s.enqueue(p['project_id'],p['version'],'generate')
    request=json.loads((s.directory/p['project_id']/a['attempt_id']/'request.json').read_text())
    assert request['prepared_audio']==p['prepared_audio']


def test_video_without_audio_rejected(context,tmp_path):
    import subprocess
    from mdac2.application.studio_cover import binary
    c,s=context;root=Path(__file__).resolve().parents[2];source=tmp_path/'silent.mp4'
    subprocess.run([str(binary(root,'ffmpeg.exe')),'-v','error','-f','lavfi','-i','color=s=32x32:d=0.2','-c:v','mpeg4',str(source)],check=True,capture_output=True)
    with pytest.raises(ValueError,match='音轨'):s.import_audio(source,source.name)
    assert source.exists() and not s.list() and not list(s.directory.iterdir())


def ncm_fixture(payload):
    """Synthetic encrypted container around generated audio, no copyrighted fixture."""
    import struct
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import pad
    from ncmdump import NeteaseCloudMusicFile
    from ncmdump.crypto import NCMRC4
    key=b'MDAC offline import fixture key'
    encrypted=AES.new(NeteaseCloudMusicFile.AES_KEY_RC4_KEY,AES.MODE_ECB).encrypt(pad(b'neteasecloudmusic'+key,16))
    encrypted=bytes(b^0x64 for b in encrypted)
    # Metadata and embedded cover are deliberately unused. Reserved image padding
    # exercises a case absent from the user's three files.
    metadata=b'https://example.invalid/cover-must-not-be-fetched'
    size=lambda n:struct.pack('<I',n)
    return b'CTENFDAM\0\0'+size(len(encrypted))+encrypted+size(len(metadata))+metadata+b'\0'*5+size(20)+size(3)+b'pic'+b'\0'*17+NCMRC4(key).decrypt(payload)


@pytest.mark.parametrize('kind,codec',[('flac','flac'),('mp3','libmp3lame')])
def test_ncm_upload_real_decode_and_worker_handoff(context,tmp_path,kind,codec):
    import subprocess,hashlib
    from urllib.parse import quote
    from mdac2.application.studio_cover import binary
    c,s=context;root=Path(__file__).resolve().parents[2];audio=tmp_path/('fixture.'+kind)
    subprocess.run([str(binary(root,'ffmpeg.exe')),'-v','error','-f','lavfi','-i','sine=frequency=440:duration=0.25','-c:a',codec,str(audio)],check=True,capture_output=True)
    data=ncm_fixture(audio.read_bytes())
    response=c.post('/api/v1/studio/import',content=data,headers={'X-Filename':quote('歌曲 初音☆.NCM')})
    assert response.status_code==201,response.text
    p=response.json();ready=Path(p['prepared_audio']['path']);record=json.loads((ready/'audio.json').read_text())
    assert Path(p['source']).read_bytes()==data and record['ncm']['format']==kind
    assert (ready/record['ncm']['file']).read_bytes()==audio.read_bytes()
    assert record['ncm']['sha256']==hashlib.sha256(audio.read_bytes()).hexdigest()
    assert record['source_sha256']==p['source_sha256'] and record['sample_rate']==44100
    assert .2<record['duration']<.4
    p=s.confirm(p['project_id'],p['version'],p['metadata']);attempt=s.enqueue(p['project_id'],p['version'],'generate')
    request=json.loads((s.directory/p['project_id']/attempt['attempt_id']/'request.json').read_text(encoding='utf-8'))
    assert request['prepared_audio']==p['prepared_audio'] and Path(request['source']).suffix=='.ncm'


def test_ncm_streaming_is_offline_and_preserves_payload(tmp_path,monkeypatch):
    from mdac2.application.studio_ncm import extract
    import urllib.request
    def forbidden(*args,**kwargs):raise AssertionError('No cover/metadata network access allowed')
    monkeypatch.setattr(urllib.request,'urlopen',forbidden)
    payload=b'fLaC'+bytes(range(256))*8193
    source=tmp_path/'input.ncm';source.write_bytes(ncm_fixture(payload));folder=tmp_path/'out';folder.mkdir()
    result=extract(source,folder)
    assert (folder/result['file']).read_bytes()==payload and result['bytes']==len(payload)
    assert result['policy']=='offline-audio-only-v1'


@pytest.mark.parametrize('data',[b'not ncm',b'CTENFDAM\0\0'+b'\xff'*4,b'CTENFDAM\0\0'+b'\x10\0\0\0'+b'\0'*16])
def test_invalid_ncm_rejected_and_project_cleaned(context,data):
    c,s=context
    response=c.post('/api/v1/studio/import',content=data,headers={'X-Filename':'broken.ncm'})
    assert response.status_code==422 and 'NCM' in response.json()['message']
    assert not s.list() and not list(s.directory.iterdir())


def test_ncm_decode_timeout_cleans_project(context,tmp_path,monkeypatch):
    import subprocess
    from mdac2.application import studio_ncm
    c,s=context;source=tmp_path/'timeout.ncm';source.write_bytes(b'original stays')
    def timeout(*args,**kwargs):raise subprocess.TimeoutExpired(args[0],kwargs['timeout'])
    monkeypatch.setattr(studio_ncm.subprocess,'run',timeout)
    with pytest.raises(ValueError,match='NCM解码超时'):s.import_audio(source,source.name)
    assert source.read_bytes()==b'original stays' and not s.list() and not list(s.directory.iterdir())


def test_manual_model_calls_and_local_cover(context,monkeypatch):
    from mdac2.application import studio_imagegen,studio_cover
    c,s=context;root=Path(__file__).resolve().parents[2]
    original_binary=studio_cover.binary
    monkeypatch.setattr(studio_cover,'binary',lambda _,name:original_binary(root,name))
    def forbidden(*args,**kwargs):raise AssertionError('Unexpected model invocation')
    monkeypatch.setattr(s.llm,'suggest',forbidden)
    monkeypatch.setattr(studio_imagegen,'generate',forbidden)
    p=imported(c);base='/api/v1/studio/projects/'+p['project_id']
    p=c.post(base+'/confirm',json=dict(version=p['version'],metadata=p['metadata'])).json()
    r=c.post(base+'/cover/generate',json=dict(version=p['version'],metadata=p['metadata']))
    assert r.status_code==422  # explicit choice required
    r=c.post(base+'/cover/generate',json=dict(version=p['version'],metadata=p['metadata'],mode='local',user_prompt='simple portrait'))
    assert r.status_code==200 and r.json()['cover']['origin']=='local_text'
    assert 'fallback_reason' not in r.json()['cover']
    p=r.json()
    r=c.post(base+'/draft',json=dict(version=p['version'],metadata=p['metadata'],cover_prompt='new draft prompt'))
    assert r.status_code==200
    again=StudioWorkflow(s.root,s.runtime,s.repository)
    assert again.get(p['project_id'])['cover_prompt']=='new draft prompt'
    p=r.json()
    assert c.post(base+'/draft',json=dict(version=p['version'],metadata=p['metadata'],cover_prompt='x'*2001)).status_code==422


def test_delivery_folder_persists_cancel_and_safe_filename(context,tmp_path,monkeypatch):
    from mdac2.application import studio_folder
    from mdac2.application.studio_metadata import package_filename
    from mdac2.data_preparation.audio_features import sha
    from urllib.parse import unquote
    c,s=context;p=imported(c);pid=p['project_id'];base='/api/v1/studio/projects/'+pid
    import zipfile
    def fake_package(path,marker):
        with zipfile.ZipFile(path,'w') as z:
            z.writestr('info.json',json.dumps({'difficulty2':'10'}));z.writestr('marker',marker)
    source=tmp_path/'package.mdm';fake_package(source,'first')
    p['metadata'].update(title='歌曲',creator='mikee')
    monkeypatch.setattr(s,'package',lambda _: (source,p))
    assert c.post(base+'/deliver').json()=={'folder_required':True}
    monkeypatch.setattr(studio_folder,'choose_folder',lambda:None)
    assert c.post('/api/v1/studio/delivery-folder/select').json()['cancelled']
    assert s.preferences()['delivery_folder']==''
    target=tmp_path/'Custom_Albums';target.mkdir()
    monkeypatch.setattr(studio_folder,'choose_folder',lambda:str(target))
    assert c.post('/api/v1/studio/delivery-folder/select').json()['preferences']['delivery_folder']==str(target.resolve())
    c.post('/api/v1/studio/preferences',json={'language':'en'})
    again=StudioWorkflow(s.root,s.runtime,s.repository)
    assert again.preferences()['delivery_folder']==str(target.resolve())
    monkeypatch.setattr(again,'package',lambda _:(source,p))
    result=again.deliver(pid);assert Path(result['path']).name=='[10.0]歌曲 - mikee.mdm'
    assert Path(result['path']).read_bytes()==source.read_bytes()
    assert again.deliver(pid)['path']==result['path'] and again.deliver(pid)['sha256']==sha(source)
    header=c.get(base+'/package').headers['content-disposition']
    assert '[10.0]歌曲 - mikee.mdm' in unquote(header)
    existing=Path(result['path']);existing.write_bytes(b'keep old chart')
    second=c.post(base+'/deliver').json()
    assert Path(second['path']).name=='[10.0]歌曲 - mikee v2.mdm' and existing.read_bytes()==b'keep old chart'
    fake_package(source,'third')
    assert Path(again.deliver(pid)['path']).name=='[10.0]歌曲 - mikee v3.mdm'
    assert Path(second['path']).read_bytes()!=source.read_bytes()
    assert not list(target.glob('.mdac-*'))
    name=package_filename(dict(creator='../a:b',title='../../bad\\name'),'9.5')
    assert '/' not in name and chr(92) not in name and ':' not in name and name.startswith('[9.5]')


def test_scene_vocalist_override_and_model_difficulty(context,monkeypatch):
    c,s=context;seen=[]
    monkeypatch.setattr(s.llm,'complete',lambda system,user:seen.append(system) or json.dumps(dict(title='Song',author='Composer',vocalist='初音ミク',franchise='other',scene='scene_01')))
    assert s.llm.suggest('Song')['scene']=='scene_10'
    assert 'Retrocity' in seen[0] and 'Jade Temple' in seen[0] and 'scene_06' in seen[0]
    data=c.get('/api/v1/studio').json()
    assert data['workflow_version']==12 and data['scenes']['scene_01']=='Space Station - [赛博, 硬核, 电子]'
    assert {m['strategy_id']:m['difficulty'] for m in data['models']}=={'r2':'9.5','v2':'10.5','v3':'9.5','v4':'9.5'}
    assert [m['strategy_id'] for m in data['models'] if m['default']]==['v4']
    p=imported(c);pid=p['project_id']
    p=s.confirm(pid,p['version'],p['metadata']|dict(strategy_id='v2'))
    a=s.enqueue(pid,p['version'],'generate')
    request=json.loads((s.directory/pid/a['attempt_id']/'request.json').read_text())
    assert request['metadata']['strategy_id']=='v2'


def test_cover_extra_prompt_is_sent_and_hashed(monkeypatch):
    import base64
    from mdac2.application import studio_imagegen as images
    root=Path(__file__).resolve().parents[2];bodies=[]
    profile=MetadataLLM.blank()|dict(protocol='openai_images',base_url='https://example.com/v1',model='image')
    def request(*args,**kwargs):
        bodies.append(kwargs['body']);return json.dumps({'data':[{'b64_json':base64.b64encode(b'image').decode()}]}).encode()
    monkeypatch.setattr(images,'request',request)
    _,first=images.generate(root,profile,'Song','Artist','simple umbrella')
    _,second=images.generate(root,profile,'Song','Artist','simple flower')
    assert 'simple umbrella' in bodies[0]['prompt'] and 'web browsing tools' in bodies[0]['prompt']
    assert first['prompt_sha256']!=second['prompt_sha256']
    assert first['user_prompt']=='simple umbrella'
    assert 'takes priority over the default art direction' in bodies[0]['prompt']
    assert 'follow it in preference to the defaults' in bodies[0]['prompt']


@pytest.mark.parametrize('model',['r2','v2','v3','v4'])
def test_all_studio_strategies_keep_weight_and_postprocess_identities(context,model):
    from mdac2.application.studio_models import STRATEGIES,pipeline_for,strategy_manifest_for,weight_manifest_for
    from mdac2.data_preparation.audio_features import sha
    c,s=context;p=imported(c);pid=p['project_id']
    options=c.get('/api/v1/studio').json()['models']
    assert [m['label'] for m in options]==['人声优先R2','综合V2','综合V3','综合V4']
    assert all(m['available'] for m in options)
    p=s.save_draft(pid,p['version'],p['metadata']|dict(strategy_id=model))
    p=s.confirm(pid,p['version'],p['draft'],dict(enabled=True,bpm=120,offset_ms=0,tap_latency_ms=0,subdivisions=12))
    a=s.enqueue(pid,p['version'],'generate')
    request=json.loads((s.directory/pid/a['attempt_id']/'request.json').read_text())
    assert request['metadata']['strategy_id']==model
    assert request['timing']['subdivisions']==12
    assert request['strategy_pipeline']==pipeline_for(model)
    assert request['weight_model']==STRATEGIES[model]['weight_model']
    assert request['weight_manifest_sha256']==sha(s.root/weight_manifest_for(STRATEGIES[model]['weight_model']))
    assert request['strategy_manifest_sha256']==sha(s.root/strategy_manifest_for(model))


def test_cover_catalog_keeps_generations_crops_and_selection_after_restart(context,monkeypatch):
    from mdac2.application import studio_cover,studio_imagegen
    c,s=context;p=imported(c);pid=p['project_id'];base='/api/v1/studio/projects/'+pid
    root=Path(__file__).resolve().parents[2];binary=studio_cover.binary
    monkeypatch.setattr(studio_cover,'binary',lambda _,name:binary(root,name))
    monkeypatch.setattr(studio_cover,'embedded',lambda *_:studio_imagegen.fallback('Embedded','Artist'))
    p=c.post(base+'/cover/options',json=dict(version=p['version'])).json()
    assert p['embedded_checked'] and len(p['cover_history'])==1 and not p.get('cover')
    embedded=p['cover_history'][0]['id']
    monkeypatch.setattr(studio_imagegen,'generate',lambda *args:(studio_imagegen.fallback(args[2],args[3]),dict(model='fixture',shape='circle')))
    p=s.generate_cover(pid,p['version'],dict(title='First',author=''),True)
    first=p['cover']['id'];first_hash=p['cover']['sha256']
    p=s.save_cover(pid,p['version'],crop=dict(x=.4,y=.6,zoom=1.2))
    cropped_hash=p['cover']['sha256'];assert cropped_hash!=first_hash and p['cover']['number']==1
    p=s.generate_cover(pid,p['version'],dict(title='Second',author=''),True)
    second=p['cover']['id'];assert p['cover']['number']==2
    p=s.generate_cover(pid,p['version'],dict(title='文字',author='作者'),mode='local')
    text=p['cover']['id'];assert len(p['cover_history'])==4
    again=StudioWorkflow(s.root,s.runtime,s.repository)
    p=again.select_cover(pid,p['version'],first)
    assert p['cover']['sha256']==cropped_hash and p['cover']['crop']['zoom']==1.2
    for chosen in [embedded,second,text]:
        p=again.select_cover(pid,p['version'],chosen);assert p['cover']['id']==chosen
    with pytest.raises(ValueError):again.select_cover(pid,p['version'],'../not-a-cover')
    with pytest.raises(Exception):again.select_cover(pid,p['version']-1,first)
    assert all(Path(item['path']).is_file() for item in p['cover_history'])


def test_cover_options_no_embedded_and_legacy_image_adoption(context,monkeypatch):
    from mdac2.application import studio_cover,studio_imagegen
    c,s=context;p=imported(c);pid=p['project_id']
    root=Path(__file__).resolve().parents[2];binary=studio_cover.binary
    monkeypatch.setattr(studio_cover,'binary',lambda _,name:binary(root,name))
    p=s.save_cover(pid,p['version'],studio_imagegen.fallback('Old',''),origin='imagegen')
    with s.repository._connection() as db:
        p.pop('cover_history');p['cover'].pop('id');p['cover'].pop('number');s._write(db,p);db.commit()
    calls=[];monkeypatch.setattr(studio_cover,'embedded',lambda *_:calls.append(True))
    p=s.prepare_covers(pid,p['version']);assert p['cover']['number']==1 and p['cover']['id'].startswith('legacy-')
    assert len(p['cover_history'])==1
    p=s.prepare_covers(pid,p['version']);assert len(calls)==1
    p=s.save_cover(pid,p['version'],crop=dict(x=.5,y=.5,zoom=1.1))
    assert p['cover']['number']==1 and len(p['cover_history'])==1


def test_image_failure_detail_is_safe_and_does_not_destroy_history(context,monkeypatch):
    import httpx
    from mdac2.application import studio_cover,studio_imagegen
    from mdac2.application.studio_llm_errors import service_error
    c,s=context;p=imported(c);pid=p['project_id']
    root=Path(__file__).resolve().parents[2];binary=studio_cover.binary
    monkeypatch.setattr(studio_cover,'binary',lambda _,name:binary(root,name))
    profile=s.llm.save(dict(name='Image',model='image',protocol='openai_images',base_url='https://example.com/v1'))
    s.llm.assign('cover',profile['id'])
    p=s.save_cover(pid,p['version'],studio_imagegen.fallback('Good',''),origin='imagegen')
    previous=p['cover']['id']
    def fail(*args):
        request=httpx.Request('POST','https://example.com/?secret=private')
        response=httpx.Response(429,request=request,json={'error':{'message':'private-key'}})
        try:response.raise_for_status()
        except httpx.HTTPStatusError as exc:raise service_error(exc,'封面服务请求')
    monkeypatch.setattr(studio_imagegen,'generate',fail)
    p=s.generate_cover(pid,p['version'],p['metadata'],True)
    assert '429' in p['cover']['fallback_message'] and 'private' not in p['cover']['fallback_message']
    assert p['cover']['origin']=='local_text'
    p=s.select_cover(pid,p['version'],previous);assert p['cover']['origin']=='imagegen'
