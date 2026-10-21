"""Authenticated local automatic workflow; bounded original-file imports."""
from pathlib import Path
import tempfile
from urllib.parse import unquote
from fastapi import Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse
from mdac2.api.security import identifier,json_body,error
from mdac2.application.studio_metadata import SCENES,package_filename
from mdac2.application.studio_scenes import AVAILABLE_SCENES
from mdac2.infrastructure.project_repository import RevisionConflict


def attach_studio(app,studio):
    app.state.studio=studio
    @app.get('/api/v1/studio')
    def index():return dict(workflow_version=12,projects=studio.list(),models=studio.models(),scenes=SCENES,scene_availability={key:key in AVAILABLE_SCENES for key in SCENES},llm=studio.llm.public(),preferences=studio.preferences())
    @app.post('/api/v1/studio/delivery-folder/select')
    async def select_delivery_folder():
        try:return await run_in_threadpool(studio.select_delivery_folder)
        except ValueError as exc:return error(422,'DELIVERY_FOLDER',str(exc))
    @app.post('/api/v1/studio/preferences')
    async def preferences(request:Request):
        b=await json_body(request);return studio.preferences(b['language'])
    @app.post('/api/v1/studio/projects/{pid}/delete')
    async def delete(pid:str,request:Request):
        b=await json_body(request);return await run_in_threadpool(studio.delete,identifier(pid),b['version'])
    @app.post('/api/v1/studio/import',status_code=201)
    async def upload(request:Request):
        name=unquote(request.headers.get('x-filename',''))
        if not name or '/' in name or '\\' in name:raise ValueError('文件名无效')
        limit=256*1024*1024;declared=request.headers.get('content-length')
        if declared and (not declared.isdigit() or int(declared)>limit):return error(413,'TOO_LARGE','音乐文件超过256MB')
        with tempfile.TemporaryDirectory(dir=studio.directory) as temporary:
            path=Path(temporary)/'upload';count=0
            with path.open('wb') as f:
                async for chunk in request.stream():
                    count+=len(chunk)
                    if count>limit:return error(413,'TOO_LARGE','音乐文件超过256MB')
                    await run_in_threadpool(f.write,chunk)
            if count==0 or (declared and count!=int(declared)):raise ValueError('上传未完成')
            try:return await run_in_threadpool(studio.import_audio,path,name)
            except ValueError as exc:return error(422,'MEDIA_IMPORT',str(exc))
    @app.get('/api/v1/studio/projects/{pid}')
    def project(pid:str):return studio.get(identifier(pid))
    @app.get('/api/v1/studio/projects/{pid}/audio')
    def audio(pid:str):return FileResponse(studio.audio_path(identifier(pid))[0],media_type='audio/wav')
    @app.get('/api/v1/studio/projects/{pid}/waveform')
    def waveform(pid:str,start:float=0,end:float|None=None):
        from mdac2.application.studio_trim import waveform as peaks
        return peaks(studio.audio_path(identifier(pid))[0],start,end)
    @app.post('/api/v1/studio/projects/{pid}/trim/detect')
    async def detect_trim(pid:str):
        try:return await run_in_threadpool(studio.detect_trim,identifier(pid))
        except ValueError as exc:return error(422,'TIMING_TRIM',str(exc))
    @app.post('/api/v1/studio/projects/{pid}/timing/detect')
    async def detect_timing(pid:str,request:Request):
        b=await json_body(request)
        try:return await run_in_threadpool(studio.detect_timing,identifier(pid),b['version'])
        except RevisionConflict:raise
        except ValueError as exc:return error(422,'TIMING_ANALYSIS',str(exc))
    @app.get('/api/v1/studio/projects/{pid}/cover')
    def cover(pid:str,original:bool=False):return FileResponse(studio.cover_path(identifier(pid),original),media_type='image/png')
    @app.post('/api/v1/studio/projects/{pid}/cover')
    async def upload_cover(pid:str,request:Request,version:int):
        data=bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data)>8*1024*1024:return error(413,'TOO_LARGE','封面图片最大8MB')
        return await run_in_threadpool(studio.save_cover,identifier(pid),version,bytes(data))
    @app.post('/api/v1/studio/projects/{pid}/cover/crop')
    async def crop_cover(pid:str,request:Request):
        b=await json_body(request)
        return await run_in_threadpool(studio.save_cover,identifier(pid),b['version'],crop=b['crop'])
    @app.post('/api/v1/studio/projects/{pid}/cover/options')
    async def cover_options(pid:str,request:Request):
        b=await json_body(request)
        return await run_in_threadpool(studio.prepare_covers,identifier(pid),b['version'])
    @app.post('/api/v1/studio/projects/{pid}/cover/select')
    async def select_cover(pid:str,request:Request):
        b=await json_body(request)
        return await run_in_threadpool(studio.select_cover,identifier(pid),b['version'],b['cover_id'])
    @app.post('/api/v1/studio/projects/{pid}/cover/generate')
    async def generate_cover(pid:str,request:Request):
        b=await json_body(request)
        if b.get('mode') not in ('local','llm'):raise ValueError('请选择本地或LLM生成')
        return await run_in_threadpool(studio.generate_cover,identifier(pid),b.get('version'),b['metadata'],b.get('force',False),b['mode'],b.get('user_prompt',''))
    @app.post('/api/v1/studio/projects/{pid}/draft')
    async def draft(pid:str,request:Request):
        b=await json_body(request);return studio.save_draft(identifier(pid),b['version'],b['metadata'],b.get('cover_prompt'),b.get('timing'),b.get('trim'))
    @app.post('/api/v1/studio/projects/{pid}/confirm')
    async def confirm(pid:str,request:Request):
        b=await json_body(request);return studio.confirm(identifier(pid),b['version'],b['metadata'],b.get('timing'),b.get('trim'))
    @app.post('/api/v1/studio/projects/{pid}/suggest')
    async def suggest(pid:str,request:Request):
        b=await json_body(request)
        try:return await run_in_threadpool(studio.suggest,identifier(pid),b['version'])
        except RevisionConflict:raise
        except ValueError as exc:return error(422,'LLM_ERROR',str(exc))
    @app.post('/api/v1/studio/projects/{pid}/run')
    async def run(pid:str,request:Request):
        b=await json_body(request);return studio.enqueue(identifier(pid),b['version'],b['operation'])
    @app.post('/api/v1/studio/projects/{pid}/difficulty')
    async def difficulty(pid:str,request:Request):
        b=await json_body(request)
        try:return studio.confirm_difficulty(identifier(pid),b['version'],b['value'])
        except RevisionConflict:raise
        except ValueError as exc:return error(422,'DIFFICULTY',str(exc))
    @app.post('/api/v1/studio/projects/{pid}/cancel')
    def cancel(pid:str):return studio.cancel(identifier(pid))
    @app.get('/api/v1/studio/projects/{pid}/package')
    def package(pid:str):
        path,d=studio.package(identifier(pid));return FileResponse(path,filename=studio.package_name(path,d),media_type='application/zip')
    @app.post('/api/v1/studio/projects/{pid}/deliver')
    def deliver(pid:str):
        try:return studio.deliver(identifier(pid))
        except ValueError as exc:return error(422,'DELIVERY_ERROR',str(exc))
    @app.get('/api/v1/studio/llm')
    def llm():return studio.llm.public()
    @app.post('/api/v1/studio/llm')
    async def settings(request:Request):
        b=await json_body(request)
        try:return await run_in_threadpool(studio.llm.save,b)
        except ValueError as exc:return error(422,'LLM_CONFIG',str(exc))
    @app.post('/api/v1/studio/llm/probe')
    async def probe(request:Request):
        b=await json_body(request)
        try:return await run_in_threadpool(studio.llm.probe,b)
        except ValueError as exc:return error(422,'LLM_CONNECTION',str(exc))

    @app.post('/api/v1/studio/llm/select')
    async def select_llm(request:Request):
        b=await json_body(request)
        try:return await run_in_threadpool(studio.llm.select,b.get('id'))
        except ValueError as exc:return error(422,'LLM_CONFIG',str(exc))
    @app.post('/api/v1/studio/llm/bind')
    async def bind_llm(request:Request):
        b=await json_body(request)
        try:return await run_in_threadpool(studio.llm.assign,b.get('role'),b.get('id'))
        except ValueError as exc:return error(422,'LLM_CONFIG',str(exc))
    @app.post('/api/v1/studio/llm/delete')
    async def delete_llm(request:Request):
        b=await json_body(request)
        try:return await run_in_threadpool(studio.llm.delete,b.get('id'))
        except ValueError as exc:return error(422,'LLM_CONFIG',str(exc))
