"""Local same-origin launcher. No models or remote services are started."""
import argparse
from contextlib import asynccontextmanager
import os
from pathlib import Path
import secrets
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from mdac2.adapters.audio_io import AudioDecoder
from mdac2.api.projects import create_app
from mdac2.api.assets import attach_assets
from mdac2.api.manual import attach_manual
from mdac2.api.jobs import attach_jobs
from mdac2.api.export import attach_export
from mdac2.application.manual_workflow import ManualWorkflow
from mdac2.application.export_runtime import ExportRuntime
from mdac2.infrastructure.artifact_store import ArtifactStore
from mdac2.infrastructure.job_backend import JobBackend
from mdac2.infrastructure.timing_models import TimingModels
from mdac2.application.timing_workflow import TimingWorkflow
from mdac2.api.timing import attach_timing
from mdac2.application.onset_workflow import OnsetWorkflow
from mdac2.api.onsets import attach_onsets
from mdac2.application.structure_workflow import StructureWorkflow
from mdac2.api.structure import attach_structure
from mdac2.application.studio_workflow import StudioWorkflow
from mdac2.api.studio import attach_studio


def build_app(root,token=None,port=8000,*,runtime_root=None):
    root=Path(root).absolute();runtime=Path(runtime_root) if runtime_root else root/'chart'
    backend=JobBackend(runtime/'state.sqlite');store=ArtifactStore(runtime/'store')
    binary=root/'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin'
    decoder=AudioDecoder(binary/'ffmpeg.exe',binary/'ffprobe.exe')
    app=create_app(backend,store,token=token,hosts=(f'127.0.0.1:{port}',f'localhost:{port}'),public_static=True)
    attach_assets(app,backend,store,decoder)
    workflow=ManualWorkflow(backend,store);attach_manual(app,workflow)
    models=TimingModels(root)
    attach_jobs(app,backend);worker=ExportRuntime(backend,store,decoder,runtime/'queue.sqlite',timing_models=models)
    attach_timing(app,TimingWorkflow(workflow),backend,models)
    attach_onsets(app,OnsetWorkflow(workflow),backend)
    attach_structure(app,StructureWorkflow(workflow),backend)
    attach_export(app,workflow,backend,store,decoder)
    studio=StudioWorkflow(root,runtime,backend);attach_studio(app,studio)
    @asynccontextmanager
    async def lifespan(app):
        studio.start()
        try:
            worker.start()
            try:yield
            finally:worker.close()
        finally:studio.close()
    app.router.lifespan_context=lifespan
    @app.get('/')
    def index():return FileResponse(root/'web/dist/index.html')
    app.mount('/assets',StaticFiles(directory=root/'web/dist/assets'),name='assets')
    return app


def main():
    import uvicorn
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='.');parser.add_argument('--port',type=int,default=8000);parser.add_argument('--runtime')
    args=parser.parse_args()
    app=build_app(args.root,None,args.port,runtime_root=args.runtime)
    print(f'Open http://127.0.0.1:{args.port} ; projects: {app.state.studio.runtime}',flush=True)
    uvicorn.run(app,host='127.0.0.1',port=args.port,access_log=False)


if __name__=='__main__':main()
