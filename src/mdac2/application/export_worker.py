"""Trusted child entrypoint; all compute and FFmpeg descendants belong to runner."""
import argparse
from pathlib import Path
from mdac2.adapters.audio_io import AudioDecoder
from mdac2.application.export_builder import build_export
from mdac2.application.manual_workflow import ManualWorkflow
from mdac2.infrastructure.artifact_store import ArtifactStore
from mdac2.infrastructure.job_backend import JobBackend


def main():
    parser=argparse.ArgumentParser()
    for name in ('database','store','ffmpeg','ffprobe','job','result'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    backend=JobBackend(args.database);store=ArtifactStore(args.store)
    request=backend.get_job(args.job).request.to_dict()
    if request['operation']!='export':raise ValueError('Unsupported worker operation')
    chart,timing=ManualWorkflow(backend,store).load(request['project_id'],request['project_revision'])
    result=build_export(backend,store,AudioDecoder(args.ffmpeg,args.ffprobe),project_id=request['project_id'],expected_revision=chart.revision_id,
        timing=timing.mapping,timing_revision=timing.revision_id,title=request['options']['title'],artist=request['options']['artist'])
    Path(args.result).write_bytes(result.to_json())


if __name__=='__main__':main()
