"""Isolated bounded mixed-audio similarity worker, with networking disabled."""
import argparse
from pathlib import Path
import socket
from mdac2.application.manual_workflow import ManualWorkflow
from mdac2.application.structure_workflow import StructureWorkflow, producer, request_options
from mdac2.infrastructure.artifact_store import ArtifactStore
from mdac2.infrastructure.job_backend import JobBackend


def main():
    parser=argparse.ArgumentParser()
    for name in ('database','store','job','result'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args(); backend=JobBackend(args.database); store=ArtifactStore(args.store)
    request=backend.get_job(args.job).request.to_dict()
    if request['operation']!='structure_candidate' or request['producer']!=producer(): raise ValueError('Structure runtime identity changed')
    if request['options']!=request_options(request['project_id'],request['project_revision']): raise ValueError('Structure policy changed')
    flow=StructureWorkflow(ManualWorkflow(backend,store))
    if flow.context(request['project_id'],request['project_revision'])[1]!=request['inputs']: raise ValueError('Structure input identity changed')
    def forbidden(*args,**kwargs): raise RuntimeError('Analysis networking forbidden')
    socket.socket.connect=socket.socket.connect_ex=socket.create_connection=socket.getaddrinfo=forbidden
    result=flow.propose(request['project_id'],request['project_revision'])
    Path(args.result).write_bytes(result.to_json())


if __name__=='__main__': main()
